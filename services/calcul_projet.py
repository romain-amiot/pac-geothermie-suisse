from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from models import ProjectInputs
from Fonctions.puissance_pointe import calcul_pointe_et_energie_annuelle
from Fonctions.rentabilite import (
    gshp_capex_net_after_subsidy,
    couts_annuels_gshp_zuberi,
    load_price_path_electricity_ttc_by_canton,
    load_price_path_fuel,
    payback_discounted_from_cashflows,
    project_requires_reference_system,
    annual_om_cost_chf,
)
from Fonctions.climat import (
    cooling_hours_for_postcode,
    cooling_degree_hours_for_postcode,
    nearest_station_for_postcode,
)
from Fonctions.localisation import postcode_info


BASE_DIR = Path(__file__).resolve().parents[1]

SCEN_ELEC_CSV = str(BASE_DIR / "Data" / "processed" / "scenarios_electricity_ttc_by_canton.csv")
SCEN_GAS_CSV = str(BASE_DIR / "Data" / "processed" / "scenarios_gas.csv")
SCEN_OIL_CSV = str(BASE_DIR / "Data" / "processed" / "scenarios_mazout.csv")

COOLING_QREF_CALIBRATED_CSV = BASE_DIR / "Tools" / "ajuster_coef_froid" / "cooling_qref_calibrated_weighted.csv"
CLIMATE_STATION_REFERENCE_CSV = BASE_DIR / "Tools" / "ajuster_coef_froid" / "climate_station_reference.csv"

CO2_FACTORS_G_PER_KWH: dict[str, float] = {
    "Électricité (réseau)": 90,
    "Gaz naturel": 230,
    "Mazout (fioul)": 324,
}

BUILDING_TYPE_PRICE_MAP = {
    "maison_individuelle": "maison_individuelle",
    "residentiel_collectif": "appartement",
    "grand_batiment_compact": "appartement",
    "mixte": "appartement",
    "activites": "maison_individuelle",
    "equipement_collectif": "maison_individuelle",
    "bureaux": "appartement",
    "ecole": "maison_individuelle",
    "commerce": "maison_individuelle",
}

CENTRAL_SCENARIO = {"label": "Central", "electricity": "neutral", "fuel": "neutral"}

PRICE_SENSITIVITY_SCENARIOS = {
    "Électricité basse": {
        "electricity": "optimistic",
        "fuel": "neutral",
        "description": "Test de sensibilité : prix de l'électricité plus bas que le cas central.",
    },
    "Électricité haute": {
        "electricity": "pessimistic",
        "fuel": "neutral",
        "description": "Test de sensibilité : prix de l'électricité plus haut que le cas central.",
    },
    "Énergie actuelle basse": {
        "electricity": "neutral",
        "fuel": "optimistic",
        "description": "Test de sensibilité : prix de l'énergie actuelle plus bas que le cas central.",
    },
    "Énergie actuelle haute": {
        "electricity": "neutral",
        "fuel": "pessimistic",
        "description": "Test de sensibilité : prix de l'énergie actuelle plus haut que le cas central.",
    },
    "Différentiel favorable PAC": {
        "electricity": "optimistic",
        "fuel": "pessimistic",
        "description": "Électricité plus basse et énergie actuelle plus haute.",
    },
    "Différentiel défavorable PAC": {
        "electricity": "pessimistic",
        "fuel": "optimistic",
        "description": "Électricité plus haute et énergie actuelle plus basse.",
    },
}


# ============================================================
# FROID
# ============================================================

COOLING_QREF_FALLBACK_BY_TYPOLOGY_AND_CLIMATE = {
    "residentiel": {"froid": 2.0, "tempere": 4.0, "chaud": 7.0, "tres_chaud": 12.0, "global": 5.0, "reference": 5.0},
    "bureaux": {"froid": 14.0, "tempere": 25.0, "chaud": 40.0, "tres_chaud": 60.0, "global": 38.0, "reference": 38.0},
    "ecole": {"froid": 8.0, "tempere": 15.0, "chaud": 25.0, "tres_chaud": 40.0, "global": 20.0, "reference": 20.0},
    "commerce": {"froid": 25.0, "tempere": 45.0, "chaud": 65.0, "tres_chaud": 90.0, "global": 60.0, "reference": 60.0},
    "equipement_collectif": {"froid": 20.0, "tempere": 35.0, "chaud": 50.0, "tres_chaud": 70.0, "global": 45.0, "reference": 45.0},
    "activites": {"froid": 6.0, "tempere": 12.0, "chaud": 18.0, "tres_chaud": 30.0, "global": 16.0, "reference": 16.0},
    "mixte": {"froid": 10.0, "tempere": 20.0, "chaud": 35.0, "tres_chaud": 55.0, "global": 30.0, "reference": 30.0},
}

COOLING_SPF_BY_MODE = {"free_cooling": 12.0, "hybrid": 8.0, "active_cooling": 5.0}
F_VITRAGE = {"faible": 0.90, "moyen": 1.00, "fort": 1.12}
F_SOLAIRE = {"bonne": 0.82, "moyenne": 1.00, "faible": 1.18}
F_USAGE = {"faible": 0.90, "normal": 1.00, "eleve": 1.15}
F_NIGHT_BASE = {True: 0.90, False: 1.00}
CDH_REF = 1600.0


# ============================================================
# GÉOMÉTRIE
# ============================================================

def estimate_sdep_vh_from_typology(
    *,
    shab_m2: float,
    building_type_key: str,
    hauteur_m: float,
) -> tuple[float, float, dict[str, Any]]:
    """
    Estime Sdép et Vh à partir de la typologie calibrée.

    La surface de déperdition est estimée par :
        Sdép = K_typologique * Shab

    Les coefficients K proviennent de l'étude typologique réalisée
    sur les bâtiments genevois. Ils intègrent donc déjà, en moyenne,
    les effets de compacité, de forme générale, d'allongement et de
    morphologie du bâtiment.

    Le volume chauffé reste estimé par :
        Vh = Shab * hauteur_m
    """
    shab = _safe_float(shab_m2, None)
    hauteur = _safe_float(hauteur_m, None)

    if shab is None or shab <= 0:
        raise ValueError("shab_m2 doit être > 0.")
    if hauteur is None or hauteur <= 0:
        raise ValueError("hauteur_m doit être > 0.")

    key = _norm(building_type_key)

    if key not in K_SDEP_BY_BUILDING_TYPE:
        raise ValueError(
            "Typologie inconnue pour l'estimation typologique calibrée : "
            f"{building_type_key!r}. Typologies disponibles : "
            f"{sorted(K_SDEP_BY_BUILDING_TYPE.keys())}"
        )

    k_typologique = float(K_SDEP_BY_BUILDING_TYPE[key])

    sdep_est = k_typologique * float(shab)
    vh_est = float(shab) * float(hauteur)

    meta = {
        "mode": "typologie_calibree_geneve",
        "building_type_key": key,
        "k_typologique": k_typologique,
        "shab_m2": float(shab),
        "hauteur_m": float(hauteur),
        "sdep_m2": float(sdep_est),
        "vh_m3": float(vh_est),
    }

    return float(sdep_est), float(vh_est), meta


# Coefficients typologiques calibrés à partir de l'étude des bâtiments genevois.
# Ils intègrent en moyenne les effets de compacité, de forme générale,
# d'allongement et de morphologie du bâtiment.
K_SDEP_BY_BUILDING_TYPE = {
    "maison_individuelle": 2.10,
    "residentiel_collectif": 1.92,
    "grand_batiment_compact": 1.71,
    "mixte": 1.69,
    "activites": 1.99,
    "equipement_collectif": 2.15,
}

# Ancienne méthode géométrique détaillée.
# Conservée pour compatibilité, mais elle ne doit plus être utilisée dans
# l'interface finale si l'on choisit l'approche typologique calibrée.
FORM_RATIO_BY_SHAPE = {"carre": 1.0, "rectangulaire": 1.5, "allonge": 2.5, "irregulier": 1.8}
MITOYENNETE_FACTOR = {"isole": 1.00, "1_cote": 0.75, "2_cotes": 0.50, "3_cotes": 0.25}

# Valeurs par défaut encore utiles pour l'interface, notamment la hauteur.
# Valeurs de hauteur sous plafond par défaut retenues pour le calcul typologique.
# Elles servent à préremplir l'interface et comme hauteurs de référence pour
# corriger la partie verticale de Sdép.
#
# Justification synthétique :
# - résidentiel : 2.50 m ;
# - mixte : 2.70 m ;
# - activités / tertiaire : 2.75 m ;
# - équipement collectif / bâtiment public : 3.00 m.
DEFAULT_GEOMETRY_BY_BUILDING_TYPE = {
    "maison_individuelle": {"forme_generale": "rectangulaire", "mitoyennete": "isole", "hauteur_m": 2.50, "toiture_exposee": True, "plancher_expose": True},
    "residentiel_collectif": {"forme_generale": "rectangulaire", "mitoyennete": "isole", "hauteur_m": 2.50, "toiture_exposee": True, "plancher_expose": True},
    "grand_batiment_compact": {"forme_generale": "carre", "mitoyennete": "isole", "hauteur_m": 2.50, "toiture_exposee": True, "plancher_expose": True},
    "mixte": {"forme_generale": "rectangulaire", "mitoyennete": "isole", "hauteur_m": 2.70, "toiture_exposee": True, "plancher_expose": True},
    "activites": {"forme_generale": "rectangulaire", "mitoyennete": "isole", "hauteur_m": 2.75, "toiture_exposee": True, "plancher_expose": True},
    "equipement_collectif": {"forme_generale": "rectangulaire", "mitoyennete": "isole", "hauteur_m": 3.00, "toiture_exposee": True, "plancher_expose": True},
}

_QREF_CACHE: pd.DataFrame | None = None
_CLIMATE_REF_CACHE: pd.DataFrame | None = None


# ============================================================
# INCERTITUDE
# ============================================================

UNCERTAINTY_N_SIMS = 5000
UNCERTAINTY_PROJECT_LIFETIME_YEARS = 25
UNCERTAINTY_PAYBACK_MAX_YEARS = 50
UNCERTAINTY_DISCOUNT_RATE = 0.03

UNCERTAINTY_DEFAULTS = {
    "heating_need_sigma": 0.12,
    "cooling_need_sigma": 0.30,
    "heating_need_min": 0.80,
    "heating_need_max": 1.30,
    "cooling_need_min": 0.50,
    "cooling_need_max": 2.00,
    "capex_min": 0.88,
    "capex_mode": 1.00,
    "capex_max": 1.12,
    "spf_heat_min": 0.90,
    "spf_heat_mode": 1.00,
    "spf_heat_max": 1.10,
    "spf_cool_min": 0.90,
    "spf_cool_mode": 1.00,
    "spf_cool_max": 1.10,
    "price_elec_sigma": 0.107,
    "price_fuel_sigma": 0.040,
    "om_min": 0.80,
    "om_mode": 1.00,
    "om_max": 1.20,
    "capex_heat_need_elasticity": 0.00,
    "om_heat_need_elasticity": 0.10,
}

COOLING_SIGMA_BY_TYPOLOGY = {
    "bureaux": 0.22,
    "ecole": 0.30,
    "residentiel": 0.40,
    "commerce": 0.28,
    "activites": 0.35,
    "equipement_collectif": 0.35,
    "mixte": 0.35,
}


# Risque spécifique aux énergies fossiles pour le Monte Carlo.
#
# Les paramètres ci-dessous sont calibrés à partir des séries historiques
# utilisées dans le mémoire :
# - gaz naturel : Prix_gaz_Energie360.csv ;
# - mazout : Prix_mazout_Midland.csv.
#
# Méthode de calibration :
# 1) agrégation des prix en moyennes annuelles ;
# 2) calcul des rendements logarithmiques annuels r_t = ln(P_t/P_{t-1}) ;
# 3) identification d'une année de choc si r_t > moyenne(r) + écart-type(r) ;
# 4) probabilité de choc = fréquence historique des années de choc ;
# 5) amplitude de choc = hausses observées pendant les années de choc.
#
# annual_drift est volontairement fixé à 0.0 afin de ne pas doubler la tendance
# déjà contenue dans les trajectoires centrales de prix p_ref_life / p_ref_payback.
# Le multiplicateur Monte Carlo représente donc l'incertitude autour du scénario
# central, et non une deuxième trajectoire prospective.
FOSSIL_RISK_PREMIUM = {
    "Gaz naturel": {
        # Le drift est fixé à zéro afin de ne pas doubler la tendance déjà
        # contenue dans la trajectoire centrale de prix.
        "annual_drift": 0.0,

        # Volatilité résiduelle annualisée, estimée sur les années hors choc,
        # puis utilisée dans un processus mean-reverting. La valeur n'est donc
        # pas la volatilité historique brute appliquée en marche aléatoire.
        # Incertitude persistante sur le niveau futur du gaz par rapport au
        # scénario central. Elle élargit la distribution sans imposer de biais
        # haussier : la médiane du facteur systémique vaut 1.
        "systematic_sigma_log": 0.14,

        # Volatilité résiduelle annualisée, estimée sur les années hors choc,
        # puis utilisée dans un processus mean-reverting. Elle est volontairement
        # plus faible que la volatilité historique brute pour éviter de compter
        # deux fois les chocs.
        "annual_sigma_residual": 0.10,
        "mean_reversion_phi": 0.70,

        # Fréquence historique : 2 années de choc sur 25 rendements annuels.
        "shock_probability": 0.08,

        # Amplitudes inspirées des chocs historiques observés. Le choc 2022 du
        # gaz est très extrême ; on le garde dans la borne haute mais pas comme
        # valeur centrale.
        "shock_min": 0.15,
        "shock_mode": 0.35,
        "shock_max": 0.85,

        # Choc temporaire avec retour progressif vers la trajectoire centrale.
        "shock_duration_min": 1,
        "shock_duration_max": 3,
        "shock_decay": 0.50,

        "min_multiplier": 0.65,
        "max_multiplier": 2.25,
        "calibration_source": "Prix_gaz_Energie360.csv",
        "calibration_method": (
            "rendements logarithmiques annuels ; choc si r_t > moyenne + 1 écart-type ; "
            "volatilité résiduelle hors choc utilisée dans un processus mean-reverting"
        ),
        "shock_years": [2008, 2022],
        "historical_shock_probability": 0.08,
        "historical_full_sigma_log": 0.188,
        "historical_nonshock_sigma_log": 0.123,
    },
    "Mazout (fioul)": {
        "annual_drift": 0.0,

        # Paramètres remis sur les valeurs historiques brutes de calibration.
        #
        # Calibration sur Prix_mazout_Midland.csv :
        # - 30 rendements logarithmiques annuels exploitables ;
        # - années de choc détectées avec r_t > moyenne + écart-type :
        #   2000, 2005, 2008 et 2022 ;
        # - fréquence historique : 4 / 30 = 13.3 %/an ;
        # - amplitudes excédentaires observées après retrait de la tendance
        #   moyenne : environ +32 %, +33 %, +57 % et +58 %.
        #
        # Cette version ne recentre pas les chocs et ne réduit pas
        # arbitrairement leur amplitude. Elle sert à tester l'effet complet des
        # chocs historiques, maintenant que le bug d'alignement des séries
        # mazout/électricité est corrigé.
        "systematic_sigma_log": 0.22,
        "annual_sigma_residual": 0.176,
        "mean_reversion_phi": 0.70,

        "shock_probability": 0.133,

        # Amplitude historique excédentaire des années de choc.
        "shock_min": 0.32,
        "shock_mode": 0.45,
        "shock_max": 0.58,

        # Durée temporaire du choc, comme dans le modèle fossile initial.
        "shock_duration_min": 1,
        "shock_duration_max": 3,
        "shock_decay": 0.50,

        # Pas de recentrage : on veut observer l'effet brut des paramètres
        # historiques après correction de l'alignement des prix.
        "center_shocks_log": False,

        "min_multiplier": 0.75,
        "max_multiplier": 2.80,
        "calibration_source": "Prix_mazout_Midland.csv",
        "calibration_method": (
            "rendements logarithmiques annuels ; choc si r_t > moyenne + 1 écart-type ; "
            "fréquence historique brute 4/30 = 13.3 %/an ; amplitudes historiques "
            "excédentaires min/médiane/max environ 32 %, 45 % et 58 %"
        ),
        "shock_years": [2000, 2005, 2008, 2022],
        "historical_shock_probability": 0.133,
        "effective_shock_probability": 0.133,
        "historical_full_sigma_log": 0.222,
        "historical_nonshock_sigma_log": 0.176,
        "historical_excess_shock_amplitudes": [0.320, 0.332, 0.574, 0.583],
    },
}
FOSSIL_ENERGY_LABELS = set(FOSSIL_RISK_PREMIUM.keys())


# ============================================================
# OUTILS GÉNÉRAUX
# ============================================================

def g_per_kwh_to_kg_per_kwh(g_per_kwh: float) -> float:
    return g_per_kwh / 1000.0


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .strip()
        .lower()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("ù", "u")
        .replace("ç", "c")
        .replace("-", "_")
        .replace(" ", "_")
    )


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        if isinstance(value, str):
            value = value.strip().replace(",", ".")
            if value == "":
                return default
        val = float(value)
        if pd.isna(val):
            return default
        return val
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _get_price_building_type(building_type_key: str) -> str:
    key = _norm(building_type_key)
    return BUILDING_TYPE_PRICE_MAP.get(key, "appartement")


def _normalize_factors_to_median_one(values: list[float]) -> list[float]:
    import numpy as np
    if not values:
        return values
    med = float(np.median(values))
    if med <= 0:
        return values
    return [float(v) / med for v in values]


def normalize_calibration_typology(building_type_key: str) -> str:
    key = _norm(building_type_key)
    mapping = {
        "maison_individuelle": "residentiel",
        "residentiel_collectif": "residentiel",
        "grand_batiment_compact": "residentiel",
        "residentiel": "residentiel",
        "logement": "residentiel",
        "logements": "residentiel",
        "mixte": "mixte",
        "activites": "activites",
        "activite": "activites",
        "equipement_collectif": "equipement_collectif",
        "equipement": "equipement_collectif",
        "bureaux": "bureaux",
        "bureau": "bureaux",
        "office": "bureaux",
        "ecole": "ecole",
        "enseignement": "ecole",
        "commerce": "commerce",
    }
    return mapping.get(key, key if key else "mixte")


def normalize_climate_class(climate_class: str) -> str:
    key = _norm(climate_class)
    mapping = {
        "froid": "froid",
        "cold": "froid",
        "tempere": "tempere",
        "temperate": "tempere",
        "chaud": "chaud",
        "hot": "chaud",
        "tres_chaud": "tres_chaud",
        "treschaud": "tres_chaud",
        "very_hot": "tres_chaud",
        "global": "global",
        "reference": "reference",
    }
    return mapping.get(key, key if key else "tempere")


def climate_class_from_cdh(cooling_degree_hours: float) -> str:
    cdh = _safe_float(cooling_degree_hours, 0.0) or 0.0
    if cdh < 250:
        return "froid"
    if cdh < 650:
        return "tempere"
    if cdh < 1100:
        return "chaud"
    return "tres_chaud"


def station_name_from_station_info(station_info: Any) -> str:
    if station_info is None:
        return ""
    if isinstance(station_info, dict):
        for key in ["station_name", "name", "nom", "station", "ville"]:
            if key in station_info and station_info[key]:
                return str(station_info[key])
    for attr in ["station_name", "name", "nom", "station", "ville"]:
        if hasattr(station_info, attr):
            val = getattr(station_info, attr)
            if val:
                return str(val)
    return ""


def _lognormal_factor(rng: Any, sigma: float) -> float:
    return float(rng.lognormal(mean=0.0, sigma=sigma))


def _bounded_lognormal_factor(rng: Any, sigma: float, low: float, high: float) -> float:
    return _clamp(_lognormal_factor(rng, sigma), low, high)


def _triangular_mean(left: float, mode: float, right: float) -> float:
    return (float(left) + float(mode) + float(right)) / 3.0


def _expected_log_shock_size(fossil_cfg: dict) -> float:
    """
    Approximation de E[log(1 + choc)] pour recentrer les chocs.

    On utilise une approximation par grille plutôt que log(1 + moyenne), car
    le choc résiduel du mazout peut être légèrement négatif.
    """
    import numpy as np

    def approx_triangular_log_mean(left: float, mode: float, right: float) -> float:
        # Approximation déterministe suffisamment précise pour le recentrage.
        xs = np.linspace(float(left), float(right), 401)
        left_f, mode_f, right_f = float(left), float(mode), float(right)

        # Densité triangulaire.
        density = np.zeros_like(xs)
        if mode_f > left_f:
            mask = (xs >= left_f) & (xs <= mode_f)
            density[mask] = 2 * (xs[mask] - left_f) / ((right_f - left_f) * (mode_f - left_f))
        if right_f > mode_f:
            mask = (xs > mode_f) & (xs <= right_f)
            density[mask] = 2 * (right_f - xs[mask]) / ((right_f - left_f) * (right_f - mode_f))

        if density.sum() <= 0:
            return math.log1p(max(left_f, -0.95))

        vals = np.log1p(np.clip(xs, -0.95, None))
        area_num = float(np.trapezoid(vals * density, xs)) if hasattr(np, "trapezoid") else float(sum((vals[:-1] * density[:-1] + vals[1:] * density[1:]) * (xs[1:] - xs[:-1]) / 2.0))
        area_den = float(np.trapezoid(density, xs)) if hasattr(np, "trapezoid") else float(sum((density[:-1] + density[1:]) * (xs[1:] - xs[:-1]) / 2.0))
        if area_den <= 0:
            return math.log1p(max(left_f, -0.95))
        return float(area_num / area_den)

    p_extreme = float(fossil_cfg.get("extreme_shock_probability", 0.0))

    moderate = approx_triangular_log_mean(
        fossil_cfg["shock_min"],
        fossil_cfg["shock_mode"],
        fossil_cfg["shock_max"],
    )

    if p_extreme > 0:
        extreme = approx_triangular_log_mean(
            fossil_cfg["extreme_shock_min"],
            fossil_cfg["extreme_shock_mode"],
            fossil_cfg["extreme_shock_max"],
        )
        return (1.0 - p_extreme) * moderate + p_extreme * extreme

    return moderate


def build_fossil_price_risk_multiplier(
    *,
    rng: Any,
    n_years: int,
    energy_label: str | None,
) -> list[float]:
    """
    Génère un multiplicateur annuel pour les prix fossiles.

    Objectif du modèle : représenter un risque fossile visible, justifiable
    historiquement, sans produire des trajectoires absurdes.

    Le multiplicateur est appliqué à la trajectoire centrale :
        prix_simulé(t) = prix_central(t) * multiplicateur(t)

    La trajectoire centrale contient déjà une hypothèse de prix future. Le
    Monte Carlo n'ajoute donc pas une nouvelle tendance permanente. Il ajoute :
    1) une volatilité résiduelle mean-reverting autour du scénario central ;
    2) des chocs haussiers temporaires calibrés sur les années de choc
       observées dans les séries historiques, avec une fréquence historique
       conservée mais une amplitude résiduelle recentrée.

    Différence avec l'ancienne version : on n'utilise plus une marche aléatoire
    cumulative avec la volatilité historique complète. Cette ancienne méthode
    créait des prix fossiles durablement trop élevés et des paybacks souvent
    irréalistes autour de 2 ans.
    """
    import numpy as np

    if energy_label not in FOSSIL_RISK_PREMIUM:
        return [1.0] * int(n_years)

    fossil_cfg = FOSSIL_RISK_PREMIUM[energy_label]

    n_years = int(n_years)
    if n_years <= 0:
        return []

    annual_drift = float(fossil_cfg.get("annual_drift", 0.0))
    sigma_systematic = float(fossil_cfg.get("systematic_sigma_log", 0.0))
    sigma_residual = float(fossil_cfg.get("annual_sigma_residual", 0.10))
    phi = float(fossil_cfg.get("mean_reversion_phi", 0.70))
    phi = _clamp(phi, 0.0, 0.95)

    # 1) Incertitude persistante de niveau.
    # Elle est propre à toute la trajectoire simulée. Sa médiane vaut 1, ce qui
    # évite d'ajouter une tendance haussière au scénario central, mais elle
    # autorise des trajectoires durablement plus basses ou plus hautes. C'est ce
    # qui évite une distribution trop concentrée autour de la moyenne.
    systematic_log_level = rng.normal(loc=0.0, scale=sigma_systematic)

    # 2) Processus mean-reverting en log-multiplicateur.
    # La volatilité stationnaire est environ sigma_residual. Cela donne des
    # écarts persistants, mais évite la dérive explosive d'une marche aléatoire.
    innovation_sigma = sigma_residual * (1.0 - phi ** 2) ** 0.5
    log_multiplier = np.zeros(n_years, dtype=float)

    for t in range(n_years):
        eps = rng.normal(loc=0.0, scale=innovation_sigma)
        if t == 0:
            residual = eps
        else:
            residual = phi * (log_multiplier[t - 1] - systematic_log_level - annual_drift * (t - 1)) + eps
        log_multiplier[t] = systematic_log_level + annual_drift * t + residual

    # Chocs haussiers temporaires. Le choc est fort la première année, puis
    # décroît selon shock_decay. Il ne devient donc pas un nouveau niveau de prix
    # permanent.
    shock_probability = float(fossil_cfg["shock_probability"])
    shock_decay = float(fossil_cfg.get("shock_decay", 0.50))
    shock_decay = _clamp(shock_decay, 0.0, 1.0)

    expected_log_shock_per_year = 0.0
    if bool(fossil_cfg.get("center_shocks_log", False)):
        expected_log_shock_per_year = (
            shock_probability * _expected_log_shock_size(fossil_cfg)
        )

    for year_idx in range(n_years):
        if rng.random() < shock_probability:
            # Distribution mixte :
            # - par défaut, choc modéré ;
            # - si extreme_shock_probability est défini, une petite fraction
            #   des chocs devient un choc extrême.
            if (
                "extreme_shock_probability" in fossil_cfg
                and rng.random() < float(fossil_cfg["extreme_shock_probability"])
            ):
                shock_size = rng.triangular(
                    left=float(fossil_cfg["extreme_shock_min"]),
                    mode=float(fossil_cfg["extreme_shock_mode"]),
                    right=float(fossil_cfg["extreme_shock_max"]),
                )
            else:
                shock_size = rng.triangular(
                    left=float(fossil_cfg["shock_min"]),
                    mode=float(fossil_cfg["shock_mode"]),
                    right=float(fossil_cfg["shock_max"]),
                )

            duration = int(
                rng.integers(
                    int(fossil_cfg["shock_duration_min"]),
                    int(fossil_cfg["shock_duration_max"]) + 1,
                )
            )

            for k in range(duration):
                idx = year_idx + k
                if idx >= n_years:
                    break
                effective_shock = shock_size * (shock_decay ** k)
                log_multiplier[idx] += np.log1p(effective_shock)

    if expected_log_shock_per_year != 0.0:
        log_multiplier = log_multiplier - expected_log_shock_per_year

    multiplier = np.exp(log_multiplier)
    multiplier = np.clip(
        multiplier,
        float(fossil_cfg["min_multiplier"]),
        float(fossil_cfg["max_multiplier"]),
    )

    return [float(x) for x in multiplier]

def _triangular_factor(rng: Any, low: float, mode: float, high: float) -> float:
    return float(rng.triangular(left=low, mode=mode, right=high))


def _constant_series_like(index: Any, value: float) -> pd.Series:
    return pd.Series(float(value), index=index)


def _series_first_n_years(series: pd.Series, n_years: int) -> pd.Series:
    if series is None or len(series) == 0:
        raise ValueError("Série vide ou None.")

    s = pd.Series(series).copy()
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([float("inf"), float("-inf")], pd.NA)
    s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()

    if len(s) == 0 or s.isna().all():
        raise ValueError("Série de prix non exploitable après nettoyage.")

    s = s.astype(float)

    if len(s) >= n_years:
        return s.iloc[:n_years].copy()

    last_value = float(s.iloc[-1])
    missing = n_years - len(s)
    if len(s.index) > 0 and isinstance(s.index[-1], (int, float)):
        start = int(s.index[-1]) + 1
        extension_index = range(start, start + missing)
    else:
        extension_index = range(len(s), len(s) + missing)
    extension = pd.Series([last_value] * missing, index=extension_index)
    return pd.concat([s, extension])



def _clean_price_series(series: pd.Series, *, name: str = "price") -> pd.Series:
    """
    Nettoie une série de prix utilisée dans les calculs déterministes.

    Cette fonction évite que des valeurs manquantes ou des années non alignées
    entre électricité et combustible fossile fassent disparaître les indicateurs
    déterministes ou les graphes de coûts cumulés.
    """
    if series is None or len(series) == 0:
        raise ValueError(f"Série de prix vide : {name}.")

    s = pd.Series(series).copy()
    s.index = pd.to_numeric(pd.Index(s.index), errors="coerce")
    s = pd.to_numeric(s, errors="coerce")
    s = s[~pd.isna(s.index)]
    s.index = s.index.astype(int)
    s = s.sort_index()
    s = s.replace([float("inf"), float("-inf")], pd.NA)
    s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()

    if s.empty or s.isna().all():
        raise ValueError(f"Série de prix non exploitable : {name}.")

    return s.astype(float)


def _align_price_series_to_index(
    series: pd.Series,
    target_index: Any,
    *,
    name: str = "price",
) -> pd.Series:
    """
    Aligne une série de prix sur l'index d'une autre série.

    Cas important corrigé ici :
    - la série mazout peut commencer en 2025 ;
    - la série électricité peut commencer en 2026 ;
    - sans alignement, les coûts déterministes mélangent des années différentes
      et peuvent produire des graphes incomplets ou incohérents.

    Les années manquantes sont interpolées, puis prolongées par la dernière
    valeur connue si nécessaire.
    """
    s = _clean_price_series(series, name=name)
    target = pd.Index(pd.to_numeric(pd.Index(target_index), errors="coerce"))
    target = target[~pd.isna(target)].astype(int)

    if len(target) == 0:
        raise ValueError(f"Index cible vide pour l'alignement de {name}.")

    union_index = sorted(set(s.index.astype(int)) | set(target.astype(int)))
    s = s.reindex(union_index).interpolate(method="linear", limit_direction="both").ffill().bfill()
    s = s.reindex(target).interpolate(method="linear", limit_direction="both").ffill().bfill()

    if s.isna().any():
        raise ValueError(f"Impossible d'aligner complètement la série de prix : {name}.")

    return s.astype(float)


def discounted_payback_from_annual_savings(*, capex_net: float, annual_savings: pd.Series, discount_rate: float = 0.03) -> float | None:
    cumulative = -float(capex_net)
    for i, saving in enumerate(annual_savings, start=1):
        saving_f = _safe_float(saving, 0.0) or 0.0
        discounted_saving = float(saving_f) / ((1.0 + discount_rate) ** i)
        previous = cumulative
        cumulative += discounted_saving
        if cumulative >= 0:
            if discounted_saving <= 0:
                return float(i)
            fraction = abs(previous) / discounted_saving
            return float((i - 1) + fraction)
    return None


def npv_from_annual_savings(*, capex_net: float, annual_savings: pd.Series, discount_rate: float = 0.03) -> float:
    npv = -float(capex_net)
    for i, saving in enumerate(annual_savings, start=1):
        saving_f = _safe_float(saving, 0.0) or 0.0
        npv += float(saving_f) / ((1.0 + discount_rate) ** i)
    return float(npv)


def _finite_percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    import numpy as np
    finite_values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite_values:
        return None
    return float(np.percentile(finite_values, p))


def _share_below(values: list[float], threshold: float, n_total: int) -> float:
    if n_total <= 0:
        return 0.0
    ok = [v for v in values if v is not None and math.isfinite(float(v)) and float(v) <= threshold]
    return len(ok) / n_total


def _median_is_representative(values: list[float], n_total: int) -> bool:
    if n_total <= 0:
        return False
    finite_values = [v for v in values if v is not None and math.isfinite(float(v))]
    return len(finite_values) >= 0.5 * n_total


# ============================================================
# LECTURE DES FICHIERS DE CALIBRATION FROID
# ============================================================

def load_cooling_qref_table() -> pd.DataFrame | None:
    global _QREF_CACHE
    if _QREF_CACHE is not None:
        return _QREF_CACHE
    candidate_paths = [
        COOLING_QREF_CALIBRATED_CSV,
        BASE_DIR / "cooling_qref_calibrated_weighted.csv",
        Path.cwd() / "cooling_qref_calibrated_weighted.csv",
        Path.cwd() / "Tools" / "ajuster_coef_froid" / "cooling_qref_calibrated_weighted.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
            df["typologie"] = df["typologie"].astype(str).map(normalize_calibration_typology)
            df["climate_class"] = df["climate_class"].astype(str).map(normalize_climate_class)
            for col in ["qref_weighted_median_kwh_m2_an", "qref_weighted_mean_kwh_m2_an", "qref_unweighted_median_kwh_m2_an"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            _QREF_CACHE = df
            return _QREF_CACHE
    _QREF_CACHE = None
    return None


def load_climate_station_reference() -> pd.DataFrame | None:
    global _CLIMATE_REF_CACHE
    if _CLIMATE_REF_CACHE is not None:
        return _CLIMATE_REF_CACHE
    candidate_paths = [
        CLIMATE_STATION_REFERENCE_CSV,
        BASE_DIR / "climate_station_reference.csv",
        Path.cwd() / "climate_station_reference.csv",
        Path.cwd() / "Tools" / "ajuster_coef_froid" / "climate_station_reference.csv",
    ]
    for path in candidate_paths:
        if path.exists():
            df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
            if "station_name" in df.columns:
                df["station_name_norm"] = df["station_name"].astype(str).map(_norm)
            else:
                df["station_name_norm"] = ""
            if "climate_class" in df.columns:
                df["climate_class"] = df["climate_class"].astype(str).map(normalize_climate_class)
            for col in ["cdh_26_ref", "f_climat_cdh26_vs_ref", "f_night_potential_vs_ref", "f_hot_nights_vs_ref"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            _CLIMATE_REF_CACHE = df
            return _CLIMATE_REF_CACHE
    _CLIMATE_REF_CACHE = None
    return None


def get_station_climate_reference(station_name: str) -> dict[str, Any] | None:
    df = load_climate_station_reference()
    if df is None or df.empty:
        return None
    station_norm = _norm(station_name)
    if not station_norm or "station_name_norm" not in df.columns:
        return None
    exact = df[df["station_name_norm"] == station_norm]
    if not exact.empty:
        return exact.iloc[0].to_dict()
    contains = df[df["station_name_norm"].str.contains(station_norm, na=False)]
    if not contains.empty:
        return contains.iloc[0].to_dict()
    return None


def get_fallback_qref(typologie: str, climate_class: str) -> float:
    typ = normalize_calibration_typology(typologie)
    clim = normalize_climate_class(climate_class)
    table = COOLING_QREF_FALLBACK_BY_TYPOLOGY_AND_CLIMATE
    if typ in table and clim in table[typ]:
        return float(table[typ][clim])
    if typ in table and "global" in table[typ]:
        return float(table[typ]["global"])
    return float(table["mixte"]["global"])


def get_calibrated_qref(typologie: str, climate_class: str) -> tuple[float, str]:
    typ = normalize_calibration_typology(typologie)
    clim = normalize_climate_class(climate_class)
    fallback = get_fallback_qref(typ, clim)
    df = load_cooling_qref_table()
    if df is None or df.empty:
        return fallback, "fallback_interne_csv_absent"
    value_cols = ["qref_weighted_median_kwh_m2_an", "qref_weighted_mean_kwh_m2_an", "qref_unweighted_median_kwh_m2_an"]
    for target_climate, source_label in [(clim, "csv_typologie_climat"), ("global", "csv_typologie_global"), ("reference", "csv_typologie_reference")]:
        rows = df[(df["typologie"] == typ) & (df["climate_class"] == target_climate)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        for col in value_cols:
            if col in row.index:
                val = _safe_float(row[col], None)
                if val is not None and val > 0:
                    return float(val), source_label
    return fallback, "fallback_interne_absence_ligne"


# ============================================================
# GÉOMÉTRIE
# ============================================================

def estimate_sdep_vh_from_perimeter(shab_m2: float, niveaux: int, perimetre_m: float, hauteur_m: float, toiture_exposee: bool, plancher_expose: bool) -> tuple[float, float]:
    a_foot = shab_m2 / niveaux
    s_murs = perimetre_m * (niveaux * hauteur_m)
    s_toit = a_foot if toiture_exposee else 0.0
    s_plancher = a_foot if plancher_expose else 0.0
    sdep_est = s_murs + s_toit + s_plancher
    vh_est = shab_m2 * hauteur_m
    return sdep_est, vh_est


def estimate_sdep_vh_geometry(
    *,
    shab_m2: float,
    niveaux: int,
    hauteur_m: float,
    forme_generale: str,
    mitoyennete: str,
    toiture_exposee: bool,
    plancher_expose: bool,
    longueur_m: float | None = None,
    largeur_m: float | None = None,
) -> tuple[float, float, dict]:
    if shab_m2 <= 0:
        raise ValueError("shab_m2 doit être > 0.")
    if niveaux <= 0:
        raise ValueError("niveaux doit être > 0.")
    if hauteur_m <= 0:
        raise ValueError("hauteur_m doit être > 0.")
    if mitoyennete not in MITOYENNETE_FACTOR:
        raise ValueError(f"mitoyennete invalide: {mitoyennete}")
    a_sol = shab_m2 / niveaux
    if longueur_m is not None and largeur_m is not None:
        if longueur_m <= 0 or largeur_m <= 0:
            raise ValueError("longueur_m et largeur_m doivent être > 0.")
        longueur = longueur_m
        largeur = largeur_m
        ratio_forme = longueur / largeur
        plan_source = "dimensions_reelles"
    else:
        if forme_generale not in FORM_RATIO_BY_SHAPE:
            raise ValueError(f"forme_generale invalide: {forme_generale}")
        ratio_forme = FORM_RATIO_BY_SHAPE[forme_generale]
        longueur = (a_sol * ratio_forme) ** 0.5
        largeur = (a_sol / ratio_forme) ** 0.5
        plan_source = "forme_generale"
    perimetre = 2.0 * (longueur + largeur)
    s_murs_brut = perimetre * (niveaux * hauteur_m)
    f_mitoyennete = MITOYENNETE_FACTOR[mitoyennete]
    s_murs = s_murs_brut * f_mitoyennete
    s_toit = a_sol if toiture_exposee else 0.0
    s_plancher = a_sol if plancher_expose else 0.0
    sdep_est = s_murs + s_toit + s_plancher
    vh_est = shab_m2 * hauteur_m
    meta = {
        "a_sol_m2": a_sol,
        "ratio_forme": ratio_forme,
        "longueur_m": longueur,
        "largeur_m": largeur,
        "perimetre_m": perimetre,
        "s_murs_brut_m2": s_murs_brut,
        "f_mitoyennete": f_mitoyennete,
        "s_murs_m2": s_murs,
        "s_toit_m2": s_toit,
        "s_plancher_m2": s_plancher,
        "plan_source": plan_source,
    }
    return sdep_est, vh_est, meta


# ============================================================
# FROID
# ============================================================

def softened_climate_factor(*, climate_class: str, cooling_degree_hours: float, station_ref: dict[str, Any] | None) -> tuple[float, str]:
    if station_ref is not None:
        raw = _safe_float(station_ref.get("f_climat_cdh26_vs_ref"), None)
        if raw is not None and raw > 0:
            softened = 1.0 + 0.35 * (raw - 1.0)
            return _clamp(softened, 0.75, 1.30), "station_reference_adoucie"
    cdh = _safe_float(cooling_degree_hours, 0.0) or 0.0
    raw = cdh / CDH_REF if CDH_REF > 0 else 1.0
    softened = 1.0 + 0.25 * (raw - 1.0)
    climate_class = normalize_climate_class(climate_class)
    if climate_class == "froid":
        return _clamp(softened, 0.75, 1.05), "cdh_adouci"
    if climate_class == "tempere":
        return _clamp(softened, 0.80, 1.12), "cdh_adouci"
    if climate_class == "chaud":
        return _clamp(softened, 0.85, 1.20), "cdh_adouci"
    if climate_class == "tres_chaud":
        return _clamp(softened, 0.95, 1.30), "cdh_adouci"
    return _clamp(softened, 0.80, 1.20), "cdh_adouci"


def night_ventilation_factor(*, night_ventilation: bool, station_ref: dict[str, Any] | None) -> tuple[float, str]:
    if not night_ventilation:
        return 1.0, "non_active"
    base = F_NIGHT_BASE[True]
    if station_ref is not None:
        potential = _safe_float(station_ref.get("f_night_potential_vs_ref"), None)
        if potential is not None and potential > 0:
            if potential >= 1.2:
                return 0.88, "potentiel_nocturne_bon"
            if potential >= 0.9:
                return 0.92, "potentiel_nocturne_moyen"
            return 0.97, "potentiel_nocturne_faible"
    return base, "valeur_generique"


def cooling_mode_factor(cooling_mode: str) -> float:
    mode = _norm(cooling_mode)
    if mode == "no_cooling":
        return 0.0
    if mode == "free_cooling":
        return 0.40
    if mode == "hybrid":
        return 0.60
    if mode == "active_cooling":
        return 1.00
    return 1.00


def estimate_cooling_need_kwh(
    *,
    building_type_key: str,
    cooling_mode: str,
    surface_climatisee_m2: float,
    cooling_degree_hours: float,
    vitrage_level: str,
    solar_protection_level: str,
    usage_level: str,
    night_ventilation: bool,
    station_info: Any | None = None,
) -> dict[str, Any]:
    surface = _safe_float(surface_climatisee_m2, 0.0) or 0.0
    mode = _norm(cooling_mode)
    station_name = station_name_from_station_info(station_info)
    station_ref = get_station_climate_reference(station_name)
    if station_ref is not None and station_ref.get("climate_class"):
        climate_class = normalize_climate_class(str(station_ref.get("climate_class")))
        climate_class_source = "station_reference"
    else:
        climate_class = climate_class_from_cdh(cooling_degree_hours)
        climate_class_source = "cdh_fallback"
    typologie_calibration = normalize_calibration_typology(building_type_key)
    if mode == "no_cooling" or surface <= 0:
        return {
            "typologie_calibration": typologie_calibration,
            "climate_class": climate_class,
            "climate_class_source": climate_class_source,
            "station_name": station_name,
            "q_ref_kwh_m2a": 0.0,
            "q_ref_source": "no_cooling",
            "f_climat": 0.0,
            "f_climat_source": "no_cooling",
            "f_vitrage": 1.0,
            "f_solaire": 1.0,
            "f_usage": 1.0,
            "f_night": 1.0,
            "f_night_source": "no_cooling",
            "f_mode": 0.0,
            "q_froid_utile_kwh_an": 0.0,
            "q_froid_utile_kwh_m2a": 0.0,
            "spf_froid": None,
            "conso_elec_clim_kwh_an": 0.0,
        }
    q_ref, q_ref_source = get_calibrated_qref(typologie=typologie_calibration, climate_class=climate_class)
    f_climat, f_climat_source = softened_climate_factor(climate_class=climate_class, cooling_degree_hours=cooling_degree_hours, station_ref=station_ref)
    f_vitrage = F_VITRAGE.get(_norm(vitrage_level), 1.00)
    f_solaire = F_SOLAIRE.get(_norm(solar_protection_level), 1.00)
    f_usage = F_USAGE.get(_norm(usage_level), 1.00)
    f_night, f_night_source = night_ventilation_factor(night_ventilation=night_ventilation, station_ref=station_ref)
    f_mode = cooling_mode_factor(mode)
    q_froid_utile = surface * q_ref * f_climat * f_vitrage * f_solaire * f_usage * f_night * f_mode
    spf_froid = COOLING_SPF_BY_MODE.get(mode)
    conso_elec = q_froid_utile / spf_froid if spf_froid and spf_froid > 0 else 0.0
    return {
        "typologie_calibration": typologie_calibration,
        "climate_class": climate_class,
        "climate_class_source": climate_class_source,
        "station_name": station_name,
        "q_ref_kwh_m2a": q_ref,
        "q_ref_source": q_ref_source,
        "f_climat": f_climat,
        "f_climat_source": f_climat_source,
        "f_vitrage": f_vitrage,
        "f_solaire": f_solaire,
        "f_usage": f_usage,
        "f_night": f_night,
        "f_night_source": f_night_source,
        "f_mode": f_mode,
        "q_froid_utile_kwh_an": q_froid_utile,
        "q_froid_utile_kwh_m2a": q_froid_utile / surface if surface > 0 else 0.0,
        "spf_froid": spf_froid,
        "conso_elec_clim_kwh_an": conso_elec,
    }


# ============================================================
# INCERTITUDE ET INDICATEURS ÉCONOMIQUES
# ============================================================

def get_uncertainty_config(*, inputs: ProjectInputs, froid: dict[str, Any], scenario_label: str) -> dict[str, Any]:
    cfg = UNCERTAINTY_DEFAULTS.copy()
    if inputs.sdep_mode == "Saisie directe":
        cfg["heating_need_sigma"] = 0.10
    elif inputs.sdep_mode == "Estimation par périmètre":
        cfg["heating_need_sigma"] = 0.15
    elif inputs.sdep_mode in {"Estimation typologique calibrée", "Estimation typologique"}:
        cfg["heating_need_sigma"] = 0.22
    else:
        cfg["heating_need_sigma"] = 0.22
    typ = str(froid.get("typologie_calibration", "mixte"))
    cfg["cooling_need_sigma"] = COOLING_SIGMA_BY_TYPOLOGY.get(typ, 0.35)
    cooling_mode = str(froid.get("cooling_mode_effective", inputs.cooling_mode))
    if cooling_mode == "active_cooling":
        cfg["cooling_need_sigma"] *= 0.90
    elif cooling_mode == "hybrid":
        cfg["cooling_need_sigma"] *= 1.05
    elif cooling_mode == "free_cooling":
        cfg["cooling_need_sigma"] *= 1.20
    elif cooling_mode == "no_cooling":
        cfg["cooling_need_sigma"] = 0.0
    qref_source = str(froid.get("q_ref_source", ""))
    if qref_source.startswith("fallback"):
        cfg["cooling_need_sigma"] *= 1.30
    if froid.get("climate_class_source") == "cdh_fallback":
        cfg["cooling_need_sigma"] *= 1.10
    if not inputs.want_cooling or inputs.surface_climatisee_m2 <= 0:
        cfg["cooling_need_sigma"] = 0.0
    cfg["cooling_need_sigma"] = _clamp(cfg["cooling_need_sigma"], 0.0, 0.75)
    if scenario_label == "Central":
        cfg["price_elec_sigma"] = 0.107
        cfg["price_fuel_sigma"] = 0.040
    else:
        cfg["price_elec_sigma"] = 0.107
        cfg["price_fuel_sigma"] = 0.050
    return cfg


def compute_confidence_level(*, inputs: ProjectInputs, froid: dict[str, Any]) -> dict[str, Any]:
    score = 0
    reasons_positive: list[str] = []
    reasons_negative: list[str] = []
    if inputs.sdep_mode == "Saisie directe":
        score += 2
        reasons_positive.append("Sdép et Vh saisis directement")
    elif inputs.sdep_mode == "Estimation par périmètre":
        score += 1
        reasons_positive.append("géométrie estimée par périmètre")
    elif inputs.sdep_mode in {"Estimation typologique calibrée", "Estimation typologique"}:
        reasons_negative.append("Sdép estimée par coefficient typologique calibré")
    else:
        reasons_negative.append("géométrie estimée par typologie")
    qref_source = str(froid.get("q_ref_source", ""))
    if qref_source and not qref_source.startswith("fallback"):
        score += 2
        reasons_positive.append("q_ref froid issu du fichier de calibration")
    else:
        reasons_negative.append("q_ref froid issu d'un fallback")
    typ = str(froid.get("typologie_calibration", ""))
    if typ in ["bureaux", "ecole", "commerce"]:
        score += 1
        reasons_positive.append(f"typologie {typ} relativement mieux documentée")
    elif typ in ["residentiel", "activites", "equipement_collectif", "mixte"]:
        reasons_negative.append(f"typologie {typ} encore fragile")
    if inputs.current_efficiency is not None and inputs.current_efficiency > 0:
        score += 1
        reasons_positive.append("rendement du système actuel renseigné")
    else:
        reasons_negative.append("rendement du système actuel incertain")
    if inputs.has_existing_ac:
        if inputs.eer_current_ac is not None and inputs.eer_current_ac > 0:
            score += 1
            reasons_positive.append("performance de la clim actuelle renseignée")
        else:
            reasons_negative.append("clim actuelle présente mais performance inconnue")
    if score >= 5:
        level = "élevé"
    elif score >= 3:
        level = "moyen"
    else:
        level = "faible"
    return {"level": level, "score": score, "reasons_positive": reasons_positive, "reasons_negative": reasons_negative}


def compute_economic_indicators(
    *,
    cost_ref_series: pd.Series | None,
    cost_pac_series: pd.Series | None,
    capex_net: float,
    discount_rate: float = UNCERTAINTY_DISCOUNT_RATE,
    payback_max_years: int = UNCERTAINTY_PAYBACK_MAX_YEARS,
) -> dict[str, Any]:
    if cost_ref_series is None or cost_pac_series is None:
        return {"cout_ref_year0": None, "cout_pac_year0": None, "economies_year0": None, "payback": None, "npv": None}
    cout_ref_year0 = float(cost_ref_series.iloc[0])
    cout_pac_year0 = float(cost_pac_series.iloc[0])
    economies_year0 = cout_ref_year0 - cout_pac_year0
    res_pb_standard = payback_discounted_from_cashflows(
        capex_chf=capex_net,
        cost_ref=cost_ref_series,
        cost_new=cost_pac_series,
        discount_rate=discount_rate,
    )
    npv = res_pb_standard.get("npv")
    cost_ref_extended = _series_first_n_years(cost_ref_series, payback_max_years)
    cost_pac_extended = _series_first_n_years(cost_pac_series, payback_max_years)
    payback_extended = discounted_payback_from_annual_savings(
        capex_net=capex_net,
        annual_savings=cost_ref_extended - cost_pac_extended,
        discount_rate=discount_rate,
    )
    return {"cout_ref_year0": cout_ref_year0, "cout_pac_year0": cout_pac_year0, "economies_year0": economies_year0, "payback": payback_extended, "npv": npv}


def monte_carlo_project_uncertainty(
    *,
    inputs: ProjectInputs,
    froid: dict[str, Any],
    scenario_label: str,
    p_elec: pd.Series,
    p_actuel_heat: pd.Series,
    besoin_chauffage_kwh: float,
    besoin_froid_utile_kwh: float,
    capex_brut: float,
    subvention: float,
    spf_heat: float,
    spf_cool: float | None,
    om_pac_chf_per_year: float,
    om_ref_chf_per_year: float,
    current_efficiency: float,
    has_existing_ac: bool,
    eer_current_ac: float | None,
    n_sims: int = UNCERTAINTY_N_SIMS,
    horizon_years: int = UNCERTAINTY_PROJECT_LIFETIME_YEARS,
    payback_max_years: int = UNCERTAINTY_PAYBACK_MAX_YEARS,
    discount_rate: float = UNCERTAINTY_DISCOUNT_RATE,
    seed: int = 42,
) -> dict[str, Any]:
    import numpy as np
    if p_elec is None or p_actuel_heat is None:
        return {"available": False, "reason": "series_prix_absentes"}
    if current_efficiency is None or current_efficiency <= 0:
        return {"available": False, "reason": "rendement_reference_invalide"}
    if spf_heat is None or spf_heat <= 0:
        return {"available": False, "reason": "spf_chauffage_invalide"}
    rng = np.random.default_rng(seed)
    cfg = get_uncertainty_config(inputs=inputs, froid=froid, scenario_label=scenario_label)
    confidence = compute_confidence_level(inputs=inputs, froid=froid)
    p_elec_life = _series_first_n_years(p_elec, horizon_years)
    p_ref_life = _series_first_n_years(p_actuel_heat, horizon_years)
    p_elec_payback = _series_first_n_years(p_elec, payback_max_years)
    p_ref_payback = _series_first_n_years(p_actuel_heat, payback_max_years)
    capex_factors = [_triangular_factor(rng, cfg["capex_min"], cfg["capex_mode"], cfg["capex_max"]) for _ in range(n_sims)]
    capex_factors = _normalize_factors_to_median_one(capex_factors)
    paybacks_extended: list[float] = []
    paybacks_within_life: list[float] = []
    npvs_life: list[float] = []
    annual_saving_year0: list[float] = []
    capex_net_samples: list[float] = []
    heat_need_samples: list[float] = []
    cool_need_samples: list[float] = []
    amortized_within_life_flags: list[bool] = []
    amortized_within_extended_flags: list[bool] = []
    for i_sim in range(n_sims):
        f_heat_need = _bounded_lognormal_factor(rng, sigma=cfg["heating_need_sigma"], low=cfg["heating_need_min"], high=cfg["heating_need_max"])
        if cfg["cooling_need_sigma"] > 0:
            f_cool_need = _bounded_lognormal_factor(rng, sigma=cfg["cooling_need_sigma"], low=cfg["cooling_need_min"], high=cfg["cooling_need_max"])
        else:
            f_cool_need = 1.0
        heat_need_i = max(0.0, float(besoin_chauffage_kwh) * f_heat_need)
        cool_need_i = max(0.0, float(besoin_froid_utile_kwh) * f_cool_need)
        capex_brut_i = max(0.0, float(capex_brut) * capex_factors[i_sim])
        subvention_i = min(float(subvention), capex_brut_i)
        capex_net_i = max(0.0, capex_brut_i - subvention_i)
        f_spf_heat = _triangular_factor(rng, cfg["spf_heat_min"], cfg["spf_heat_mode"], cfg["spf_heat_max"])
        spf_heat_i = max(0.1, float(spf_heat) * f_spf_heat)
        if spf_cool is not None and spf_cool > 0:
            f_spf_cool = _triangular_factor(rng, cfg["spf_cool_min"], cfg["spf_cool_mode"], cfg["spf_cool_max"])
            spf_cool_i = max(0.1, float(spf_cool) * f_spf_cool)
        else:
            spf_cool_i = None
        # Prix de l'électricité : incertitude calibrée sur les variations
        # annuelles cantonales ElCom 2011--2026 (sigma log ≈ 0.107,
        # bornes empiriques ≈ P5--P95 : 0.90--1.30).
        f_price_elec = _bounded_lognormal_factor(
            rng,
            sigma=cfg["price_elec_sigma"],
            low=0.90,
            high=1.30,
        )
        p_elec_life_i = p_elec_life * f_price_elec
        p_elec_payback_i = p_elec_payback * f_price_elec

        # Prix de l'énergie actuelle :
        # - gaz/mazout : prime de risque fossile + bruit cumulatif + chocs temporaires ;
        # - autres énergies : bruit résiduel simple.
        if inputs.current_energy in FOSSIL_ENERGY_LABELS:
            fossil_mult_payback = build_fossil_price_risk_multiplier(
                rng=rng,
                n_years=len(p_ref_payback),
                energy_label=inputs.current_energy,
            )

            fossil_mult_payback_s = pd.Series(
                fossil_mult_payback,
                index=p_ref_payback.index,
            )

            fossil_mult_life_s = fossil_mult_payback_s.iloc[:len(p_ref_life)].copy()
            fossil_mult_life_s.index = p_ref_life.index

            p_ref_life_i = p_ref_life * fossil_mult_life_s
            p_ref_payback_i = p_ref_payback * fossil_mult_payback_s

        else:
            f_price_ref = _bounded_lognormal_factor(
                rng,
                sigma=cfg["price_fuel_sigma"],
                low=0.85,
                high=1.20,
            )
            p_ref_life_i = p_ref_life * f_price_ref
            p_ref_payback_i = p_ref_payback * f_price_ref
        f_om_pac_random = _triangular_factor(rng, cfg["om_min"], cfg["om_mode"], cfg["om_max"])
        f_om_ref_random = _triangular_factor(rng, cfg["om_min"], cfg["om_mode"], cfg["om_max"])
        f_om_correlated = f_heat_need ** cfg["om_heat_need_elasticity"]
        om_pac_i = max(0.0, float(om_pac_chf_per_year) * f_om_pac_random * f_om_correlated)
        om_ref_i = max(0.0, float(om_ref_chf_per_year) * f_om_ref_random * f_om_correlated)
        def build_savings_series(p_elec_i: pd.Series, p_ref_i: pd.Series) -> pd.Series:
            cost_pac_heat_i = (heat_need_i / spf_heat_i) * p_elec_i
            if cool_need_i > 0 and spf_cool_i is not None:
                cost_pac_cool_i = (cool_need_i / spf_cool_i) * p_elec_i
            else:
                cost_pac_cool_i = pd.Series(0.0, index=p_elec_i.index)
            cost_pac_i = cost_pac_heat_i.add(cost_pac_cool_i, fill_value=0.0).add(om_pac_i, fill_value=0.0)
            cost_ref_heat_i = (heat_need_i / float(current_efficiency)) * p_ref_i
            if has_existing_ac and cool_need_i > 0 and eer_current_ac is not None and eer_current_ac > 0:
                cost_ref_cool_i = (cool_need_i / float(eer_current_ac)) * p_elec_i
            else:
                cost_ref_cool_i = pd.Series(0.0, index=p_elec_i.index)
            cost_ref_i = cost_ref_heat_i.add(cost_ref_cool_i, fill_value=0.0).add(om_ref_i, fill_value=0.0)
            return cost_ref_i - cost_pac_i
        savings_life_i = build_savings_series(p_elec_life_i, p_ref_life_i)
        savings_payback_i = build_savings_series(p_elec_payback_i, p_ref_payback_i)
        pb_life = discounted_payback_from_annual_savings(capex_net=capex_net_i, annual_savings=savings_life_i, discount_rate=discount_rate)
        pb_extended = discounted_payback_from_annual_savings(capex_net=capex_net_i, annual_savings=savings_payback_i, discount_rate=discount_rate)
        van_life = npv_from_annual_savings(capex_net=capex_net_i, annual_savings=savings_life_i, discount_rate=discount_rate)
        capex_net_samples.append(capex_net_i)
        heat_need_samples.append(heat_need_i)
        cool_need_samples.append(cool_need_i)
        npvs_life.append(van_life)
        annual_saving_year0.append(float(savings_life_i.iloc[0]))
        if pb_life is not None:
            amortized_within_life_flags.append(True)
            paybacks_within_life.append(pb_life)
        else:
            amortized_within_life_flags.append(False)
        if pb_extended is not None:
            amortized_within_extended_flags.append(True)
            paybacks_extended.append(pb_extended)
        else:
            amortized_within_extended_flags.append(False)
    n_amortized_life = sum(amortized_within_life_flags)
    n_not_amortized_life = len(amortized_within_life_flags) - n_amortized_life
    n_amortized_extended = sum(amortized_within_extended_flags)
    n_not_amortized_extended = len(amortized_within_extended_flags) - n_amortized_extended
    n_valid_npv_life = sum(1 for v in npvs_life if v is not None and math.isfinite(float(v)))
    amortization_probability_life = n_amortized_life / len(amortized_within_life_flags) if amortized_within_life_flags else 0.0
    amortization_probability_extended = n_amortized_extended / len(amortized_within_extended_flags) if amortized_within_extended_flags else 0.0
    return {
        "available": True,
        "method": "monte_carlo_central_chocs_fossiles_historiques_mean_reverting",
        "n_sims": n_sims,
        "horizon_years": horizon_years,
        "payback_max_years": payback_max_years,
        "discount_rate": discount_rate,
        "confidence": confidence,
        "payback_p10": _finite_percentile(paybacks_extended, 10),
        "payback_p20": _finite_percentile(paybacks_extended, 20),
        "payback_p50": _finite_percentile(paybacks_extended, 50),
        "payback_p80": _finite_percentile(paybacks_extended, 80),
        "payback_p90": _finite_percentile(paybacks_extended, 90),
        "payback_mean": float(np.mean(paybacks_extended)) if paybacks_extended else None,
        "payback_median_representative": _median_is_representative(paybacks_extended, n_sims),
        "amortization_probability": amortization_probability_life,
        "amortization_probability_life": amortization_probability_life,
        "amortization_probability_extended": amortization_probability_extended,
        "probability_payback_le_25y": _share_below(paybacks_extended, 25.0, n_sims),
        "probability_payback_le_30y": _share_below(paybacks_extended, 30.0, n_sims),
        "probability_payback_le_40y": _share_below(paybacks_extended, 40.0, n_sims),
        "probability_payback_le_50y": _share_below(paybacks_extended, 50.0, n_sims),
        "n_amortized": n_amortized_life,
        "n_not_amortized": n_not_amortized_life,
        "n_amortized_life": n_amortized_life,
        "n_not_amortized_life": n_not_amortized_life,
        "n_amortized_extended": n_amortized_extended,
        "n_not_amortized_extended": n_not_amortized_extended,
        "n_valid_npv_life": n_valid_npv_life,
        "npv_p10": _finite_percentile(npvs_life, 10),
        "npv_p50": _finite_percentile(npvs_life, 50),
        "npv_p90": _finite_percentile(npvs_life, 90),
        "annual_saving_year0_p10": _finite_percentile(annual_saving_year0, 10),
        "annual_saving_year0_p50": _finite_percentile(annual_saving_year0, 50),
        "annual_saving_year0_p90": _finite_percentile(annual_saving_year0, 90),
        "capex_net_p10": _finite_percentile(capex_net_samples, 10),
        "capex_net_p50": _finite_percentile(capex_net_samples, 50),
        "capex_net_p90": _finite_percentile(capex_net_samples, 90),
        "heating_need_p10": _finite_percentile(heat_need_samples, 10),
        "heating_need_p50": _finite_percentile(heat_need_samples, 50),
        "heating_need_p90": _finite_percentile(heat_need_samples, 90),
        "cooling_need_p10": _finite_percentile(cool_need_samples, 10),
        "cooling_need_p50": _finite_percentile(cool_need_samples, 50),
        "cooling_need_p90": _finite_percentile(cool_need_samples, 90),
        "simulation_samples": {
            "paybacks_extended": paybacks_extended,
            "paybacks_within_life": paybacks_within_life,
            "npvs_life": npvs_life,
            "annual_saving_year0": annual_saving_year0,
            "capex_net": capex_net_samples,
            "heating_need": heat_need_samples,
            "cooling_need": cool_need_samples,
        },
        "assumptions": {
            "heating_need_sigma": cfg["heating_need_sigma"],
            "cooling_need_sigma": cfg["cooling_need_sigma"],
            "heating_need_bounds": [cfg["heating_need_min"], cfg["heating_need_max"]],
            "cooling_need_bounds": [cfg["cooling_need_min"], cfg["cooling_need_max"]],
            "capex_range": [cfg["capex_min"], cfg["capex_mode"], cfg["capex_max"]],
            "subsidy_randomized": False,
            "spf_heat_range": [cfg["spf_heat_min"], cfg["spf_heat_mode"], cfg["spf_heat_max"]],
            "spf_cool_range": [cfg["spf_cool_min"], cfg["spf_cool_mode"], cfg["spf_cool_max"]],
            "price_elec_sigma": cfg["price_elec_sigma"],
            "price_elec_bounds": [0.90, 1.30],
            "price_fuel_sigma": cfg["price_fuel_sigma"],
            "om_range": [cfg["om_min"], cfg["om_mode"], cfg["om_max"]],
            "capex_heat_need_elasticity": cfg["capex_heat_need_elasticity"],
            "om_heat_need_elasticity": cfg["om_heat_need_elasticity"],
            "capex_factors_normalized_to_median_one": True,
            "fossil_risk_applied": inputs.current_energy in FOSSIL_ENERGY_LABELS,
            "fossil_risk_model": (
                {
                    **FOSSIL_RISK_PREMIUM.get(inputs.current_energy),
                    "interpretation": (
                        "Paramètres calibrés à partir des rendements annuels historiques. "
                        "Le drift est fixé à zéro pour éviter de doubler la tendance déjà "
                        "présente dans la trajectoire centrale."
                    ),
                }
                if inputs.current_energy in FOSSIL_ENERGY_LABELS
                else None
            ),
        },
    }


def compute_one_at_a_time_sensitivity(
    *,
    capex_net: float,
    besoin_chauffage_kwh: float,
    energie_froid_utile_kwh: float,
    p_elec: pd.Series,
    p_actuel_heat: pd.Series,
    spf_heat: float,
    spf_cool: float | None,
    current_efficiency: float,
    has_existing_ac: bool,
    eer_current_ac: float | None,
    om_pac_chf_per_year: float,
    om_ref_chf_per_year: float,
    discount_rate: float = UNCERTAINTY_DISCOUNT_RATE,
) -> list[dict[str, Any]]:
    def rebuild_costs(
        *,
        capex_factor: float = 1.0,
        elec_price_factor: float = 1.0,
        ref_energy_price_factor: float = 1.0,
        spf_heat_factor: float = 1.0,
        heat_need_factor: float = 1.0,
        cooling_need_factor: float = 1.0,
        om_factor: float = 1.0,
    ) -> dict[str, Any]:
        heat_need_i = max(0.0, float(besoin_chauffage_kwh) * heat_need_factor)
        cool_need_i = max(0.0, float(energie_froid_utile_kwh) * cooling_need_factor)
        p_elec_i = p_elec * elec_price_factor
        p_ref_i = p_actuel_heat * ref_energy_price_factor
        spf_heat_i = max(0.1, float(spf_heat) * spf_heat_factor)
        cost_pac_heat_i = (heat_need_i / spf_heat_i) * p_elec_i
        if cool_need_i > 0 and spf_cool is not None and spf_cool > 0:
            cost_pac_cool_i = (cool_need_i / spf_cool) * p_elec_i
        else:
            cost_pac_cool_i = pd.Series(0.0, index=p_elec_i.index)
        cost_pac_total_i = cost_pac_heat_i.add(cost_pac_cool_i, fill_value=0.0).add(float(om_pac_chf_per_year) * om_factor, fill_value=0.0)
        cost_ref_heat_i = (heat_need_i / float(current_efficiency)) * p_ref_i
        if has_existing_ac and cool_need_i > 0 and eer_current_ac is not None and eer_current_ac > 0:
            cost_ref_cool_i = (cool_need_i / float(eer_current_ac)) * p_elec_i
        else:
            cost_ref_cool_i = pd.Series(0.0, index=p_elec_i.index)
        cost_ref_total_i = cost_ref_heat_i.add(cost_ref_cool_i, fill_value=0.0).add(float(om_ref_chf_per_year) * om_factor, fill_value=0.0)
        return compute_economic_indicators(cost_ref_series=cost_ref_total_i, cost_pac_series=cost_pac_total_i, capex_net=float(capex_net) * capex_factor, discount_rate=discount_rate)
    central = rebuild_costs()
    sensitivity_specs = [
        {"parametre": "Prix électricité", "low_label": "-20 %", "high_label": "+20 %", "low_kwargs": {"elec_price_factor": 0.80}, "high_kwargs": {"elec_price_factor": 1.20}},
        {"parametre": "Prix énergie actuelle", "low_label": "-20 %", "high_label": "+20 %", "low_kwargs": {"ref_energy_price_factor": 0.80}, "high_kwargs": {"ref_energy_price_factor": 1.20}},
        {"parametre": "CAPEX net", "low_label": "-15 %", "high_label": "+15 %", "low_kwargs": {"capex_factor": 0.85}, "high_kwargs": {"capex_factor": 1.15}},
        {"parametre": "SPF chauffage PAC", "low_label": "-10 %", "high_label": "+10 %", "low_kwargs": {"spf_heat_factor": 0.90}, "high_kwargs": {"spf_heat_factor": 1.10}},
        {"parametre": "Besoin chauffage", "low_label": "-15 %", "high_label": "+15 %", "low_kwargs": {"heat_need_factor": 0.85}, "high_kwargs": {"heat_need_factor": 1.15}},
        {"parametre": "Besoin froid", "low_label": "-30 %", "high_label": "+30 %", "low_kwargs": {"cooling_need_factor": 0.70}, "high_kwargs": {"cooling_need_factor": 1.30}},
        {"parametre": "Maintenance", "low_label": "-15 %", "high_label": "+15 %", "low_kwargs": {"om_factor": 0.85}, "high_kwargs": {"om_factor": 1.15}},
    ]
    rows = []
    for spec in sensitivity_specs:
        low = rebuild_costs(**spec["low_kwargs"])
        high = rebuild_costs(**spec["high_kwargs"])
        if central["payback"] is not None and low["payback"] is not None and high["payback"] is not None:
            impact_payback_abs = max(abs(float(low["payback"]) - float(central["payback"])), abs(float(high["payback"]) - float(central["payback"])))
        else:
            impact_payback_abs = None
        if central["npv"] is not None and low["npv"] is not None and high["npv"] is not None:
            impact_npv_abs = max(abs(float(low["npv"]) - float(central["npv"])), abs(float(high["npv"]) - float(central["npv"])))
        else:
            impact_npv_abs = None
        rows.append({
            "parametre": spec["parametre"],
            "variation_basse": spec["low_label"],
            "payback_bas": low["payback"],
            "npv_bas": low["npv"],
            "variation_centrale": "central",
            "payback_central": central["payback"],
            "npv_central": central["npv"],
            "variation_haute": spec["high_label"],
            "payback_haut": high["payback"],
            "npv_haut": high["npv"],
            "impact_payback_abs": impact_payback_abs,
            "impact_npv_abs": impact_npv_abs,
        })
    return sorted(rows, key=lambda r: r["impact_npv_abs"] if r["impact_npv_abs"] is not None else -1, reverse=True)


# ============================================================
# ÉVALUATION PROJET
# ============================================================

def evaluer_projet(inputs: ProjectInputs) -> dict[str, Any]:
    postcode_meta = postcode_info(inputs.postcode)
    canton_from_postcode = postcode_meta["canton_name"]
    if inputs.canton != canton_from_postcode:
        inputs.canton = canton_from_postcode
    is_replacement = project_requires_reference_system(inputs.project_type)
    building_type_price = _get_price_building_type(inputs.building_type_key)
    station_info = nearest_station_for_postcode(inputs.postcode)
    res_ch = calcul_pointe_et_energie_annuelle(postcode=inputs.postcode, ubat=inputs.ubat, sdep_m2=float(inputs.sdep_m2), ventilation_r=inputs.ventilation_r, vh_m3=float(inputs.vh_m3))
    besoin_total = float(res_ch["energie_annuelle_kwh"])
    p_max_kw = float(res_ch["p_pointe_kw"])
    cooling_hours = cooling_hours_for_postcode(inputs.postcode)
    cooling_degree_hours = cooling_degree_hours_for_postcode(inputs.postcode)
    effective_cooling_mode = inputs.cooling_mode
    if not inputs.want_cooling:
        effective_cooling_mode = "no_cooling"
    cool_res = estimate_cooling_need_kwh(building_type_key=inputs.building_type_key, cooling_mode=effective_cooling_mode, surface_climatisee_m2=inputs.surface_climatisee_m2, cooling_degree_hours=cooling_degree_hours, vitrage_level=inputs.vitrage_level, solar_protection_level=inputs.solar_protection_level, usage_level=inputs.usage_level, night_ventilation=inputs.night_ventilation, station_info=station_info)
    energie_froid_utile_kwh = float(cool_res["q_froid_utile_kwh_an"])
    co2_factor_elec_kg_kwh = g_per_kwh_to_kg_per_kwh(CO2_FACTORS_G_PER_KWH["Électricité (réseau)"])
    energie_label = None
    rendement_actuel = None
    co2_factor_actuel = None
    if is_replacement:
        energie_label = inputs.current_energy
        rendement_actuel = float(inputs.current_efficiency)
        co2_factor_actuel = g_per_kwh_to_kg_per_kwh(CO2_FACTORS_G_PER_KWH[energie_label])
    om_pac_chf_per_year = annual_om_cost_chf("PAC géothermique", p_max_kw)
    gshp = couts_annuels_gshp_zuberi(p_nom_kw=p_max_kw, chaleur_utile_kwh=besoin_total, prix_elec_chf_kwh=0.25, om_chf_per_year=om_pac_chf_per_year)
    capex_brut, subvention_pac, capex_net = gshp_capex_net_after_subsidy(p_nom_kw=p_max_kw, canton=inputs.canton, project_type=inputs.project_type, current_energy_label=energie_label)
    def load_price_pair(electricity_scenario: str, fuel_scenario: str) -> tuple[pd.Series, pd.Series | None]:
        p_elec_local = load_price_path_electricity_ttc_by_canton(
            SCEN_ELEC_CSV,
            canton=inputs.canton,
            building_type=building_type_price,
            scenario=electricity_scenario,
        )
        p_elec_local = _clean_price_series(p_elec_local, name="electricite")

        p_gas_local = load_price_path_fuel(SCEN_GAS_CSV, scenario=fuel_scenario)
        p_oil_local = load_price_path_fuel(SCEN_OIL_CSV, scenario=fuel_scenario)

        if not is_replacement:
            return p_elec_local, None

        if energie_label == "Gaz naturel":
            p_ref_local = _align_price_series_to_index(
                p_gas_local,
                p_elec_local.index,
                name="gaz",
            )
        elif energie_label == "Mazout (fioul)":
            p_ref_local = _align_price_series_to_index(
                p_oil_local,
                p_elec_local.index,
                name="mazout",
            )
        else:
            p_ref_local = p_elec_local.copy()

        return p_elec_local, p_ref_local
    om_systeme_actuel_chf_per_year = annual_om_cost_chf(energie_label, p_max_kw) if is_replacement and energie_label is not None else 0.0
    def build_cost_and_emission_series(*, p_elec: pd.Series, p_actuel_heat: pd.Series | None, om_systeme_actuel_chf_per_year: float) -> dict[str, Any]:
        p_elec = _clean_price_series(p_elec, name="electricite")
        if p_actuel_heat is not None:
            p_actuel_heat = _align_price_series_to_index(
                p_actuel_heat,
                p_elec.index,
                name=str(energie_label or "systeme_actuel"),
            )

        cost_pac_heat_energy_series = (besoin_total / float(gshp.spf)) * p_elec
        want_gshp_cooling = effective_cooling_mode != "no_cooling" and energie_froid_utile_kwh > 0
        if want_gshp_cooling and cool_res["spf_froid"] is not None:
            cost_pac_cool_series = float(cool_res["conso_elec_clim_kwh_an"]) * p_elec
        else:
            cost_pac_cool_series = pd.Series(0.0, index=p_elec.index)
        om_pac_series = _constant_series_like(p_elec.index, gshp.om_chf_per_year)
        cost_pac_total_series = cost_pac_heat_energy_series.add(cost_pac_cool_series, fill_value=0.0).add(om_pac_series, fill_value=0.0)
        emissions_pac_series = pd.Series(float(gshp.conso_elec_kwh_per_year) * co2_factor_elec_kg_kwh + float(cool_res["conso_elec_clim_kwh_an"]) * co2_factor_elec_kg_kwh, index=p_elec.index)
        cost_actuel_heat_energy_series = None
        cost_actuel_cool_series = None
        om_actuel_series = None
        cost_actuel_total_series = None
        emissions_actuel_series = None
        emissions_clim_actuelle_kg = 0.0
        if is_replacement:
            if p_actuel_heat is None:
                raise ValueError("Série de prix du système actuel absente.")
            cost_actuel_heat_energy_series = (besoin_total / float(rendement_actuel)) * p_actuel_heat
            if inputs.has_existing_ac and energie_froid_utile_kwh > 0 and inputs.eer_current_ac:
                cost_actuel_cool_series = (energie_froid_utile_kwh / float(inputs.eer_current_ac)) * p_elec
                emissions_clim_actuelle_kg = (energie_froid_utile_kwh / float(inputs.eer_current_ac)) * co2_factor_elec_kg_kwh
            else:
                cost_actuel_cool_series = pd.Series(0.0, index=p_elec.index)
            om_actuel_series = _constant_series_like(p_elec.index, om_systeme_actuel_chf_per_year)
            cost_actuel_total_series = cost_actuel_heat_energy_series.add(cost_actuel_cool_series, fill_value=0.0).add(om_actuel_series, fill_value=0.0)
            conso_energie_actuelle_kwh = besoin_total / float(rendement_actuel)
            emissions_actuel_series = pd.Series(conso_energie_actuelle_kwh * float(co2_factor_actuel) + emissions_clim_actuelle_kg, index=p_elec.index)
        return {
            "cost_pac_heat_energy_series": cost_pac_heat_energy_series,
            "cost_pac_cool_series": cost_pac_cool_series,
            "om_pac_series": om_pac_series,
            "cost_pac_total_series": cost_pac_total_series,
            "emissions_pac_series": emissions_pac_series,
            "cost_actuel_heat_energy_series": cost_actuel_heat_energy_series,
            "cost_actuel_cool_series": cost_actuel_cool_series,
            "om_actuel_series": om_actuel_series,
            "cost_actuel_total_series": cost_actuel_total_series,
            "emissions_actuel_series": emissions_actuel_series,
        }
    scenario_results: dict[str, dict[str, Any]] = {}
    price_sensitivity_results: dict[str, dict[str, Any]] = {}
    parameter_sensitivity: list[dict[str, Any]] = []
    p_elec_central, p_actuel_heat_central = load_price_pair(CENTRAL_SCENARIO["electricity"], CENTRAL_SCENARIO["fuel"])
    central_series = build_cost_and_emission_series(p_elec=p_elec_central, p_actuel_heat=p_actuel_heat_central, om_systeme_actuel_chf_per_year=om_systeme_actuel_chf_per_year)
    central_eco = compute_economic_indicators(cost_ref_series=central_series["cost_actuel_total_series"], cost_pac_series=central_series["cost_pac_total_series"], capex_net=capex_net, discount_rate=UNCERTAINTY_DISCOUNT_RATE)
    central_payload = {
        "label": "Central",
        "electricity_scenario": CENTRAL_SCENARIO["electricity"],
        "fuel_scenario": CENTRAL_SCENARIO["fuel"],
        "years": list(p_elec_central.index),
        **central_series,
        "cout_pac_total_year0": central_eco["cout_pac_year0"],
        "cout_actuel_total_year0": central_eco["cout_ref_year0"],
        "economies_annuelles_year0": central_eco["economies_year0"],
        "payback": central_eco["payback"],
        "npv": central_eco["npv"],
    }
    if is_replacement and p_actuel_heat_central is not None:
        central_payload["uncertainty"] = monte_carlo_project_uncertainty(inputs=inputs, froid={"cooling_mode_effective": effective_cooling_mode, **cool_res}, scenario_label="Central", p_elec=p_elec_central, p_actuel_heat=p_actuel_heat_central, besoin_chauffage_kwh=besoin_total, besoin_froid_utile_kwh=energie_froid_utile_kwh, capex_brut=capex_brut, subvention=subvention_pac, spf_heat=float(gshp.spf), spf_cool=cool_res["spf_froid"], om_pac_chf_per_year=float(gshp.om_chf_per_year), om_ref_chf_per_year=float(om_systeme_actuel_chf_per_year), current_efficiency=float(rendement_actuel), has_existing_ac=inputs.has_existing_ac, eer_current_ac=inputs.eer_current_ac, seed=42)
    else:
        central_payload["uncertainty"] = {"available": False, "reason": "pas_de_systeme_reference"}
    scenario_results["Central"] = central_payload
    if is_replacement and p_actuel_heat_central is not None:
        parameter_sensitivity = compute_one_at_a_time_sensitivity(capex_net=capex_net, besoin_chauffage_kwh=besoin_total, energie_froid_utile_kwh=energie_froid_utile_kwh, p_elec=p_elec_central, p_actuel_heat=p_actuel_heat_central, spf_heat=float(gshp.spf), spf_cool=cool_res["spf_froid"], current_efficiency=float(rendement_actuel), has_existing_ac=inputs.has_existing_ac, eer_current_ac=inputs.eer_current_ac, om_pac_chf_per_year=float(gshp.om_chf_per_year), om_ref_chf_per_year=float(om_systeme_actuel_chf_per_year), discount_rate=UNCERTAINTY_DISCOUNT_RATE)
    if is_replacement:
        for label, cfg_price in PRICE_SENSITIVITY_SCENARIOS.items():
            p_elec_s, p_ref_s = load_price_pair(cfg_price["electricity"], cfg_price["fuel"])
            series_s = build_cost_and_emission_series(p_elec=p_elec_s, p_actuel_heat=p_ref_s, om_systeme_actuel_chf_per_year=om_systeme_actuel_chf_per_year)
            eco_s = compute_economic_indicators(cost_ref_series=series_s["cost_actuel_total_series"], cost_pac_series=series_s["cost_pac_total_series"], capex_net=capex_net, discount_rate=UNCERTAINTY_DISCOUNT_RATE)
            price_sensitivity_results[label] = {
                "label": label,
                "description": cfg_price["description"],
                "electricity_scenario": cfg_price["electricity"],
                "fuel_scenario": cfg_price["fuel"],
                "cout_pac_total_year0": eco_s["cout_pac_year0"],
                "cout_actuel_total_year0": eco_s["cout_ref_year0"],
                "economies_annuelles_year0": eco_s["economies_year0"],
                "payback": eco_s["payback"],
                "npv": eco_s["npv"],
            }
    return {
        "inputs": inputs,
        "station_climatique": station_info,
        "chauffage": res_ch,
        "froid": {"cooling_hours": cooling_hours, "cooling_degree_hours": cooling_degree_hours, "cooling_mode_effective": effective_cooling_mode, **cool_res},
        "pac": {
            "p_nom_kw": p_max_kw,
            "spf": float(gshp.spf),
            "capex_brut": capex_brut,
            "subvention": subvention_pac,
            "capex_net": capex_net,
            "cout_annuel_total_annee_0": central_payload["cout_pac_total_year0"],
            "om_annuel": float(gshp.om_chf_per_year),
            "emissions_totales_kg": float(central_payload["emissions_pac_series"].iloc[0]) if central_payload.get("emissions_pac_series") is not None else None,
        },
        "confiance": compute_confidence_level(inputs=inputs, froid={"cooling_mode_effective": effective_cooling_mode, **cool_res}),
        "scenarios": scenario_results,
        "central": central_payload,
        "price_sensitivity": price_sensitivity_results,
        "parameter_sensitivity": parameter_sensitivity,
    }
