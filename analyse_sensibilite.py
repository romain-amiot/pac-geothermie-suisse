from __future__ import annotations

"""
analyse_cas_et_sensibilite_ecole.py

Script refondu pour le mémoire PAC géothermique.

A) Lance les 3 cas de référence avec Monte Carlo et exporte des graphiques :
   - temps de retour simulé ;
   - VAN simulée ;
   - coûts cumulés actualisés ;
   - émissions cumulées de CO2.

B) Lance une analyse de sensibilité déterministe uniquement sur un 4e cas :
   Ecole_GE_gaz.
Paramètres testés sur l'école :
   - prix de l'énergie actuelle : -30 % à +30 % par pas de 10 % ;
   - prix de l'électricité : -30 % à +30 % par pas de 10 % ;
   - isolation : tous les coefficients UBAT de la liste ;
   - COP PAC : 4.5 à 5.5 par pas de 0.1 ;
   - canton : 10 cantons représentatifs.

Sorties : outputs/analyse_cas_et_sensibilite_ecole/
"""

from pathlib import Path
from contextlib import contextmanager
from typing import Any
import importlib
import math
import sys

import pandas as pd
import matplotlib.pyplot as plt

DISCOUNT_RATE = 0.03
HORIZON_YEARS = 25
PAYBACK_MAX_YEARS = 50
RUN_MONTE_CARLO_FOR_REFERENCE_CASES = True


def find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, Path.cwd(), Path.cwd().parent]:
        if (candidate / "models.py").exists():
            return candidate
        if (candidate / "services" / "calcul_projet.py").exists() and (candidate / "models.py").exists():
            return candidate
    raise RuntimeError("Racine du projet introuvable. Lance le script depuis la racine du projet.")


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from models import ProjectInputs  # noqa: E402

try:
    calcul_module = importlib.import_module("services.calcul_projet")
except Exception:
    calcul_module = importlib.import_module("calcul_projet")

# ---------------------------------------------------------------------------
# Constantes alignées sur le code principal corrigé / mémoire
# ---------------------------------------------------------------------------

DEFAULT_VENTILATION_R = 0.20
DEFAULT_COP_PAC = 5.0

COOLING_SPF_BY_MODE = getattr(
    calcul_module,
    "COOLING_SPF_BY_MODE",
    {"free_cooling": 12.0, "hybrid": 8.0, "active_cooling": 5.0},
)

def cooling_mode_factor_from_main(cooling_mode: str) -> float:
    func = getattr(calcul_module, "cooling_mode_factor", None)
    if callable(func):
        return float(func(cooling_mode))
    return {
        "no_cooling": 0.0,
        "free_cooling": 0.40,
        "hybrid": 0.60,
        "active_cooling": 1.00,
    }.get(str(cooling_mode), 1.00)

UNCERTAINTY_DEFAULTS_MAIN = getattr(calcul_module, "UNCERTAINTY_DEFAULTS", {})
PRICE_ELEC_SIGMA = float(UNCERTAINTY_DEFAULTS_MAIN.get("price_elec_sigma", 0.107))
PRICE_ELEC_BOUNDS = (0.90, 1.30)
OM_RANGE = (
    float(UNCERTAINTY_DEFAULTS_MAIN.get("om_min", 0.80)),
    float(UNCERTAINTY_DEFAULTS_MAIN.get("om_mode", 1.00)),
    float(UNCERTAINTY_DEFAULTS_MAIN.get("om_max", 1.20)),
)

K_SDEP_BY_BUILDING_TYPE = getattr(
    calcul_module,
    "K_SDEP_BY_BUILDING_TYPE",
    {
        "maison_individuelle": 2.10,
        "residentiel_collectif": 1.92,
        "grand_batiment_compact": 1.71,
        "mixte": 1.69,
        "activites": 1.99,
        "equipement_collectif": 2.15,
    },
)

DEFAULT_GEOMETRY_BY_BUILDING_TYPE = getattr(
    calcul_module,
    "DEFAULT_GEOMETRY_BY_BUILDING_TYPE",
    {
        "maison_individuelle": {"hauteur_m": 2.50},
        "residentiel_collectif": {"hauteur_m": 2.50},
        "grand_batiment_compact": {"hauteur_m": 2.50},
        "mixte": {"hauteur_m": 2.70},
        "activites": {"hauteur_m": 2.75},
        "equipement_collectif": {"hauteur_m": 3.00},
    },
)

try:
    from Fonctions.puissance_pointe import UBAT_CHOICES
except Exception:
    UBAT_CHOICES = {
        "Exceptionnel": ("Exceptionnel", 0.30),
        "Très performant": ("Très performant", 0.40),
        "Années 2000": ("Années 2000", 0.75),
        "Années 1990": ("Années 1990", 0.95),
        "Années 1980": ("Années 1980", 1.15),
        "Années 1970": ("Années 1970", 1.40),
        "Peu isolé": ("Peu isolé", 1.80),
    }

OUT = PROJECT_ROOT / "outputs" / "analyse_cas_et_sensibilite_ecole"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilitaires robustes
# ---------------------------------------------------------------------------

def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def f_or_none(x: Any) -> float | None:
    return float(x) if is_finite(x) else None


def r(x: Any, n: int = 2) -> float | None:
    y = f_or_none(x)
    return round(y, n) if y is not None else None


def clean_series(x: Any, fill_missing_with_zero: bool = False) -> pd.Series:
    """
    Nettoie une série numérique sans transformer automatiquement les années
    manquantes en zéro.

    Remplacer les valeurs manquantes par zéro est dangereux pour les coûts
    énergétiques : cela peut créer artificiellement une année sans coût. On
    interpole donc les trous internes et on prolonge les extrémités. Le zéro
    n'est utilisé que pour les composantes explicitement nulles.
    """
    if x is None:
        return pd.Series(dtype=float)
    s = pd.Series(x).copy()
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([float("inf"), float("-inf")], pd.NA)

    if fill_missing_with_zero:
        return s.fillna(0.0).astype(float)

    if len(s) == 0:
        return pd.Series(dtype=float)

    s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
    return s.astype(float)


def align_series(*series: Any, fill_missing_with_zero: bool = False) -> list[pd.Series]:
    cleaned = [clean_series(s, fill_missing_with_zero=fill_missing_with_zero) for s in series]
    if not cleaned:
        return []

    idx = cleaned[0].index
    for s in cleaned[1:]:
        idx = idx.union(s.index)

    out: list[pd.Series] = []
    for s in cleaned:
        y = s.reindex(idx)
        if fill_missing_with_zero:
            y = y.fillna(0.0)
        else:
            y = y.interpolate(method="linear", limit_direction="both").ffill().bfill()
        out.append(y.astype(float))

    return out


def extend_series(s: pd.Series, n_years: int) -> pd.Series:
    s = clean_series(s)
    if len(s) == 0:
        return s
    if len(s) >= n_years:
        return s.iloc[:n_years].copy()
    last = float(s.iloc[-1])
    missing = n_years - len(s)
    start = int(s.index[-1]) + 1 if len(s.index) and isinstance(s.index[-1], (int, float)) else len(s)
    extra = pd.Series([last] * missing, index=range(start, start + missing))
    return pd.concat([s, extra])


def clean_price_series_for_mc(series: Any) -> pd.Series:
    """
    Nettoie une série de prix avant Monte Carlo.

    Sur certains scénarios fossiles, des années manquantes peuvent rendre la VAN
    Monte Carlo non finie, alors que le temps de retour reste calculable.
    On corrige donc les séries de prix utilisées dans le Monte Carlo en comblant
    les valeurs manquantes.
    """
    if series is None:
        return pd.Series(dtype=float)

    s = pd.Series(series).copy()
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([float("inf"), float("-inf")], pd.NA)
    s = s.interpolate(method="linear", limit_direction="both")
    s = s.ffill().bfill()

    return s.astype(float)


def safe_series_first_n_years_for_mc(series: Any, n_years: int) -> pd.Series:
    """
    Version robuste de calcul_projet._series_first_n_years pour l'analyse.

    Elle évite que des NaN présents dans une série de prix se propagent dans
    npvs_life et produisent une distribution de VAN inutilisable.
    """
    s = clean_price_series_for_mc(series)

    if s is None or len(s) == 0:
        raise ValueError("Série vide ou None.")

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


def discounted_cumulative(costs: pd.Series, initial: float = 0.0) -> pd.Series:
    costs = clean_series(costs)
    cumulative = float(initial)
    vals = []
    for i, val in enumerate(costs, start=1):
        cumulative += float(val) / ((1.0 + DISCOUNT_RATE) ** i)
        vals.append(cumulative)
    return pd.Series(vals, index=costs.index)


def discounted_payback(capex_net: float, cost_ref: pd.Series, cost_pac: pd.Series, max_years: int = PAYBACK_MAX_YEARS) -> float | None:
    capex = f_or_none(capex_net)
    if capex is None:
        return None
    cost_ref, cost_pac = align_series(cost_ref, cost_pac)
    cost_ref = extend_series(cost_ref, max_years)
    cost_pac = extend_series(cost_pac, max_years)
    cost_ref, cost_pac = align_series(cost_ref, cost_pac)
    cum = -capex
    for i, saving in enumerate(cost_ref - cost_pac, start=1):
        ds = float(saving) / ((1.0 + DISCOUNT_RATE) ** i)
        prev = cum
        cum += ds
        if cum >= 0:
            return float(i) if ds <= 0 else float((i - 1) + abs(prev) / ds)
    return None


def npv(capex_net: float, cost_ref: pd.Series, cost_pac: pd.Series, horizon: int = HORIZON_YEARS) -> float | None:
    capex = f_or_none(capex_net)
    if capex is None:
        return None
    cost_ref, cost_pac = align_series(cost_ref, cost_pac)
    cost_ref = extend_series(cost_ref, horizon)
    cost_pac = extend_series(cost_pac, horizon)
    cost_ref, cost_pac = align_series(cost_ref, cost_pac)
    val = -capex
    for i, saving in enumerate(cost_ref - cost_pac, start=1):
        val += float(saving) / ((1.0 + DISCOUNT_RATE) ** i)
    return float(val)


def safe_name(name: str) -> str:
    out = str(name)
    for a, b in [(" ", "_"), ("/", "_"), ("\\", "_"), (":", "_"), ("é", "e"), ("è", "e"), ("ê", "e"), ("à", "a"), ("ç", "c"), ("(", ""), (")", "")]:
        out = out.replace(a, b)
    return out


# ---------------------------------------------------------------------------
# Patchs temporaires : désactiver Monte Carlo / forcer le COP
# ---------------------------------------------------------------------------

@contextmanager
def patched_calculation(spf_override: float | None = None, disable_monte_carlo: bool = False):
    original_gshp = getattr(calcul_module, "couts_annuels_gshp_zuberi", None)
    original_mc = getattr(calcul_module, "monte_carlo_project_uncertainty", None)
    original_series_first = getattr(calcul_module, "_series_first_n_years", None)
    try:
        if original_series_first is not None:
            calcul_module._series_first_n_years = safe_series_first_n_years_for_mc
        if spf_override is not None and original_gshp is not None:
            def patched_gshp(*args: Any, **kwargs: Any):
                kwargs["spf"] = float(spf_override)
                return original_gshp(*args, **kwargs)
            calcul_module.couts_annuels_gshp_zuberi = patched_gshp
        if disable_monte_carlo and original_mc is not None:
            def disabled_mc(*args: Any, **kwargs: Any) -> dict[str, Any]:
                return {"available": False, "reason": "disabled_in_analysis_script"}
            calcul_module.monte_carlo_project_uncertainty = disabled_mc
        yield
    finally:
        if original_gshp is not None:
            calcul_module.couts_annuels_gshp_zuberi = original_gshp
        if original_mc is not None:
            calcul_module.monte_carlo_project_uncertainty = original_mc
        if original_series_first is not None:
            calcul_module._series_first_n_years = original_series_first


def evaluate(inputs: ProjectInputs, spf_override: float | None = None, run_mc: bool = False) -> dict[str, Any]:
    with patched_calculation(spf_override=spf_override, disable_monte_carlo=not run_mc):
        return calcul_module.evaluer_projet(inputs)



# ---------------------------------------------------------------------------
# Estimation typologique cohérente avec l'interface corrigée
# ---------------------------------------------------------------------------

MITOYENNETE_SDEP_FACTOR = {
    "isole": 1.00,
    "1_cote": 0.75,
    "2_cotes": 0.50,
    "3_cotes": 0.25,
}

TYPOLOGY_ALIAS_FOR_K = {
    "bureaux": "activites",
    "ecole": "equipement_collectif",
    "commerce": "activites",
}


def typology_key_for_k(building_type_key: str) -> str:
    return TYPOLOGY_ALIAS_FOR_K.get(building_type_key, building_type_key)


def default_height_for_typology(building_type_key: str) -> float:
    key = typology_key_for_k(building_type_key)
    data = DEFAULT_GEOMETRY_BY_BUILDING_TYPE.get(key, {})
    return float(data.get("hauteur_m", 2.70))


def estimate_typological_sdep_vh(
    *,
    shab_m2: float,
    building_type_key: str,
    niveaux: int,
    hauteur_m: float | None = None,
    mitoyennete: str = "isole",
    toiture_exposee: bool = True,
    plancher_expose: bool = True,
) -> dict[str, float]:
    """
    Même logique que l'interface corrigée :
    - Sdép typologique = K × Shab ;
    - empreinte au sol = Shab / nombre de niveaux saisi ;
    - décomposition vertical / toiture / plancher ;
    - correction de la partie verticale par hauteur / hauteur_ref ;
    - mitoyenneté appliquée uniquement à la partie verticale.
    """
    if shab_m2 <= 0:
        raise ValueError("shab_m2 doit être positif.")
    if niveaux <= 0:
        raise ValueError("niveaux doit être positif.")

    k_key = typology_key_for_k(building_type_key)
    if k_key not in K_SDEP_BY_BUILDING_TYPE:
        raise ValueError(f"Typologie inconnue pour K : {building_type_key!r}")

    if mitoyennete not in MITOYENNETE_SDEP_FACTOR:
        raise ValueError(f"Mitoyenneté inconnue : {mitoyennete!r}")

    hauteur_ref = default_height_for_typology(k_key)
    hauteur = float(hauteur_m if hauteur_m is not None else hauteur_ref)

    k_typ = float(K_SDEP_BY_BUILDING_TYPE[k_key])
    sdep_typ = k_typ * float(shab_m2)

    empreinte = float(shab_m2) / int(niveaux)
    s_horizontal_ref = 2.0 * empreinte
    s_vertical_ref = max(0.0, sdep_typ - s_horizontal_ref)

    facteur_hauteur = hauteur / hauteur_ref if hauteur_ref > 0 else 1.0
    s_vertical = s_vertical_ref * facteur_hauteur

    facteur_mitoyennete = float(MITOYENNETE_SDEP_FACTOR[mitoyennete])
    s_vertical_mit = s_vertical * facteur_mitoyennete

    s_toiture = empreinte if toiture_exposee else 0.0
    s_plancher = empreinte if plancher_expose else 0.0

    sdep = max(1.0, s_vertical_mit + s_toiture + s_plancher)
    vh = float(shab_m2) * hauteur

    return {
        "sdep_m2": float(sdep),
        "vh_m3": float(vh),
        "hauteur_m": float(hauteur),
        "hauteur_ref_m": float(hauteur_ref),
        "k_typologique": float(k_typ),
        "empreinte_sol_m2": float(empreinte),
        "s_vertical_ref_m2": float(s_vertical_ref),
        "s_vertical_mit_m2": float(s_vertical_mit),
        "s_toiture_m2": float(s_toiture),
        "s_plancher_m2": float(s_plancher),
    }


# ---------------------------------------------------------------------------
# Cas d'étude
# ---------------------------------------------------------------------------

def make_inputs(
    postcode: str,
    canton: str,
    building_type_key: str,
    shab_m2: float,
    ubat: float,
    current_energy: str,
    want_cooling: bool,
    surface_climatisee_m2: float,
    cooling_mode: str,
    niveaux: int,
    hauteur_m: float | None = None,
    mitoyennete: str = "isole",
    toiture_exposee: bool = True,
    plancher_expose: bool = True,
    ventilation_r: float = DEFAULT_VENTILATION_R,
    current_efficiency: float = 0.90,
    vitrage_level: str = "moyen",
    solar_protection_level: str = "moyenne",
    usage_level: str = "normal",
    night_ventilation: bool = False,
    has_existing_ac: bool = True,
    eer_current_ac: float | None = 3.0,
) -> ProjectInputs:
    geo = estimate_typological_sdep_vh(
        shab_m2=shab_m2,
        building_type_key=building_type_key,
        niveaux=niveaux,
        hauteur_m=hauteur_m,
        mitoyennete=mitoyennete,
        toiture_exposee=toiture_exposee,
        plancher_expose=plancher_expose,
    )

    return ProjectInputs(
        project_type="replacement",
        canton=canton,
        postcode=postcode,
        building_type_key=building_type_key,
        ubat=ubat,
        ventilation_r=ventilation_r,
        sdep_mode="Saisie directe",
        sdep_m2=geo["sdep_m2"],
        vh_m3=geo["vh_m3"],
        shab_m2=shab_m2,
        niveaux=int(niveaux),
        perimetre_m=None,
        hauteur_m=geo["hauteur_m"],
        toiture_exposee=toiture_exposee,
        plancher_expose=plancher_expose,
        forme_generale=None,
        mitoyennete=mitoyennete,
        longueur_m=None,
        largeur_m=None,
        current_energy=current_energy,
        current_efficiency=current_efficiency,
        want_cooling=want_cooling,
        surface_climatisee_m2=surface_climatisee_m2,
        cooling_mode=cooling_mode,
        vitrage_level=vitrage_level,
        solar_protection_level=solar_protection_level,
        usage_level=usage_level,
        night_ventilation=night_ventilation,
        has_existing_ac=has_existing_ac,
        eer_current_ac=eer_current_ac,
    )


def reference_cases() -> dict[str, ProjectInputs]:
    return {
        "Maison_GE_gaz": make_inputs(
            postcode="1200",
            canton="Genève",
            building_type_key="maison_individuelle",
            shab_m2=200,
            niveaux=2,
            hauteur_m=2.50,
            ubat=0.75,
            current_energy="Gaz naturel",
            want_cooling=True,
            surface_climatisee_m2=200,
            cooling_mode="hybrid",
            night_ventilation=True,
        ),
        "grand_immeuble_VD_mazout": make_inputs(
            postcode="1000",
            canton="Vaud",
            building_type_key="residentiel_collectif",
            shab_m2=2500,
            niveaux=8,
            hauteur_m=2.50,
            ubat=1.15,
            current_energy="Mazout (fioul)",
            want_cooling=False,
            surface_climatisee_m2=0,
            cooling_mode="no_cooling",
            has_existing_ac=False,
            eer_current_ac=None,
        ),
        "bureaux_ZH_gaz_froid": make_inputs(
            postcode="8000",
            canton="Zurich",
            building_type_key="bureaux",
            shab_m2=3500,
            niveaux=6,
            hauteur_m=2.75,
            ubat=0.95,
            current_energy="Gaz naturel",
            want_cooling=True,
            surface_climatisee_m2=2400,
            cooling_mode="active_cooling",
            vitrage_level="fort",
            usage_level="eleve",
            night_ventilation=False,
        ),
    }


def school_case() -> ProjectInputs:
    return make_inputs(
        postcode="1200",
        canton="Genève",
        building_type_key="equipement_collectif",
        shab_m2=2000,
        niveaux=3,
        hauteur_m=3.00,
        ubat=1.15,
        current_energy="Gaz naturel",
        want_cooling=True,
        surface_climatisee_m2=2000,
        cooling_mode="hybrid",
        vitrage_level="moyen",
        solar_protection_level="moyenne",
        usage_level="normal",
        night_ventilation=False,
        has_existing_ac=True,
        eer_current_ac=3.0,
    )


def clone_inputs(inputs: ProjectInputs, **updates: Any) -> ProjectInputs:
    if hasattr(inputs, "model_dump"):
        data = dict(inputs.model_dump())
    elif hasattr(inputs, "dict"):
        data = dict(inputs.dict())
    else:
        data = dict(inputs.__dict__)
    data.update(updates)
    return ProjectInputs(**data)


# ---------------------------------------------------------------------------
# Extraction séries/métriques
# ---------------------------------------------------------------------------

def cost_components(results: dict[str, Any]) -> dict[str, pd.Series] | None:
    c = results.get("central", {})
    if not isinstance(c, dict):
        return None
    keys = [
        "cost_actuel_heat_energy_series", "cost_actuel_cool_series", "om_actuel_series",
        "cost_pac_heat_energy_series", "cost_pac_cool_series", "om_pac_series",
    ]
    if all(c.get(k) is not None for k in keys):
        ref_heat, ref_cool, ref_om, pac_heat, pac_cool, pac_om = align_series(
            c["cost_actuel_heat_energy_series"], c["cost_actuel_cool_series"], c["om_actuel_series"],
            c["cost_pac_heat_energy_series"], c["cost_pac_cool_series"], c["om_pac_series"],
        )
        return {"ref_heat": ref_heat, "ref_cool": ref_cool, "ref_om": ref_om, "pac_heat": pac_heat, "pac_cool": pac_cool, "pac_om": pac_om}
    if c.get("cost_actuel_total_series") is not None and c.get("cost_pac_total_series") is not None:
        ref, pac = align_series(c["cost_actuel_total_series"], c["cost_pac_total_series"])
        zero_ref = pd.Series([0.0] * len(ref), index=ref.index)
        zero_pac = pd.Series([0.0] * len(pac), index=pac.index)
        return {"ref_heat": ref, "ref_cool": zero_ref, "ref_om": zero_ref, "pac_heat": pac, "pac_cool": zero_pac, "pac_om": zero_pac}
    return None


def total_costs(comp: dict[str, pd.Series], current_factor: float = 1.0, electricity_factor: float = 1.0) -> tuple[pd.Series, pd.Series]:
    ref_heat, ref_cool, ref_om, pac_heat, pac_cool, pac_om = align_series(
        comp["ref_heat"], comp["ref_cool"], comp["ref_om"], comp["pac_heat"], comp["pac_cool"], comp["pac_om"]
    )
    cost_ref = ref_heat * current_factor + ref_cool * electricity_factor + ref_om
    cost_pac = (pac_heat + pac_cool) * electricity_factor + pac_om
    return align_series(cost_ref, cost_pac)


def metrics(results: dict[str, Any]) -> dict[str, Any]:
    c = results.get("central", {}) if isinstance(results.get("central"), dict) else {}
    p = results.get("pac", {}) if isinstance(results.get("pac"), dict) else {}
    h = results.get("chauffage", {}) if isinstance(results.get("chauffage"), dict) else {}
    f = results.get("froid", {}) if isinstance(results.get("froid"), dict) else {}
    unc = c.get("uncertainty", {}) if isinstance(c.get("uncertainty"), dict) else {}

    capex = f_or_none(p.get("capex_net"))
    comp = cost_components(results)
    pb = f_or_none(c.get("payback"))
    van = f_or_none(c.get("npv"))
    if comp is not None and capex is not None:
        cr, cp = total_costs(comp)
        pb = pb if pb is not None else discounted_payback(capex, cr, cp)
        van = van if van is not None else npv(capex, cr, cp)

    return {
        "p_pointe_kw": f_or_none(h.get("p_pointe_kw")),
        "besoin_chauffage_kwh_an": f_or_none(h.get("energie_annuelle_kwh")),
        "besoin_froid_kwh_an": f_or_none(f.get("q_froid_utile_kwh_an")),
        "capex_brut_chf": f_or_none(p.get("capex_brut")),
        "subvention_chf": f_or_none(p.get("subvention")),
        "capex_net_chf": capex,
        "cout_actuel_annee_1_chf": f_or_none(c.get("cout_actuel_total_year0")),
        "cout_pac_annee_1_chf": f_or_none(c.get("cout_pac_total_year0")),
        "economies_annee_1_chf": f_or_none(c.get("economies_annuelles_year0")),
        "payback_ans": pb,
        "van_chf": van,
        "mc_payback_p50_ans": f_or_none(unc.get("payback_p50")),
        "mc_van_p50_chf": f_or_none(unc.get("npv_p50")),
        "mc_prob_amorti_25_ans": f_or_none(unc.get("probability_payback_le_25y", unc.get("amortization_probability_life"))),
    }


def summarize(case: str, inputs: ProjectInputs, results: dict[str, Any]) -> dict[str, Any]:
    m = metrics(results)
    out = {
        "cas": case,
        "postcode": inputs.postcode,
        "canton": inputs.canton,
        "typologie": inputs.building_type_key,
        "surface_chauffee_m2": inputs.shab_m2,
        "niveaux": inputs.niveaux,
        "hauteur_m": inputs.hauteur_m,
        "sdep_m2": inputs.sdep_m2,
        "vh_m3": inputs.vh_m3,
        "surface_climatisee_m2": inputs.surface_climatisee_m2,
        "energie_actuelle": inputs.current_energy,
        "ubat": inputs.ubat,
        "mode_froid": inputs.cooling_mode,
    }
    out.update({k: r(v, 2 if ("payback" in k or "kw" in k or "prob" in k) else 0) for k, v in m.items()})
    return out


# ---------------------------------------------------------------------------
# Graphiques cas de référence
# ---------------------------------------------------------------------------

def plot_cumulative_costs(case: str, results: dict[str, Any]) -> str | None:
    comp = cost_components(results)
    capex = f_or_none(results.get("pac", {}).get("capex_net"))
    if comp is None or capex is None:
        return None
    cr, cp = total_costs(comp)
    cum_ref = discounted_cumulative(cr, 0.0)
    cum_pac = discounted_cumulative(cp, capex)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(cum_ref.index, cum_ref.values, marker="o", label="Système actuel")
    ax.plot(cum_pac.index, cum_pac.values, marker="o", label="PAC géothermique")
    ax.set_title(f"Coûts cumulés actualisés — {case}")
    ax.set_xlabel("Année")
    ax.set_ylabel("Coût cumulé actualisé (CHF)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = FIG / f"couts_cumules_{safe_name(case)}.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return str(path)


def plot_cumulative_emissions(case: str, results: dict[str, Any]) -> str | None:
    c = results.get("central", {})
    if not isinstance(c, dict) or c.get("emissions_actuel_series") is None or c.get("emissions_pac_series") is None:
        return None
    ref, pac = align_series(c["emissions_actuel_series"], c["emissions_pac_series"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ref.index, ref.cumsum() / 1000.0, marker="o", label="Système actuel")
    ax.plot(pac.index, pac.cumsum() / 1000.0, marker="o", label="PAC géothermique")
    ax.set_title(f"Émissions cumulées de CO₂ — {case}")
    ax.set_xlabel("Année")
    ax.set_ylabel("Émissions cumulées (tCO₂e)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = FIG / f"emissions_cumulees_{safe_name(case)}.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return str(path)


def finite_list(values: Any) -> list[float]:
    """
    Filtre les listes de résultats Monte Carlo sans remplacer les NaN par 0.

    Important :
    clean_series() est utile pour les séries de coûts, car une composante absente
    peut être considérée comme nulle. Mais pour une distribution Monte Carlo,
    remplacer NaN par 0 crée un faux histogramme centré sur zéro.
    """
    if values is None:
        return []

    out: list[float] = []
    for x in values:
        if is_finite(x):
            out.append(float(x))
    return out


def plot_mc_payback(case: str, results: dict[str, Any]) -> str | None:
    c = results.get("central", {})
    unc = c.get("uncertainty", {}) if isinstance(c, dict) else {}
    samples = unc.get("simulation_samples", {}) if isinstance(unc, dict) else {}
    paybacks = finite_list(samples.get("paybacks_extended", []))
    if not paybacks:
        return None
    pb20, pb50, pb80 = f_or_none(unc.get("payback_p20")), f_or_none(unc.get("payback_p50")), f_or_none(unc.get("payback_p80"))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(paybacks, bins=min(50, max(20, int(len(paybacks) ** 0.5))), edgecolor="black", alpha=0.75)
    if pb50 is not None:
        ax.axvline(pb50, linewidth=2, label=f"Référence MC : médiane {pb50:.1f} ans")
    if pb20 is not None and pb80 is not None:
        ax.axvspan(pb20, pb80, alpha=0.18, label=f"P20–P80 : {pb20:.1f}–{pb80:.1f} ans")
    ax.axvline(HORIZON_YEARS, linestyle=":", linewidth=2, label=f"Horizon {HORIZON_YEARS} ans")
    ax.set_title(f"Temps de retour simulé — {case}")
    ax.set_xlabel("Temps de retour actualisé simulé (ans)")
    ax.set_ylabel("Nombre de simulations amorties")
    ax.grid(True, alpha=0.3); ax.legend()
    path = FIG / f"mc_payback_{safe_name(case)}.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return str(path)


def plot_mc_van(case: str, results: dict[str, Any]) -> str | None:
    c = results.get("central", {})
    unc = c.get("uncertainty", {}) if isinstance(c, dict) else {}
    samples = unc.get("simulation_samples", {}) if isinstance(unc, dict) else {}
    vans = finite_list(samples.get("npvs_life", []))
    if not vans:
        return None
    p10, p50, p90 = f_or_none(unc.get("npv_p10")), f_or_none(unc.get("npv_p50")), f_or_none(unc.get("npv_p90"))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(vans, bins=35, edgecolor="black", alpha=0.75)
    ax.axvline(0, linestyle=":", linewidth=2, label="VAN = 0")
    if p50 is not None:
        ax.axvline(p50, linewidth=2, label=f"Référence MC : VAN médiane {p50:,.0f} CHF".replace(",", " "))
    if p10 is not None and p90 is not None:
        ax.axvspan(p10, p90, alpha=0.18, label="P10–P90")
    ax.set_title(f"VAN simulée — {case}")
    ax.set_xlabel("VAN simulée sur 25 ans (CHF)")
    ax.set_ylabel("Nombre de simulations")
    ax.grid(True, alpha=0.3); ax.legend()
    path = FIG / f"mc_van_{safe_name(case)}.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------------
# Sensibilité école
# ---------------------------------------------------------------------------

CANTONS = {
    "BE": ("3000", "Berne"), "FR": ("1700", "Fribourg"), "GE": ("1200", "Genève"),
    "JU": ("2800", "Jura"), "LU": ("6000", "Lucerne"), "NE": ("2000", "Neuchâtel"),
    "SG": ("9000", "Saint-Gall"), "VD": ("1000", "Vaud"), "ZG": ("6300", "Zoug"), "ZH": ("8000", "Zurich"),
}


def ubat_values() -> list[tuple[str, float]]:
    vals = []
    seen = set()
    for key, val in UBAT_CHOICES.items():
        try:
            label, ub = val[0], float(val[1])
        except Exception:
            label, ub = str(key), float(val)
        if ub not in seen:
            seen.add(ub)
            vals.append((label, ub))
    return sorted(vals, key=lambda x: x[1])


def adjusted_price_metrics(base_results: dict[str, Any], current_pct: float = 0.0, elec_pct: float = 0.0) -> dict[str, Any]:
    comp = cost_components(base_results)
    capex = f_or_none(base_results.get("pac", {}).get("capex_net"))
    if comp is None or capex is None:
        return {"payback_ans": None, "van_chf": None, "economies_annee_1_chf": None}
    cr, cp = total_costs(comp, 1.0 + current_pct / 100.0, 1.0 + elec_pct / 100.0)
    return {
        "payback_ans": discounted_payback(capex, cr, cp),
        "van_chf": npv(capex, cr, cp),
        "economies_annee_1_chf": float(cr.iloc[0] - cp.iloc[0]) if len(cr) else None,
    }


def run_school_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    errors = []
    base_inputs = school_case()
    base_results = evaluate(base_inputs, run_mc=False)
    base_df = pd.DataFrame([{k: r(v, 2 if ("payback" in k or "kw" in k) else 0) for k, v in metrics(base_results).items()}])
    base_df.insert(0, "cas", "Ecole_GE_gaz")

    rows = []

    for pct in range(-30, 31, 10):
        m = adjusted_price_metrics(base_results, current_pct=float(pct), elec_pct=0.0)
        rows.append({"parametre": "Prix énergie actuelle", "valeur": pct, "valeur_label": f"{pct:+d} %", "type": "numeric", "amplitude_testee": "-30 % à +30 %, pas de 10 %", **{k: r(v, 2 if "payback" in k else 0) for k, v in m.items()}})

    for pct in range(-30, 31, 10):
        m = adjusted_price_metrics(base_results, current_pct=0.0, elec_pct=float(pct))
        rows.append({"parametre": "Prix électricité", "valeur": pct, "valeur_label": f"{pct:+d} %", "type": "numeric", "amplitude_testee": "-30 % à +30 %, pas de 10 %", **{k: r(v, 2 if "payback" in k else 0) for k, v in m.items()}})

    for label, ub in ubat_values():
        try:
            res = evaluate(clone_inputs(base_inputs, ubat=ub), run_mc=False)
            m = metrics(res)
            rows.append({"parametre": "Isolation / UBAT", "valeur": ub, "valeur_label": f"{ub:.2f}", "label": label, "type": "numeric", "amplitude_testee": "tous les coefficients UBAT de la liste", "payback_ans": r(m.get("payback_ans"), 2), "van_chf": r(m.get("van_chf"), 0), "economies_annee_1_chf": r(m.get("economies_annee_1_chf"), 0)})
        except Exception as exc:
            errors.append({"parametre": "Isolation / UBAT", "valeur": ub, "erreur": repr(exc)})

    for cop in [round(i / 10.0, 1) for i in range(45, 56)]:
        try:
            res = evaluate(base_inputs, spf_override=cop, run_mc=False)
            m = metrics(res)
            rows.append({
                "parametre": "COP PAC",
                "valeur": cop,
                "valeur_label": f"{cop:.1f}",
                "type": "numeric",
                "amplitude_testee": "4.5 à 5.5 par pas de 0.1",
                "payback_ans": r(m.get("payback_ans"), 2),
                "van_chf": r(m.get("van_chf"), 0),
                "economies_annee_1_chf": r(m.get("economies_annee_1_chf"), 0),
            })
        except Exception as exc:
            errors.append({"parametre": "COP PAC", "valeur": cop, "erreur": repr(exc)})

    for abbr, (pc, canton) in CANTONS.items():
        try:
            res = evaluate(clone_inputs(base_inputs, postcode=pc, canton=canton), run_mc=False)
            m = metrics(res)
            rows.append({"parametre": "Canton", "valeur": abbr, "valeur_label": abbr, "type": "categorical", "amplitude_testee": "10 cantons/codes postaux représentatifs", "canton": canton, "postcode": pc, "payback_ans": r(m.get("payback_ans"), 2), "van_chf": r(m.get("van_chf"), 0), "economies_annee_1_chf": r(m.get("economies_annee_1_chf"), 0)})
        except Exception as exc:
            errors.append({"parametre": "Canton", "valeur": abbr, "erreur": repr(exc)})

    df = pd.DataFrame(rows)
    ranking = []
    for param, g in df.groupby("parametre"):
        pb = pd.to_numeric(g["payback_ans"], errors="coerce").dropna()
        van_s = pd.to_numeric(g["van_chf"], errors="coerce").dropna()
        ranking.append({
            "parametre": param,
            "amplitude_testee": g["amplitude_testee"].iloc[0],
            "payback_min_ans": r(pb.min(), 2) if len(pb) else None,
            "payback_max_ans": r(pb.max(), 2) if len(pb) else None,
            "impact_payback_ans": r(pb.max() - pb.min(), 2) if len(pb) >= 2 else None,
            "van_min_chf": r(van_s.min(), 0) if len(van_s) else None,
            "van_max_chf": r(van_s.max(), 0) if len(van_s) else None,
            "impact_van_chf": r(van_s.max() - van_s.min(), 0) if len(van_s) >= 2 else None,
        })
    rank_df = pd.DataFrame(ranking).sort_values("impact_payback_ans", ascending=False)
    return base_df, df, rank_df, errors


def plot_school_parameter_payback(df: pd.DataFrame) -> list[str]:
    paths = []
    for param, g in df.groupby("parametre"):
        g = g.copy()
        g["payback_ans"] = pd.to_numeric(g["payback_ans"], errors="coerce")
        g = g.dropna(subset=["payback_ans"])
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 5))
        if g["type"].iloc[0] == "numeric":
            g["x"] = pd.to_numeric(g["valeur"], errors="coerce")
            g = g.sort_values("x")
            ax.plot(g["x"], g["payback_ans"], marker="o")
            ax.set_xlabel(param)
        else:
            g = g.sort_values("valeur_label")
            ax.bar(g["valeur_label"].astype(str), g["payback_ans"])
            ax.set_xlabel(param)
            ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("Temps de retour actualisé (ans)")
        ax.set_title(f"Effet de {param} sur le temps de retour — École GE gaz\nAmplitude testée : {g['amplitude_testee'].iloc[0]}")
        ax.grid(True, axis="y", alpha=0.3)
        path = FIG / f"sensibilite_ecole_payback_{safe_name(param)}.png"
        fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
        paths.append(str(path))
    return paths


def plot_school_rankings(rank_df: pd.DataFrame) -> dict[str, str | None]:
    out = {"payback": None, "van": None}
    if not rank_df.empty:
        for metric, xlabel, fname in [
            ("impact_payback_ans", "Amplitude de variation du temps de retour (ans)", "sensibilite_ecole_ranking_payback.png"),
            ("impact_van_chf", "Amplitude de variation de la VAN (CHF)", "sensibilite_ecole_ranking_van.png"),
        ]:
            d = rank_df.dropna(subset=[metric]).sort_values(metric, ascending=True)
            if d.empty:
                continue
            fig, ax = plt.subplots(figsize=(8.5, 5))
            ax.barh(d["parametre"], d[metric])
            ax.set_xlabel(xlabel)
            ax.set_title("Influence globale des paramètres — École GE gaz")
            ax.grid(True, axis="x", alpha=0.3)
            path = FIG / fname
            fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
            out["payback" if "payback" in metric else "van"] = str(path)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Supprime les anciens PNG pour éviter de relire un graphique obsolète
    # si un graphique n'est pas régénéré.
    for old_png in FIG.glob("*.png"):
        try:
            old_png.unlink()
        except Exception:
            pass

    errors = []
    ref_rows = []
    graph_rows = []

    for case, inputs in reference_cases().items():
        print(f"\n=== Cas de référence : {case} ===")
        try:
            res = evaluate(inputs, run_mc=RUN_MONTE_CARLO_FOR_REFERENCE_CASES)
            ref_rows.append(summarize(case, inputs, res))
            graph_rows.append({
                "cas": case,
                "graph_couts_cumules": plot_cumulative_costs(case, res),
                "graph_emissions_cumulees": plot_cumulative_emissions(case, res),
                "graph_mc_payback": plot_mc_payback(case, res),
                "graph_mc_van": plot_mc_van(case, res),
            })
        except Exception as exc:
            errors.append({"partie": "reference", "cas": case, "parametre": "", "valeur": "", "erreur": repr(exc)})
            print(f"ERREUR {case}: {exc}")

    print("\n=== Sensibilité école GE gaz ===")
    try:
        school_base, sens_df, rank_df, school_errors = run_school_sensitivity()
        errors.extend([{"partie": "sensibilite_ecole", "cas": "Ecole_GE_gaz", **e} for e in school_errors])
        param_graphs = plot_school_parameter_payback(sens_df)
        global_graphs = plot_school_rankings(rank_df)
        school_graphs = pd.DataFrame([{
            "cas": "Ecole_GE_gaz",
            "graphs_payback_parametres": ";".join(param_graphs),
            "graph_ranking_payback": global_graphs.get("payback"),
            "graph_ranking_van": global_graphs.get("van"),
        }])
    except Exception as exc:
        errors.append({"partie": "sensibilite_ecole", "cas": "Ecole_GE_gaz", "parametre": "", "valeur": "", "erreur": repr(exc)})
        school_base = pd.DataFrame(); sens_df = pd.DataFrame(); rank_df = pd.DataFrame(); school_graphs = pd.DataFrame()
        print(f"ERREUR sensibilité école: {exc}")

    # Métadonnées de cohérence avec le code principal corrigé.
    pd.DataFrame([
        {"parametre": "ventilation_r", "valeur": DEFAULT_VENTILATION_R, "source": "mémoire / app corrigée"},
        {"parametre": "price_elec_sigma", "valeur": PRICE_ELEC_SIGMA, "source": "UNCERTAINTY_DEFAULTS du code principal"},
        {"parametre": "price_elec_min", "valeur": PRICE_ELEC_BOUNDS[0], "source": "calibration ElCom 2011--2026"},
        {"parametre": "price_elec_max", "valeur": PRICE_ELEC_BOUNDS[1], "source": "calibration ElCom 2011--2026"},
        {"parametre": "om_min", "valeur": OM_RANGE[0], "source": "UNCERTAINTY_DEFAULTS du code principal"},
        {"parametre": "om_mode", "valeur": OM_RANGE[1], "source": "UNCERTAINTY_DEFAULTS du code principal"},
        {"parametre": "om_max", "valeur": OM_RANGE[2], "source": "UNCERTAINTY_DEFAULTS du code principal"},
        {"parametre": "spf_froid_free_cooling", "valeur": COOLING_SPF_BY_MODE.get("free_cooling"), "source": "COOLING_SPF_BY_MODE du code principal"},
        {"parametre": "spf_froid_hybrid", "valeur": COOLING_SPF_BY_MODE.get("hybrid"), "source": "COOLING_SPF_BY_MODE du code principal"},
        {"parametre": "spf_froid_active_cooling", "valeur": COOLING_SPF_BY_MODE.get("active_cooling"), "source": "COOLING_SPF_BY_MODE du code principal"},
        {"parametre": "f_mode_free_cooling", "valeur": cooling_mode_factor_from_main("free_cooling"), "source": "cooling_mode_factor du code principal"},
        {"parametre": "f_mode_hybrid", "valeur": cooling_mode_factor_from_main("hybrid"), "source": "cooling_mode_factor du code principal"},
        {"parametre": "f_mode_active_cooling", "valeur": cooling_mode_factor_from_main("active_cooling"), "source": "cooling_mode_factor du code principal"},
    ]).to_csv(OUT / "06_parametres_alignement_code_principal.csv", sep=";", index=False, encoding="utf-8-sig")

    pd.DataFrame(ref_rows).to_csv(OUT / "00_reference_cases_resume.csv", sep=";", index=False, encoding="utf-8-sig")
    pd.DataFrame(graph_rows).to_csv(OUT / "01_reference_cases_graphs.csv", sep=";", index=False, encoding="utf-8-sig")
    school_base.to_csv(OUT / "02_ecole_base_resume.csv", sep=";", index=False, encoding="utf-8-sig")
    sens_df.to_csv(OUT / "03_ecole_sensibilite_detail.csv", sep=";", index=False, encoding="utf-8-sig")
    rank_df.to_csv(OUT / "04_ecole_sensibilite_ranking.csv", sep=";", index=False, encoding="utf-8-sig")
    school_graphs.to_csv(OUT / "05_ecole_sensibilite_graphs.csv", sep=";", index=False, encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(OUT / "99_erreurs.csv", sep=";", index=False, encoding="utf-8-sig")

    print("\nAnalyse terminée.")
    print(f"Sorties : {OUT}")
    print(f"Figures : {FIG}")
    if errors:
        print("Des erreurs ont été enregistrées dans 99_erreurs.csv")


if __name__ == "__main__":
    main()
