from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math
import unicodedata

import pandas as pd


# ============================================================
#  Paramètres fixes
# ============================================================

DISCOUNT_RATE = 0.03
LIFETIME_YEARS = 25
OM_FRACTION = 0.01
GSHP_SPF = 5


# ============================================================
#  Types
# ============================================================

Scenario = Literal["optimistic", "neutral", "pessimistic"]
ProjectType = Literal["replacement", "new_building"]


# ============================================================
#  Paramètres subventions PAC géothermiques
# ============================================================

# Aucun forfait arbitraire n'est appliqué si le canton n'est pas renseigné.
DEFAULT_SUBSIDY_FIXED_CHF = 0.0
DEFAULT_SUBSIDY_PER_KW = 0.0
DEFAULT_SUBSIDY_CAPEX_SHARE = 0.40

ELIGIBLE_OLD_SYSTEMS_FOR_SUBSIDY = {"Gaz naturel", "Mazout (fioul)"}

# Barèmes cantonaux simplifiés pour PAC sol/eau ou eau/eau.
# P désigne la puissance de référence en kW.
#
# Types de règles :
# - linear:        S = fixed_chf + per_kw_chf * P
# - threshold:     S = low_fixed_chf si P <= threshold_kw, sinon fixed_chf + per_kw_chf * P
# - threshold_add: S = low_fixed_chf si P <= threshold_kw, sinon low_fixed_chf + per_kw_chf * (P - threshold_kw)
# - above_threshold_add: S = fixed_chf + per_kw_chf * max(P - threshold_kw, 0)
# - flat:          S = fixed_chf
#
# Remarque : certains cantons utilisent normalement la SRE ou le type de bâtiment.
# Quand ces informations ne sont pas disponibles dans l'outil, une approximation prudente
# est utilisée, documentée dans le mémoire.
GEOTHERMAL_SUBSIDY_BY_CANTON = {
    # Formules directement exprimables en CHF + CHF/kW
    "Geneva": {"type": "linear", "fixed_chf": 3_000.0, "per_kw_chf": 800.0},
    "Genève": {"type": "linear", "fixed_chf": 3_000.0, "per_kw_chf": 800.0},
    "Fribourg": {"type": "linear", "fixed_chf": 5_000.0, "per_kw_chf": 300.0},
    "Neuchâtel": {"type": "linear", "fixed_chf": 8_000.0, "per_kw_chf": 400.0},
    "Jura": {"type": "linear", "fixed_chf": 5_000.0, "per_kw_chf": 180.0},
    "Aargau": {"type": "linear", "fixed_chf": 6_000.0, "per_kw_chf": 180.0},
    "Argovie": {"type": "linear", "fixed_chf": 6_000.0, "per_kw_chf": 180.0},
    "Schwyz": {"type": "linear", "fixed_chf": 4_800.0, "per_kw_chf": 360.0},
    "Solothurn": {"type": "linear", "fixed_chf": 6_000.0, "per_kw_chf": 450.0},
    "Soleure": {"type": "linear", "fixed_chf": 6_000.0, "per_kw_chf": 450.0},
    "Glarus": {"type": "linear", "fixed_chf": 6_000.0, "per_kw_chf": 250.0},
    "Glaris": {"type": "linear", "fixed_chf": 6_000.0, "per_kw_chf": 250.0},
    "Nidwalden": {"type": "linear", "fixed_chf": 4_800.0, "per_kw_chf": 360.0},
    "Nidwald": {"type": "linear", "fixed_chf": 4_800.0, "per_kw_chf": 360.0},
    "Zug": {"type": "linear", "fixed_chf": 20_000.0, "per_kw_chf": 400.0},
    "Zoug": {"type": "linear", "fixed_chf": 20_000.0, "per_kw_chf": 400.0},

    # Barèmes avec seuils
    "Vaud": {"type": "threshold", "threshold_kw": 20.0, "low_fixed_chf": 20_000.0, "fixed_chf": 4_000.0, "per_kw_chf": 800.0},
    "Bern": {"type": "threshold", "threshold_kw": 15.0, "low_fixed_chf": 10_000.0, "fixed_chf": 4_800.0, "per_kw_chf": 360.0},
    "Berne": {"type": "threshold", "threshold_kw": 15.0, "low_fixed_chf": 10_000.0, "fixed_chf": 4_800.0, "per_kw_chf": 360.0},
    "Zurich": {"type": "threshold_add", "threshold_kw": 15.0, "low_fixed_chf": 6_800.0, "per_kw_chf": 420.0},
    "Zürich": {"type": "threshold_add", "threshold_kw": 15.0, "low_fixed_chf": 6_800.0, "per_kw_chf": 420.0},
    "St Gallen": {"type": "threshold", "threshold_kw": 20.0, "low_fixed_chf": 6_000.0, "fixed_chf": 2_400.0, "per_kw_chf": 180.0},
    "St. Gallen": {"type": "threshold", "threshold_kw": 20.0, "low_fixed_chf": 6_000.0, "fixed_chf": 2_400.0, "per_kw_chf": 180.0},
    "St-Gall": {"type": "threshold", "threshold_kw": 20.0, "low_fixed_chf": 6_000.0, "fixed_chf": 2_400.0, "per_kw_chf": 180.0},
    "Saint-Gall": {"type": "threshold", "threshold_kw": 20.0, "low_fixed_chf": 6_000.0, "fixed_chf": 2_400.0, "per_kw_chf": 180.0},
    "Lucerne": {"type": "threshold", "threshold_kw": 15.0, "low_fixed_chf": 8_500.0, "fixed_chf": 4_000.0, "per_kw_chf": 300.0},
    "Luzern": {"type": "threshold", "threshold_kw": 15.0, "low_fixed_chf": 8_500.0, "fixed_chf": 4_000.0, "per_kw_chf": 300.0},

    # Barèmes dépendant partiellement du type de bâtiment ou de la surface.
    # L'outil ne dispose pas toujours de ces variables au moment du calcul de subvention ;
    # on retient donc une règle simplifiée, sans créer de valeur par défaut pour les autres cantons.
    "Valais": {"type": "flat", "fixed_chf": 13_000.0},
    "Grisons": {"type": "flat", "fixed_chf": 17_500.0},
    "Graubünden": {"type": "flat", "fixed_chf": 17_500.0},
    "Thurgau": {"type": "above_threshold_add", "threshold_kw": 20.0, "fixed_chf": 9_000.0, "per_kw_chf": 300.0},
    "Thurgovie": {"type": "above_threshold_add", "threshold_kw": 20.0, "fixed_chf": 9_000.0, "per_kw_chf": 300.0},
}


# ============================================================
#  O&M annuel fondé sur ancrages JRC (proxy Allemagne)
# ============================================================

OM_JRC_ANCHORS = {
    "Gaz naturel": {"p1_kw": 30.0, "om1": 226.0, "p2_kw": 400.0, "om2": 2200.0},
    "Mazout (fioul)": {"p1_kw": 30.0, "om1": 185.0, "p2_kw": 400.0, "om2": 1650.0},
    "PAC géothermique": {"p1_kw": 15.0, "om1": 300.0, "p2_kw": 220.0, "om2": 600.0},
    "Électricité (réseau)": {"p1_kw": 20.0, "om1": 30.0, "p2_kw": 120.0, "om2": 60.0},
}


# ============================================================
#  Helpers
# ============================================================

def _slug(text: str) -> str:
    s = str(text).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    for old, new in {
        "-": " ",
        "_": " ",
        ".": " ",
        "'": " ",
        "’": " ",
        "(": " ",
        ")": " ",
        "/": " ",
    }.items():
        s = s.replace(old, new)
    return " ".join(s.split())


def annuity_factor(r: float = DISCOUNT_RATE, n_years: int = LIFETIME_YEARS) -> float:
    if r <= 0:
        raise ValueError("Le taux d'actualisation r doit être > 0.")
    if n_years <= 0:
        raise ValueError("La durée de vie n_years doit être > 0.")

    x = (1.0 + r) ** n_years
    return (r * x) / (x - 1.0)


def normalize_canton_name(canton: str) -> str:
    """
    Normalise un canton vers les noms utilisés par :
    - les subventions,
    - le CSV des prix électricité.
    """
    aliases = {
        "ge": "Geneva",
        "geneve": "Geneva",
        "geneva": "Geneva",

        "vd": "Vaud",
        "vaud": "Vaud",

        "be": "Bern",
        "berne": "Bern",
        "bern": "Bern",

        "fr": "Fribourg",
        "fribourg": "Fribourg",

        "vs": "Valais",
        "valais": "Valais",

        "ne": "Neuchâtel",
        "neuchatel": "Neuchâtel",

        "ju": "Jura",
        "jura": "Jura",

        "zh": "Zurich",
        "zurich": "Zurich",
        "zurich ": "Zurich",

        "ag": "Aargau",
        "argovie": "Aargau",
        "aargau": "Aargau",

        "sg": "St Gallen",
        "saint gall": "St Gallen",
        "saint gallen": "St Gallen",
        "st gall": "St Gallen",
        "st gallen": "St Gallen",

        "tg": "Thurgau",
        "thurgovie": "Thurgau",
        "thurgau": "Thurgau",

        "sz": "Schwyz",
        "schwyz": "Schwyz",
        "schwytz": "Schwyz",

        "so": "Solothurn",
        "soleure": "Solothurn",
        "solothurn": "Solothurn",

        "gr": "Grisons",
        "grisons": "Grisons",
        "graubunden": "Grisons",
        "graubuenden": "Grisons",

        "lu": "Lucerne",
        "lucerne": "Lucerne",
        "luzern": "Lucerne",

        "ur": "Uri",
        "uri": "Uri",

        "ow": "Obwalden",
        "obwald": "Obwalden",
        "obwalden": "Obwalden",

        "nw": "Nidwalden",
        "nidwald": "Nidwalden",
        "nidwalden": "Nidwalden",

        "gl": "Glarus",
        "glaris": "Glarus",
        "glarus": "Glarus",

        "zg": "Zug",
        "zoug": "Zug",
        "zug": "Zug",

        "ti": "Ticino",
        "tessin": "Ticino",
        "ticino": "Ticino",

        "ai": "Appenzell Innerrhoden",
        "appenzell innerrhoden": "Appenzell Innerrhoden",
        "appenzell rhodes interieures": "Appenzell Innerrhoden",

        "ar": "Appenzell Ausserrhoden",
        "appenzell ausserrhoden": "Appenzell Ausserrhoden",
        "appenzell rhodes exterieures": "Appenzell Ausserrhoden",

        "bs": "Basel Stadt",
        "bale ville": "Basel Stadt",
        "basel stadt": "Basel Stadt",

        "bl": "Basel Landschaft",
        "bale campagne": "Basel Landschaft",
        "basel landschaft": "Basel Landschaft",

        "sh": "Schaffhausen",
        "schaffhouse": "Schaffhausen",
        "schaffhausen": "Schaffhausen",
    }

    key = _slug(canton)
    return aliases.get(key, str(canton).strip())


def normalize_building_type(building_type: str) -> str:
    aliases = {
        "maison_individuelle": "maison_individuelle",
        "maison individuelle": "maison_individuelle",
        "villa": "maison_individuelle",
        "single family": "maison_individuelle",

        "appartement": "appartement",
        "appartements": "appartement",
        "immeuble": "appartement",
        "logement collectif": "appartement",
    }
    key = _slug(building_type)
    return aliases.get(key, str(building_type).strip())


def normalize_scenario_name(scenario: str) -> str:
    aliases = {
        "optimistic": "optimistic",
        "optimiste": "optimistic",
        "favorable": "optimistic",
        "favorable a la pac": "optimistic",

        "neutral": "neutral",
        "neutre": "neutral",

        "pessimistic": "pessimistic",
        "pessimiste": "pessimistic",
        "defavorable": "pessimistic",
        "defavorable a la pac": "pessimistic",
    }
    key = _slug(scenario)
    return aliases.get(key, str(scenario).strip())


def validate_project_type(project_type: ProjectType) -> None:
    if project_type not in {"replacement", "new_building"}:
        raise ValueError(
            "project_type invalide. Valeurs attendues : "
            "'replacement' ou 'new_building'."
        )


def project_has_existing_system(project_type: ProjectType) -> bool:
    validate_project_type(project_type)
    return project_type == "replacement"


def project_is_new_building(project_type: ProjectType) -> bool:
    validate_project_type(project_type)
    return project_type == "new_building"


def project_requires_reference_system(project_type: ProjectType) -> bool:
    validate_project_type(project_type)
    return project_type == "replacement"


def _interp_loglog(x: float, x1: float, y1: float, x2: float, y2: float) -> float:
    if x <= 0:
        raise ValueError("x doit être > 0.")
    if min(x1, y1, x2, y2) <= 0:
        raise ValueError("Les paramètres d'interpolation doivent être > 0.")

    if x <= x1:
        return float(y1)
    if x >= x2:
        return float(y2)

    b = math.log(y2 / y1) / math.log(x2 / x1)
    a = y1 / (x1 ** b)
    return float(a * (x ** b))


def annual_om_cost_chf(system_label: str, p_nom_kw: float) -> float:
    if p_nom_kw <= 0:
        raise ValueError("p_nom_kw doit être > 0.")

    anchors = OM_JRC_ANCHORS.get(system_label)
    if anchors is None:
        raise ValueError(f"Système d'entretien inconnu: {system_label}")

    return _interp_loglog(
        float(p_nom_kw),
        float(anchors["p1_kw"]),
        float(anchors["om1"]),
        float(anchors["p2_kw"]),
        float(anchors["om2"]),
    )


# ============================================================
#  CAPEX GSHP calibré
# ============================================================

GSHP_CAPEX_FIXED_CHF = 15_000.0
GSHP_HP_VAR_CHF_PER_KW = 1_600.0
GSHP_PROBE_COST_CHF_PER_M = 85.0
GSHP_PROBE_M_PER_KW = 15.0


def gshp_capex_chf(
    p_nom_kw: float,
    *,
    fixed_chf: float = GSHP_CAPEX_FIXED_CHF,
    hp_var_chf_per_kw: float = GSHP_HP_VAR_CHF_PER_KW,
    probe_cost_chf_per_m: float = GSHP_PROBE_COST_CHF_PER_M,
    probe_m_per_kw: float = GSHP_PROBE_M_PER_KW,
) -> float:
    if p_nom_kw <= 0:
        raise ValueError("La puissance nominale p_nom_kw doit être > 0.")

    var_chf_per_kw = hp_var_chf_per_kw + probe_cost_chf_per_m * probe_m_per_kw
    return fixed_chf + var_chf_per_kw * p_nom_kw


def _compute_subsidy_from_rule(rule: dict, p_nom_kw: float) -> float:
    """
    Calcule la subvention brute à partir d'une règle cantonale simplifiée.
    
    Les règles sont volontairement limitées aux formes nécessaires au modèle :
    linéaire, seuil forfaitaire, supplément au-delà d'un seuil, ou forfait simple.
    """
    rule_type = str(rule.get("type", "linear"))
    p = float(p_nom_kw)

    if rule_type == "linear":
        return float(rule.get("fixed_chf", 0.0)) + float(rule.get("per_kw_chf", 0.0)) * p

    if rule_type == "threshold":
        threshold = float(rule["threshold_kw"])
        if p <= threshold:
            return float(rule["low_fixed_chf"])
        return float(rule.get("fixed_chf", 0.0)) + float(rule.get("per_kw_chf", 0.0)) * p

    if rule_type == "threshold_add":
        threshold = float(rule["threshold_kw"])
        low_fixed = float(rule["low_fixed_chf"])
        if p <= threshold:
            return low_fixed
        return low_fixed + float(rule.get("per_kw_chf", 0.0)) * (p - threshold)

    if rule_type == "above_threshold_add":
        threshold = float(rule["threshold_kw"])
        return float(rule.get("fixed_chf", 0.0)) + float(rule.get("per_kw_chf", 0.0)) * max(p - threshold, 0.0)

    if rule_type == "flat":
        return float(rule.get("fixed_chf", 0.0))

    raise ValueError(f"Type de règle de subvention inconnu: {rule_type}")


def estimate_geothermal_subsidy_chf(
    canton: str,
    p_nom_kw: float,
    *,
    project_type: ProjectType = "replacement",
    capex_chf: float | None = None,
    current_energy_label: str | None = None,
    capex_share_limit: float = DEFAULT_SUBSIDY_CAPEX_SHARE,
) -> float:
    if p_nom_kw <= 0:
        raise ValueError("p_nom_kw doit être > 0.")

    validate_project_type(project_type)

    # Pas de subvention pour les constructions neuves dans ce modèle.
    if project_type == "new_building":
        return 0.0

    # Modèle volontairement restrictif : seules les substitutions gaz/mazout sont subventionnées.
    if current_energy_label not in ELIGIBLE_OLD_SYSTEMS_FOR_SUBSIDY:
        return 0.0

    canton_norm = normalize_canton_name(canton)
    rule = GEOTHERMAL_SUBSIDY_BY_CANTON.get(canton_norm)

    # Correction importante : aucun forfait arbitraire n'est appliqué
    # si le canton n'a pas de règle explicitement renseignée.
    if rule is None:
        return 0.0

    subsidy = _compute_subsidy_from_rule(rule, p_nom_kw)

    if capex_chf is not None:
        if capex_chf <= 0:
            raise ValueError("capex_chf doit être > 0 si fourni.")
        subsidy = min(subsidy, float(capex_share_limit) * float(capex_chf))

    return max(0.0, subsidy)


def gshp_capex_net_after_subsidy(
    p_nom_kw: float,
    canton: str,
    *,
    project_type: ProjectType = "replacement",
    current_energy_label: str | None = None,
    fixed_chf: float = GSHP_CAPEX_FIXED_CHF,
    hp_var_chf_per_kw: float = GSHP_HP_VAR_CHF_PER_KW,
    probe_cost_chf_per_m: float = GSHP_PROBE_COST_CHF_PER_M,
    probe_m_per_kw: float = GSHP_PROBE_M_PER_KW,
    capex_share_limit: float = DEFAULT_SUBSIDY_CAPEX_SHARE,
) -> tuple[float, float, float]:
    validate_project_type(project_type)

    capex_brut = gshp_capex_chf(
        p_nom_kw,
        fixed_chf=fixed_chf,
        hp_var_chf_per_kw=hp_var_chf_per_kw,
        probe_cost_chf_per_m=probe_cost_chf_per_m,
        probe_m_per_kw=probe_m_per_kw,
    )

    subsidy = estimate_geothermal_subsidy_chf(
        canton=canton,
        p_nom_kw=p_nom_kw,
        project_type=project_type,
        capex_chf=capex_brut,
        current_energy_label=current_energy_label,
        capex_share_limit=capex_share_limit,
    )

    capex_net = max(0.0, capex_brut - subsidy)
    return capex_brut, subsidy, capex_net


def conso_ancienne_install_kwh(besoin_chaleur_kwh: float, rendement: float) -> float:
    if besoin_chaleur_kwh <= 0:
        raise ValueError("besoin_chaleur_kwh doit être > 0.")
    if rendement <= 0:
        raise ValueError("rendement doit être > 0.")
    return besoin_chaleur_kwh / rendement


# ============================================================
#  Résultats structurés
# ============================================================

@dataclass(frozen=True)
class GSHPCostBreakdown:
    p_nom_kw: float
    chaleur_utile_kwh: float
    prix_elec_chf_kwh: float
    capex_chf: float
    annualized_capex_chf_per_year: float
    om_chf_per_year: float
    spf: float
    conso_elec_kwh_per_year: float
    cout_elec_chf_per_year: float
    cout_total_chf_per_year: float
    lcoh_chf_per_kwh: float


# ============================================================
#  Fonctions PAC géothermique
# ============================================================

def couts_annuels_gshp_zuberi(
    p_nom_kw: float,
    chaleur_utile_kwh: float,
    prix_elec_chf_kwh: float,
    *,
    spf: float = GSHP_SPF,
    r: float = DISCOUNT_RATE,
    lifetime_years: int = LIFETIME_YEARS,
    om_chf_per_year: float | None = None,
    om_fraction: float | None = None,
) -> GSHPCostBreakdown:
    if p_nom_kw <= 0:
        raise ValueError("p_nom_kw doit être > 0.")
    if chaleur_utile_kwh <= 0:
        raise ValueError("chaleur_utile_kwh doit être > 0.")
    if prix_elec_chf_kwh < 0:
        raise ValueError("prix_elec_chf_kwh doit être >= 0.")
    if spf <= 0:
        raise ValueError("spf doit être > 0.")
    if om_fraction is not None and om_fraction < 0:
        raise ValueError("om_fraction doit être >= 0 si fourni.")

    capex = gshp_capex_chf(p_nom_kw)
    a = annuity_factor(r=r, n_years=lifetime_years)
    annualized_capex = a * capex

    if om_chf_per_year is not None:
        om = float(om_chf_per_year)
    elif om_fraction is not None:
        om = float(om_fraction) * capex
    else:
        om = annual_om_cost_chf("PAC géothermique", p_nom_kw)

    conso_elec = chaleur_utile_kwh / spf
    cout_elec = conso_elec * prix_elec_chf_kwh
    total = annualized_capex + om + cout_elec
    lcoh = total / chaleur_utile_kwh

    return GSHPCostBreakdown(
        p_nom_kw=p_nom_kw,
        chaleur_utile_kwh=chaleur_utile_kwh,
        prix_elec_chf_kwh=prix_elec_chf_kwh,
        capex_chf=capex,
        annualized_capex_chf_per_year=annualized_capex,
        om_chf_per_year=om,
        spf=spf,
        conso_elec_kwh_per_year=conso_elec,
        cout_elec_chf_per_year=cout_elec,
        cout_total_chf_per_year=total,
        lcoh_chf_per_kwh=lcoh,
    )


def build_gshp_project_costs(
    p_nom_kw: float,
    chaleur_utile_kwh: float,
    prix_elec_chf_kwh: float,
    canton: str,
    *,
    project_type: ProjectType = "replacement",
    current_energy_label: str | None = None,
    spf: float = GSHP_SPF,
    r: float = DISCOUNT_RATE,
    lifetime_years: int = LIFETIME_YEARS,
    om_chf_per_year: float | None = None,
    om_fraction: float | None = None,
    capex_share_limit: float = DEFAULT_SUBSIDY_CAPEX_SHARE,
) -> dict:
    validate_project_type(project_type)

    gshp = couts_annuels_gshp_zuberi(
        p_nom_kw=p_nom_kw,
        chaleur_utile_kwh=chaleur_utile_kwh,
        prix_elec_chf_kwh=prix_elec_chf_kwh,
        spf=spf,
        r=r,
        lifetime_years=lifetime_years,
        om_chf_per_year=om_chf_per_year,
        om_fraction=om_fraction,
    )

    capex_brut, subvention, capex_net = gshp_capex_net_after_subsidy(
        p_nom_kw=p_nom_kw,
        canton=canton,
        project_type=project_type,
        current_energy_label=current_energy_label,
        capex_share_limit=capex_share_limit,
    )

    a = annuity_factor(r=r, n_years=lifetime_years)
    annualized_capex_net = a * capex_net
    cout_total_annuel_net = annualized_capex_net + gshp.om_chf_per_year + gshp.cout_elec_chf_per_year

    return {
        "project_type": project_type,
        "capex_brut_chf": capex_brut,
        "subvention_chf": subvention,
        "capex_net_chf": capex_net,
        "annualized_capex_brut_chf_per_year": gshp.annualized_capex_chf_per_year,
        "annualized_capex_net_chf_per_year": annualized_capex_net,
        "om_chf_per_year": gshp.om_chf_per_year,
        "spf": gshp.spf,
        "conso_elec_kwh_per_year": gshp.conso_elec_kwh_per_year,
        "cout_elec_chf_per_year": gshp.cout_elec_chf_per_year,
        "cout_total_annuel_brut_chf_per_year": gshp.cout_total_chf_per_year,
        "cout_total_annuel_net_chf_per_year": cout_total_annuel_net,
        "lcoh_brut_chf_per_kwh": gshp.lcoh_chf_per_kwh,
        "lcoh_net_chf_per_kwh": cout_total_annuel_net / chaleur_utile_kwh,
    }


# ============================================================
#  Coût du système actuel
# ============================================================

def cout_annuel_systeme_actuel(
    besoin_chaleur_kwh: float,
    rendement: float,
    prix_energie_chf_kwh: float,
    *,
    om_chf_per_year: float = 0.0,
) -> float:
    if prix_energie_chf_kwh < 0:
        raise ValueError("prix_energie_chf_kwh doit être >= 0.")

    conso = conso_ancienne_install_kwh(besoin_chaleur_kwh, rendement)
    return (conso * prix_energie_chf_kwh) + float(om_chf_per_year)


# ============================================================
#  Chargement des trajectoires de prix
# ============================================================

def load_price_path_electricity_ttc_by_canton(
    csv_path: str,
    canton: str,
    building_type: str,
    scenario: Scenario,
) -> pd.Series:
    df = pd.read_csv(csv_path)

    required_cols = {"canton", "building_type", "scenario", "year", "price_ttc_chf_kwh"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV électricité: {sorted(missing)}. "
            f"Colonnes disponibles: {list(df.columns)}"
        )

    df["canton"] = df["canton"].astype(str).str.strip()
    df["building_type"] = df["building_type"].astype(str).str.strip()
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["price_ttc_chf_kwh"] = pd.to_numeric(df["price_ttc_chf_kwh"], errors="coerce")

    df = df.dropna(subset=["year", "price_ttc_chf_kwh"]).copy()
    df["year"] = df["year"].astype(int)

    canton_norm = normalize_canton_name(canton)
    building_type_norm = normalize_building_type(building_type)
    scenario_norm = normalize_scenario_name(scenario)

    g = df[
        (df["canton"] == canton_norm) &
        (df["building_type"] == building_type_norm) &
        (df["scenario"] == scenario_norm)
    ].copy()

    if g.empty:
        raise ValueError(
            "Aucune trajectoire électricité trouvée après normalisation pour "
            f"canton={canton!r} -> {canton_norm!r}, "
            f"building_type={building_type!r} -> {building_type_norm!r}, "
            f"scenario={scenario!r} -> {scenario_norm!r}. "
            f"Cantons CSV disponibles: {sorted(df['canton'].unique().tolist())}. "
            f"Building types disponibles: {sorted(df['building_type'].unique().tolist())}. "
            f"Scenarios disponibles: {sorted(df['scenario'].unique().tolist())}."
        )

    g = g.sort_values("year")
    return pd.Series(
        g["price_ttc_chf_kwh"].astype(float).values,
        index=g["year"].values,
        name="price_elec",
    )


def load_price_path_fuel(
    csv_path: str,
    scenario: Scenario,
) -> pd.Series:
    df = pd.read_csv(csv_path)

    required_cols = {"scenario", "year", "price_chf_kwh"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV fuel: {sorted(missing)}. "
            f"Colonnes disponibles: {list(df.columns)}"
        )

    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["price_chf_kwh"] = pd.to_numeric(df["price_chf_kwh"], errors="coerce")

    df = df.dropna(subset=["year", "price_chf_kwh"]).copy()
    df["year"] = df["year"].astype(int)

    scenario_norm = normalize_scenario_name(scenario)

    g = df[df["scenario"] == scenario_norm].copy()
    if g.empty:
        raise ValueError(f"Aucune trajectoire fuel pour scenario={scenario_norm}")

    g = g.sort_values("year")
    return pd.Series(
        g["price_chf_kwh"].astype(float).values,
        index=g["year"].values,
        name="price_fuel",
    )


# ============================================================
#  Séries de coûts annuels
# ============================================================

def annual_cost_series_systeme_actuel(
    chaleur_utile_kwh: float,
    rendement: float,
    price_path_chf_kwh: pd.Series,
    *,
    om_chf_per_year: float = 0.0,
) -> pd.Series:
    conso_finale = conso_ancienne_install_kwh(chaleur_utile_kwh, rendement)
    return (conso_finale * price_path_chf_kwh) + float(om_chf_per_year)


def annual_cost_series_gshp(
    chaleur_utile_kwh: float,
    price_elec_path_chf_kwh: pd.Series,
    *,
    spf: float = GSHP_SPF,
    om_chf_per_year: float,
) -> pd.Series:
    if spf <= 0:
        raise ValueError("spf doit être > 0")
    conso_elec = chaleur_utile_kwh / spf
    return (conso_elec * price_elec_path_chf_kwh) + float(om_chf_per_year)


# ============================================================
#  Payback actualisé et VAN
# ============================================================

def payback_discounted_from_cashflows(
    capex_chf: float,
    cost_ref: pd.Series,
    cost_new: pd.Series,
    *,
    discount_rate: float = DISCOUNT_RATE,
) -> dict:
    years = sorted(set(cost_ref.index) & set(cost_new.index))
    if not years:
        raise ValueError("Aucune année commune entre cost_ref et cost_new")

    savings = pd.Series(
        [float(cost_ref.loc[y] - cost_new.loc[y]) for y in years],
        index=years,
        name="savings",
    )

    y0 = years[0]

    discount_factors = pd.Series(
        [1.0 / ((1.0 + discount_rate) ** (y - y0)) for y in years],
        index=years,
        name="discount_factor",
    )

    savings_discounted = savings * discount_factors
    cumulative = savings_discounted.cumsum()

    payback = None

    for y in years:
        if cumulative.loc[y] >= capex_chf:
            idx = years.index(y)

            if idx == 0:
                payback = 0.0
            else:
                y_prev = years[idx - 1]
                c_prev = cumulative.loc[y_prev]
                c_now = cumulative.loc[y]

                if c_now == c_prev:
                    payback = float(y - y0)
                else:
                    frac = (capex_chf - c_prev) / (c_now - c_prev)
                    payback = float((y_prev - y0) + frac)
            break

    npv = float(savings_discounted.sum() - capex_chf)

    return {
        "payback_years": payback,
        "npv": npv,
        "savings": savings,
        "savings_discounted": savings_discounted,
        "cumulative_discounted": cumulative,
        "start_year": y0,
        "end_year": years[-1],
    }