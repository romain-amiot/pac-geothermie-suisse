from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# COMPATIBILITÉ STREAMLIT CLOUD / LINUX
# ============================================================
#
# Sur Windows, les chemins ne sont pas sensibles à la casse :
# "Data/..." et "data/..." pointent vers le même dossier.
# Sur Streamlit Cloud, l'application tourne sous Linux, où la casse compte.
#
# Certains modules historiques du projet peuvent chercher les fichiers dans
# "data/...", alors que le dépôt GitHub contient le dossier "Data/".
# On crée donc au démarrage un alias "data" -> "Data" avant d'importer les
# modules internes qui chargent les CSV.
#
# Correction plus propre à long terme : harmoniser tous les chemins du projet
# pour utiliser exactement le même nom de dossier, idéalement "Data" partout.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"
LOWER_DATA_DIR = PROJECT_ROOT / "data"


def ensure_data_directory_alias() -> None:
    if not DATA_DIR.exists() or LOWER_DATA_DIR.exists():
        return

    try:
        # Fonctionne sur Streamlit Cloud / Linux.
        os.symlink(DATA_DIR, LOWER_DATA_DIR, target_is_directory=True)
    except Exception:
        # Sur Windows, la création de symlink peut demander des droits admin.
        # Ce n'est pas bloquant localement, car Windows ne distingue pas Data/data.
        pass


ensure_data_directory_alias()

from models import ProjectInputs
from services.calcul_projet import (
    evaluer_projet,
    estimate_sdep_vh_from_perimeter,
    estimate_sdep_vh_geometry,
    DEFAULT_GEOMETRY_BY_BUILDING_TYPE,
    K_SDEP_BY_BUILDING_TYPE,
)
from services.charts import build_cumulative_emissions_bar_chart
from Fonctions.puissance_pointe import UBAT_CHOICES
from Fonctions.localisation import postcode_info

st.set_page_config(
    page_title="Rentabilité PAC géothermique",
    layout="wide",
)


# ============================================================
# CONSTANTES INTERFACE
# ============================================================

DEFAULT_CEILING_HEIGHT_M = 2.5
DEFAULT_VENTILATION_R = 0.20
EER_CLIM_ACTUEL_DEFAULT = 3.0

INTERACTIVE_ADJUSTMENT_STEP = 10
INTERACTIVE_ADJUSTMENT_MIN = -30
INTERACTIVE_ADJUSTMENT_MAX = 30
DISCOUNT_RATE_DISPLAY = 0.03
ECONOMIC_HORIZON_DISPLAY_YEARS = 25

CANTONS_SWISS = [
    "Argovie",
    "Appenzell Rhodes-Extérieures",
    "Appenzell Rhodes-Intérieures",
    "Bâle-Campagne",
    "Bâle-Ville",
    "Berne",
    "Fribourg",
    "Genève",
    "Glaris",
    "Grisons",
    "Jura",
    "Lucerne",
    "Neuchâtel",
    "Nidwald",
    "Obwald",
    "Schaffhouse",
    "Schwyz",
    "Soleure",
    "Saint-Gall",
    "Thurgovie",
    "Tessin",
    "Uri",
    "Valais",
    "Vaud",
    "Zoug",
    "Zurich",
]

ENERGY_CHOICES = {
    "Gaz naturel": "Gaz naturel",
    "Mazout (fioul)": "Mazout (fioul)",
    "Électricité (réseau)": "Électricité (réseau)",
}

RENDEMENTS_STANDARDS: dict[str, float] = {
    "Électricité (réseau)": 1.00,
    "Gaz naturel": 0.90,
    "Mazout (fioul)": 0.90,
}

# Libellés grand public pour conserver les coefficients Ubat internes
# sans les afficher directement à l'utilisateur.
#
# On n'utilise pas directement les clés de UBAT_CHOICES dans l'interface,
# car elles peuvent être des chiffres ou des libellés techniques.
# L'utilisateur voit uniquement ces catégories simplifiées, et le modèle
# reçoit seulement le coefficient Ubat correspondant.
DATE_CONSTRUCTION_UBAT = {
    "Construction après 2012 avec isolation exceptionnelle": 0.30,
    "Construction après 2012 sans ponts thermiques": 0.40,
    "Construction 2001-2012": 0.75,
    "Construction 1990-2000": 0.95,
    "Construction 1983-1989": 1.15,
    "Construction 1974-1982": 1.40,
    "Construction avant 1974": 1.80,
}

DATE_CONSTRUCTION_CHOICES = list(DATE_CONSTRUCTION_UBAT.keys())

TYPOLOGIES_MENU = [
    ("maison_individuelle", "Maison individuelle"),
    ("residentiel_collectif", "Immeuble résidentiel collectif"),
    ("grand_batiment_compact", "Grand bâtiment compact"),
    ("mixte", "Bâtiment mixte"),
    ("activites", "Bâtiment d’activités / tertiaire"),
    ("equipement_collectif", "Équipement collectif / bâtiment public"),
]


MITOYENNETE_LABELS = {
    "isole": "Isolé",
    "1_cote": "1 côté mitoyen",
    "2_cotes": "2 côtés mitoyens",
    "3_cotes": "3 côtés mitoyens",
}

MITOYENNETE_SDEP_FACTOR = {
    "isole": 1.00,
    "1_cote": 0.75,
    "2_cotes": 0.50,
    "3_cotes": 0.25,
}

def estimate_sdep_vh_typology_with_exposure(
    *,
    shab_m2: float,
    niveaux: int,
    hauteur_m: float,
    building_type_key: str,
    mitoyennete: str,
    toiture_exposee: bool,
    plancher_expose: bool,
) -> tuple[float, float, dict]:
    """
    Estimation typologique corrigée.

    La forme générale n'est pas demandée : elle reste intégrée dans le
    coefficient typologique K du type de bâtiment.

    Pour que le nombre d'étages et la hauteur sous plafond aient bien un effet,
    on décompose la Sdép typologique en :
    - une partie verticale, corrigée par le ratio entre la hauteur saisie et
      une hauteur de référence propre à la typologie ;
    - une partie horizontale, égale à l'empreinte au sol calculée directement
      avec le nombre de niveaux renseigné par l'utilisateur.

    La mitoyenneté est ensuite appliquée uniquement sur la partie verticale,
    car elle concerne les façades et non la toiture ou le plancher.
    """
    if shab_m2 <= 0:
        raise ValueError("La surface chauffée doit être positive.")
    if niveaux <= 0:
        raise ValueError("Le nombre de niveaux doit être positif.")
    if hauteur_m <= 0:
        raise ValueError("La hauteur moyenne doit être positive.")

    if building_type_key not in K_SDEP_BY_BUILDING_TYPE:
        raise ValueError(f"Typologie inconnue : {building_type_key!r}")

    if mitoyennete not in MITOYENNETE_SDEP_FACTOR:
        raise ValueError(f"Mitoyenneté inconnue : {mitoyennete!r}")

    k_typologique = float(K_SDEP_BY_BUILDING_TYPE[building_type_key])

    # Hauteur de référence uniquement pour corriger la partie verticale.
    # On revient ici à la méthode par ratio, car K_typologique correspond déjà
    # à une enveloppe moyenne calibrée avec une hauteur implicite de référence.
    hauteur_ref = float(
        DEFAULT_GEOMETRY_BY_BUILDING_TYPE.get(building_type_key, {}).get(
            "hauteur_m",
            2.7,
        )
    )

    # Sdép typologique issue de K.
    sdep_typologique_ref = k_typologique * float(shab_m2)

    # Empreinte au sol directement issue du nombre de niveaux renseigné par
    # l'utilisateur. On n'utilise pas de nombre de niveaux de référence.
    empreinte_sol = float(shab_m2) / int(niveaux)

    # On considère que la Sdép typologique contient une part horizontale
    # correspondant à la toiture et au plancher. Cette part est calculée avec
    # l'empreinte réelle du bâtiment, donc avec le nombre de niveaux saisi.
    s_horizontal_ref = 2.0 * empreinte_sol
    s_vertical_ref = max(0.0, sdep_typologique_ref - s_horizontal_ref)

    # La hauteur sous plafond corrige uniquement la partie verticale.
    facteur_hauteur = float(hauteur_m) / hauteur_ref if hauteur_ref > 0 else 1.0
    s_vertical_corrigee = s_vertical_ref * facteur_hauteur

    # La mitoyenneté ne concerne que les façades, donc uniquement la partie
    # verticale. Elle ne doit pas réduire la toiture ni le plancher.
    facteur_mitoyennete = float(MITOYENNETE_SDEP_FACTOR[mitoyennete])
    s_vertical_apres_mitoyennete = s_vertical_corrigee * facteur_mitoyennete

    # La surface horizontale dépend du nombre d'étages et de l'exposition.
    s_toiture = empreinte_sol if toiture_exposee else 0.0
    s_plancher = empreinte_sol if plancher_expose else 0.0

    sdep_corrigee = s_vertical_apres_mitoyennete + s_toiture + s_plancher
    sdep_corrigee = max(1.0, float(sdep_corrigee))

    vh_est = float(shab_m2) * float(hauteur_m)

    meta = {
        "k_typologique": k_typologique,
        "sdep_typologique_ref_m2": float(sdep_typologique_ref),
        "hauteur_ref_m": float(hauteur_ref),
        "hauteur_utilisateur_m": float(hauteur_m),
        "facteur_hauteur": float(facteur_hauteur),
        "niveaux_utilisateur": int(niveaux),
        "empreinte_sol_m2": float(empreinte_sol),
        "s_horizontal_ref_m2": float(s_horizontal_ref),
        "s_vertical_ref_m2": float(s_vertical_ref),
        "s_vertical_corrigee_m2": float(s_vertical_corrigee),
        "facteur_mitoyennete": facteur_mitoyennete,
        "s_vertical_apres_mitoyennete_m2": float(s_vertical_apres_mitoyennete),
        "s_toiture_m2": float(s_toiture),
        "s_plancher_m2": float(s_plancher),
        "sdep_corrigee_m2": float(sdep_corrigee),
        "vh_m3": float(vh_est),
    }

    return float(sdep_corrigee), float(vh_est), meta


# ============================================================
# OUTILS AFFICHAGE
# ============================================================

def format_chf(value: float | int | None, decimals: int = 0) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{decimals}f} CHF".replace(",", " ")
    except Exception:
        return "—"


def format_percent(value: float | int | None, decimals: int = 0) -> str:
    if value is None:
        return "—"
    try:
        return f"{100.0 * float(value):.{decimals}f} %"
    except Exception:
        return "—"


def format_payback(value: float | int | None) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.1f} ans"
    except Exception:
        return "—"


def format_table_float(value: float | int | None, decimals: int = 1) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except Exception:
        return None


# ============================================================
# OUTILS INTERACTIFS CUMULATIFS
# ============================================================

def default_interactive_adjustments() -> dict[str, int]:
    return {
        "current_energy_pct": 0,
        "electricity_price_pct": 0,
        "capex_pct": 0,
    }


def clamp_adjustment(
    value: int,
    min_value: int = INTERACTIVE_ADJUSTMENT_MIN,
    max_value: int = INTERACTIVE_ADJUSTMENT_MAX,
) -> int:
    return max(min_value, min(max_value, int(value)))


def adjustment_factor(pct: int | float) -> float:
    return 1.0 + float(pct) / 100.0


def update_interactive_adjustment(key: str, delta: int) -> None:
    if "interactive_adjustments" not in st.session_state:
        st.session_state["interactive_adjustments"] = default_interactive_adjustments()

    current = st.session_state["interactive_adjustments"].get(key, 0)
    st.session_state["interactive_adjustments"][key] = clamp_adjustment(current + delta)


def reset_interactive_adjustments() -> None:
    st.session_state["interactive_adjustments"] = default_interactive_adjustments()


def render_interactive_adjustment_controls(results: dict) -> dict[str, int]:
    inputs = results["inputs"]
    current_energy_label = inputs.current_energy or "énergie actuelle"

    if "interactive_adjustments" not in st.session_state:
        st.session_state["interactive_adjustments"] = default_interactive_adjustments()

    adj = st.session_state["interactive_adjustments"]

    # Compatibilité avec d'anciens résultats de session qui utilisaient "pac_annual_pct".
    if "electricity_price_pct" not in adj:
        adj["electricity_price_pct"] = int(adj.get("pac_annual_pct", 0))
    adj.pop("pac_annual_pct", None)

    st.markdown("#### Tester rapidement des variations cumulées")
    st.caption(
        "Chaque clic modifie le paramètre de 10 points de pourcentage. "
        "Chaque paramètre est borné entre -30 % et +30 %. Les variations se cumulent."
    )

    is_current_system_electric = current_energy_label in {
        "Électricité (réseau)",
        "Electricité (réseau)",
        "Electricite (réseau)",
        "Electricite",
        "Électricité",
    }

    c1, c2, c3 = st.columns(3)

    with c1:
        if is_current_system_electric:
            # Pas de doublon : le prix de l'électricité est déjà piloté par le bouton dédié.
            adj["current_energy_pct"] = 0

            st.markdown("**Prix énergie actuelle**")
            st.caption(
                "Non affiché : le système actuel utilise déjà l'électricité. "
                "La variation est gérée par le bouton Prix électricité."
            )
        else:
            st.markdown(f"**Prix {current_energy_label}**")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("-10 %", key="btn_current_energy_minus_10"):
                    update_interactive_adjustment("current_energy_pct", -INTERACTIVE_ADJUSTMENT_STEP)
                    st.rerun()
            with b2:
                if st.button("+10 %", key="btn_current_energy_plus_10"):
                    update_interactive_adjustment("current_energy_pct", INTERACTIVE_ADJUSTMENT_STEP)
                    st.rerun()
            st.caption(f"Variation actuelle : {adj.get('current_energy_pct', 0):+d} %")

    with c2:
        st.markdown("**Prix électricité**")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("-10 %", key="btn_electricity_price_minus_10"):
                update_interactive_adjustment("electricity_price_pct", -INTERACTIVE_ADJUSTMENT_STEP)
                st.rerun()
        with b2:
            if st.button("+10 %", key="btn_electricity_price_plus_10"):
                update_interactive_adjustment("electricity_price_pct", INTERACTIVE_ADJUSTMENT_STEP)
                st.rerun()
        st.caption(f"Variation actuelle : {adj.get('electricity_price_pct', 0):+d} %")

    with c3:
        st.markdown("**CAPEX net**")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("-10 %", key="btn_capex_minus_10"):
                update_interactive_adjustment("capex_pct", -INTERACTIVE_ADJUSTMENT_STEP)
                st.rerun()
        with b2:
            if st.button("+10 %", key="btn_capex_plus_10"):
                update_interactive_adjustment("capex_pct", INTERACTIVE_ADJUSTMENT_STEP)
                st.rerun()
        st.caption(f"Variation actuelle : {adj.get('capex_pct', 0):+d} %")

    if st.button("Réinitialiser les variations", key="btn_reset_interactive_adjustments"):
        reset_interactive_adjustments()
        st.rerun()

    adj = st.session_state["interactive_adjustments"]

    current_energy_part = ""
    if not is_current_system_electric:
        current_energy_part = f"{current_energy_label} {adj.get('current_energy_pct', 0):+d} %, "

    st.info(
        f"Hypothèse affichée : "
        f"{current_energy_part}"
        f"électricité {adj.get('electricity_price_pct', 0):+d} %, "
        f"CAPEX net {adj.get('capex_pct', 0):+d} %."
    )

    return adj


def discounted_cumulative_series(
    annual_costs: pd.Series,
    discount_rate: float = DISCOUNT_RATE_DISPLAY,
    initial_cost: float = 0.0,
) -> pd.Series:
    values = []
    cumulative = float(initial_cost)

    for i, value in enumerate(annual_costs, start=1):
        cumulative += float(value) / ((1.0 + discount_rate) ** i)
        values.append(cumulative)

    return pd.Series(values, index=annual_costs.index)


def first_n_years_series(series: pd.Series, n_years: int) -> pd.Series:
    """Garde les n premières années d'une série de coûts.

    Le cas déterministe du modèle est évalué sur l'horizon économique de 25 ans.
    Le graphique interactif doit donc utiliser le même horizon pour que la VAN
    affichée avec les boutons à 0 % corresponde à la VAN estimée plus haut.
    """
    s = pd.Series(series).copy()
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([float("inf"), float("-inf")], pd.NA)
    s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
    return s.astype(float).iloc[: int(n_years)].copy()


def economic_horizon_years_from_results(central: dict) -> int:
    """Retourne l'horizon économique utilisé pour la VAN.

    On lit l'horizon dans le bloc d'incertitude quand il est disponible.
    Sinon, on retombe sur la valeur du mémoire : 25 ans.
    """
    try:
        unc = central.get("uncertainty", {})
        horizon = int(unc.get("horizon_years", ECONOMIC_HORIZON_DISPLAY_YEARS))
        if horizon > 0:
            return horizon
    except Exception:
        pass
    return ECONOMIC_HORIZON_DISPLAY_YEARS


def local_discounted_payback(
    *,
    capex_net: float,
    cost_ref: pd.Series,
    cost_pac: pd.Series,
    discount_rate: float = DISCOUNT_RATE_DISPLAY,
) -> float | None:
    cumulative = -float(capex_net)

    for i, saving in enumerate(cost_ref - cost_pac, start=1):
        discounted_saving = float(saving) / ((1.0 + discount_rate) ** i)
        previous = cumulative
        cumulative += discounted_saving

        if cumulative >= 0:
            if discounted_saving <= 0:
                return float(i)

            fraction = abs(previous) / discounted_saving
            return float((i - 1) + fraction)

    return None


def render_interactive_cumulative_cost_chart(
    *,
    central: dict,
    pac: dict,
    adjustments: dict,
    discount_rate: float = DISCOUNT_RATE_DISPLAY,
) -> dict:
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        st.error("Plotly n'est pas installé. Lance : py -m pip install plotly")
        return {"payback": None, "van_recalculee": None, "capex_net": None}

    current_energy_factor = adjustment_factor(adjustments.get("current_energy_pct", 0))
    electricity_price_factor = adjustment_factor(adjustments.get("electricity_price_pct", 0))
    capex_factor = adjustment_factor(adjustments.get("capex_pct", 0))

    # ========================================================
    # Système actuel
    # ========================================================
    # Version rigoureuse : on ne multiplie par le facteur "énergie actuelle"
    # que la partie énergie de chauffage, pas la maintenance.
    # La partie froid existante éventuelle est électrique, donc elle suit le
    # facteur du prix de l'électricité.
    if (
        central.get("cost_actuel_heat_energy_series") is not None
        and central.get("cost_actuel_cool_series") is not None
        and central.get("om_actuel_series") is not None
    ):
        cost_actuel_heat = pd.Series(central["cost_actuel_heat_energy_series"]).astype(float)
        cost_actuel_cool = pd.Series(central["cost_actuel_cool_series"]).astype(float)
        om_actuel = pd.Series(central["om_actuel_series"]).astype(float)

        cost_ref = (
            cost_actuel_heat * current_energy_factor
            + cost_actuel_cool * electricity_price_factor
            + om_actuel
        )
    else:
        # Fallback ancien format : on ne dispose pas du détail énergie/maintenance.
        cost_ref_base = pd.Series(central["cost_actuel_total_series"]).astype(float)
        cost_ref = cost_ref_base * current_energy_factor

    # ========================================================
    # PAC géothermique
    # ========================================================
    # Version rigoureuse : le bouton "Prix électricité" modifie uniquement
    # l'électricité consommée par la PAC pour le chauffage et le froid.
    # La maintenance PAC reste inchangée.
    if (
        central.get("cost_pac_heat_energy_series") is not None
        and central.get("cost_pac_cool_series") is not None
        and central.get("om_pac_series") is not None
    ):
        cost_pac_heat = pd.Series(central["cost_pac_heat_energy_series"]).astype(float)
        cost_pac_cool = pd.Series(central["cost_pac_cool_series"]).astype(float)
        om_pac = pd.Series(central["om_pac_series"]).astype(float)

        cost_pac = (
            cost_pac_heat * electricity_price_factor
            + cost_pac_cool * electricity_price_factor
            + om_pac
        )
    else:
        # Fallback ancien format : moins rigoureux, mais évite de casser l'interface
        # si calcul_projet.py ne retourne pas encore les séries séparées.
        cost_pac_base = pd.Series(central["cost_pac_total_series"]).astype(float)
        cost_pac = cost_pac_base * electricity_price_factor

    horizon_years = economic_horizon_years_from_results(central)
    cost_ref = first_n_years_series(cost_ref, horizon_years)
    cost_pac = first_n_years_series(cost_pac, horizon_years)

    capex_net = float(pac["capex_net"]) * capex_factor

    cum_ref = discounted_cumulative_series(
        cost_ref,
        discount_rate=discount_rate,
        initial_cost=0.0,
    )

    cum_pac = discounted_cumulative_series(
        cost_pac,
        discount_rate=discount_rate,
        initial_cost=capex_net,
    )

    payback = local_discounted_payback(
        capex_net=capex_net,
        cost_ref=cost_ref,
        cost_pac=cost_pac,
        discount_rate=discount_rate,
    )

    total_ref = float(cum_ref.iloc[-1])
    total_pac = float(cum_pac.iloc[-1])
    van_recalculee = total_ref - total_pac

    # Si tous les boutons sont neutres, la VAN recalculée doit être exactement
    # la même que la VAN du cas déterministe affichée plus haut. On réutilise
    # donc la valeur centrale déjà calculée par calcul_projet.py pour éviter
    # tout écart d'arrondi ou d'horizon.
    if (
        int(adjustments.get("current_energy_pct", 0)) == 0
        and int(adjustments.get("electricity_price_pct", 0)) == 0
        and int(adjustments.get("capex_pct", 0)) == 0
        and central.get("npv") is not None
    ):
        try:
            van_recalculee = float(central["npv"])
        except Exception:
            pass

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(cum_ref.index),
            y=list(cum_ref.values),
            mode="lines+markers",
            name="Système actuel",
            hovertemplate="Année %{x}<br>Coût cumulé %{y:,.0f} CHF<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(cum_pac.index),
            y=list(cum_pac.values),
            mode="lines+markers",
            name="PAC géothermique",
            hovertemplate="Année %{x}<br>Coût cumulé %{y:,.0f} CHF<extra></extra>",
        )
    )

    if payback is not None:
        try:
            first_year = float(cum_ref.index[0])

            # Si la première valeur de l'axe est une année calendrier,
            # on convertit le temps de retour en année affichée.
            # Exemple : première année 2026, retour 22.9 ans -> 2047.9.
            if first_year > 1900:
                payback_x = first_year + float(payback) - 1.0
            else:
                payback_x = float(payback)

            x_min = min(float(cum_ref.index[0]), float(cum_pac.index[0]))
            x_max = max(float(cum_ref.index[-1]), float(cum_pac.index[-1]))

            if x_min <= payback_x <= x_max:
                fig.add_vline(
                    x=payback_x,
                    line_dash="dash",
                    annotation_text=f"Retour ≈ {payback:.1f} ans",
                    annotation_position="top",
                )

        except Exception:
            pass

    fig.update_layout(
        title="Coûts cumulés actualisés — hypothèse interactive",
        xaxis_title="Année",
        yaxis_title="Coût cumulé actualisé (CHF)",
        hovermode="x unified",
        legend_title="Système",
        margin=dict(l=30, r=30, t=60, b=30),
    )

    try:
        fig.update_xaxes(
            range=[
                float(cum_ref.index[0]),
                float(cum_ref.index[-1]),
            ]
        )
    except Exception:
        pass

    st.plotly_chart(fig, use_container_width=True)

    return {
        "payback": payback,
        "van_recalculee": van_recalculee,
        "capex_net": capex_net,
    }


# ============================================================
# GRAPHIQUES MONTE CARLO
# ============================================================

def render_payback_simulation_chart(
    *,
    unc: dict,
    deterministic_payback: float | None,
) -> None:
    samples = unc.get("simulation_samples", {})
    paybacks = samples.get("paybacks_extended", [])

    if not paybacks:
        st.info("Aucune simulation de temps de retour disponible pour le graphique.")
        return

    paybacks = [float(x) for x in paybacks if x is not None]

    if not paybacks:
        st.info("Aucune simulation amortie disponible pour le graphique.")
        return

    pb20 = unc.get("payback_p20")
    pb50 = unc.get("payback_p50")
    pb80 = unc.get("payback_p80")

    prob_25 = unc.get("amortization_probability_life", unc.get("amortization_probability"))
    prob_50 = unc.get("amortization_probability_extended")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bins = min(50, max(20, int(len(paybacks) ** 0.5)))
    ax.hist(paybacks, bins=bins, edgecolor="black", alpha=0.75)

    if deterministic_payback is not None:
        ax.axvline(
            float(deterministic_payback),
            linestyle="--",
            linewidth=2,
            label=f"Cas général simplifié : {float(deterministic_payback):.1f} ans",
        )

    if pb50 is not None:
        ax.axvline(
            float(pb50),
            linestyle="-",
            linewidth=2,
            label=f"Temps de retour médian : {float(pb50):.1f} ans",
        )

    if pb20 is not None and pb80 is not None:
        ax.axvspan(
            float(pb20),
            float(pb80),
            alpha=0.18,
            label=f"Intervalle central P20–P80 : {float(pb20):.1f}–{float(pb80):.1f} ans",
        )

    ax.axvline(
        25,
        linestyle=":",
        linewidth=2,
        label="Horizon économique 25 ans",
    )

    ax.set_xlabel("Temps de retour actualisé simulé (années)")
    ax.set_ylabel("Nombre de simulations amorties")
    ax.set_title("Distribution des temps de retour simulés")
    ax.grid(True, alpha=0.25)
    ax.legend()

    st.pyplot(fig)



def render_npv_simulation_chart(
    *,
    unc: dict,
    deterministic_npv: float | None,
) -> None:
    samples = unc.get("simulation_samples", {})
    npvs = samples.get("npvs_life", [])

    if not npvs:
        st.info("Aucune simulation de VAN disponible pour le graphique.")
        return

    npvs = [float(x) for x in npvs if x is not None]

    if not npvs:
        st.info("Aucune simulation de VAN exploitable.")
        return

    npv_p10 = unc.get("npv_p10")
    npv_p50 = unc.get("npv_p50")
    npv_p90 = unc.get("npv_p90")

    fig, ax = plt.subplots(figsize=(9, 4.8))

    ax.hist(npvs, bins=35, edgecolor="black", alpha=0.75)

    ax.axvline(
        0,
        linestyle=":",
        linewidth=2,
        label="Seuil de rentabilité VAN = 0",
    )

    if deterministic_npv is not None:
        ax.axvline(
            float(deterministic_npv),
            linestyle="--",
            linewidth=2,
            label=f"Cas général simplifié : {format_chf(deterministic_npv, decimals=0)}",
        )

    if npv_p50 is not None:
        ax.axvline(
            float(npv_p50),
            linestyle="-",
            linewidth=2,
            label=f"VAN médiane : {format_chf(npv_p50, decimals=0)}",
        )

    if npv_p10 is not None and npv_p90 is not None:
        ax.axvspan(
            float(npv_p10),
            float(npv_p90),
            alpha=0.18,
            label="Intervalle P10–P90",
        )

    ax.set_xlabel("VAN simulée sur 25 ans (CHF)")
    ax.set_ylabel("Nombre de simulations")
    ax.set_title("Distribution de la VAN simulée")
    ax.grid(True, alpha=0.25)
    ax.legend()

    st.pyplot(fig)


# ============================================================
# BLOCS D'AFFICHAGE DES RÉSULTATS
# ============================================================

def render_uncertainty_block(label: str, data: dict) -> None:
    unc = data.get("uncertainty", {})

    if not isinstance(unc, dict) or not unc.get("available"):
        reason = unc.get("reason", "non_disponible") if isinstance(unc, dict) else "non_disponible"
        st.info(f"Analyse d'incertitude non disponible pour ce scénario : {reason}.")
        return

    st.markdown("### Résultat de référence probabiliste — Monte Carlo")
    st.caption(
        "Les valeurs principales ci-dessous sont issues de la distribution Monte Carlo. "
        "La médiane est utilisée comme valeur de référence, car elle intègre les incertitudes "
        "sur les coûts, les besoins, les performances et les chocs de prix des énergies fossiles."
    )

    pb20 = unc.get("payback_p20")
    pb50 = unc.get("payback_p50")
    pb80 = unc.get("payback_p80")
    representative = bool(unc.get("payback_median_representative", False))

    prob_25 = unc.get("amortization_probability_life", unc.get("amortization_probability"))
    prob_50 = unc.get("amortization_probability_extended")

    c1, c2, c3 = st.columns(3)

    if pb50 is not None:
        if representative:
            c1.metric("Temps de retour de référence", format_payback(pb50))
        else:
            c1.metric("Temps de retour de référence conditionnel", format_payback(pb50))
    else:
        c1.metric("Temps de retour de référence", "—")

    if pb20 is not None and pb80 is not None:
        c2.metric("Intervalle central P20–P80", f"{float(pb20):.1f} – {float(pb80):.1f} ans")
    else:
        c2.metric("Intervalle central P20–P80", "—")

    c3.metric("Probabilité amorti en 25 ans", format_percent(prob_25, decimals=0))

    c4, c5, c6 = st.columns(3)

    c4.metric("Probabilité amorti en 50 ans", format_percent(prob_50, decimals=0))
    c5.metric("VAN de référence", format_chf(unc.get("npv_p50"), decimals=0))

    if unc.get("npv_p10") is not None and unc.get("npv_p90") is not None:
        c6.metric(
            "VAN P10–P90",
            f"{format_chf(unc.get('npv_p10'), decimals=0)} – {format_chf(unc.get('npv_p90'), decimals=0)}",
        )
    else:
        c6.metric("VAN P10–P90", "—")

    c7, c8, c9 = st.columns(3)
    c7.metric("Économies année 1 médianes", format_chf(unc.get("annual_saving_year0_p50"), decimals=0))
    c8.metric("CAPEX net médian simulé", format_chf(unc.get("capex_net_p50"), decimals=0))

    n_life = unc.get("n_amortized_life", unc.get("n_amortized", 0))
    n_extended = unc.get("n_amortized_extended", None)
    n_sims = unc.get("n_sims", 0)

    if n_extended is not None:
        c9.metric("Simulations amorties", f"{n_life} / {n_sims} à 25 ans")
        st.caption(f"Simulations amorties sur l'horizon étendu : {n_extended} / {n_sims}.")
    else:
        c9.metric("Simulations amorties", f"{n_life} / {n_sims}")

    if pb50 is not None and not representative:
        st.warning(
            "Moins de 50 % des simulations sont amorties sur l'horizon étendu. "
            "Le temps de retour médian affiché est donc conditionnel aux simulations amorties. "
            "Dans ce cas, l'indicateur principal à regarder est la probabilité d'amortissement."
        )

    st.markdown("### Distribution des simulations")
    st.caption(
        "La ligne pleine indique la médiane Monte Carlo, utilisée comme référence. "
        "La ligne pointillée indique le résultat déterministe sans chocs fossiles, conservé uniquement comme comparaison."
    )
    render_payback_simulation_chart(
        unc=unc,
        deterministic_payback=data.get("payback"),
    )
    render_npv_simulation_chart(
        unc=unc,
        deterministic_npv=data.get("npv"),
    )

    confidence = unc.get("confidence", {})
    if isinstance(confidence, dict) and confidence:
        level = confidence.get("level")
        score = confidence.get("score")
        reasons_positive = confidence.get("reasons_positive", [])
        reasons_negative = confidence.get("reasons_negative", [])

        with st.expander("Niveau de confiance du résultat"):
            st.write(f"**Niveau :** {level} — score {score}")

            if reasons_positive:
                st.write("Points favorables :")
                for reason in reasons_positive:
                    st.write(f"- {reason}")

            if reasons_negative:
                st.write("Points de prudence :")
                for reason in reasons_negative:
                    st.write(f"- {reason}")

    with st.expander("Détail de l'analyse d'incertitude"):
        st.write({
            "scénario": label,
            "méthode": unc.get("method"),
            "nombre_simulations": unc.get("n_sims"),
            "horizon_economique_ans": unc.get("horizon_years"),
            "horizon_payback_etendu_ans": unc.get("payback_max_years"),
            "taux_actualisation": unc.get("discount_rate"),
            "temps_retour_p20_conditionnel": unc.get("payback_p20"),
            "temps_retour_p50_conditionnel": unc.get("payback_p50"),
            "temps_retour_p80_conditionnel": unc.get("payback_p80"),
            "temps_retour_median_representatif": unc.get("payback_median_representative"),
            "probabilite_amortissement_25_ans": unc.get("amortization_probability_life"),
            "probabilite_amortissement_50_ans": unc.get("amortization_probability_extended"),
            "van_p10": unc.get("npv_p10"),
            "van_p50": unc.get("npv_p50"),
            "van_p90": unc.get("npv_p90"),
            "hypotheses": unc.get("assumptions"),
        })


def build_parameter_sensitivity_display(rows: list[dict]) -> pd.DataFrame:
    display_rows = []

    for row in rows:
        display_rows.append({
            "Paramètre": row.get("parametre"),
            "Variation basse": row.get("variation_basse"),
            "Retour bas": format_table_float(row.get("payback_bas"), 1),
            "VAN basse": format_table_float(row.get("npv_bas"), 0),
            "Central": "central",
            "Retour central": format_table_float(row.get("payback_central"), 1),
            "VAN centrale": format_table_float(row.get("npv_central"), 0),
            "Variation haute": row.get("variation_haute"),
            "Retour haut": format_table_float(row.get("payback_haut"), 1),
            "VAN haute": format_table_float(row.get("npv_haut"), 0),
            "Impact VAN max": format_table_float(row.get("impact_npv_abs"), 0),
        })

    return pd.DataFrame(display_rows)


def build_price_sensitivity_display(price_sensitivity: dict) -> pd.DataFrame:
    rows = []

    for label, data in price_sensitivity.items():
        rows.append({
            "Scénario prix": label,
            "Description": data.get("description"),
            "Électricité": data.get("electricity_scenario"),
            "Énergie actuelle": data.get("fuel_scenario"),
            "Coût actuel année 1": format_table_float(data.get("cout_actuel_total_year0"), 0),
            "Coût PAC année 1": format_table_float(data.get("cout_pac_total_year0"), 0),
            "Économies année 1": format_table_float(data.get("economies_annuelles_year0"), 0),
            "Temps retour": format_table_float(data.get("payback"), 1),
            "VAN": format_table_float(data.get("npv"), 0),
        })

    return pd.DataFrame(rows)


# ============================================================
# FORMULAIRE
# ============================================================

def build_inputs_from_form() -> ProjectInputs:
    st.sidebar.header("Paramètres du projet")

    project_type_label = st.sidebar.radio(
        "Type de projet",
        options=["Remplacement ancien chauffage", "Nouveau bâtiment"],
    )
    project_type = "replacement" if project_type_label == "Remplacement ancien chauffage" else "new_building"

    if project_type == "replacement":
        st.write(
            "Cet outil permet de déterminer la rentabilité de l'installation d'une pompe "
            "à chaleur géothermique par rapport à votre système de chauffage actuel. "
            "Il prend en compte au mieux les caractéristiques de votre bâtiment et propose "
            "une aide à la décision, mais il ne remplace pas un devis détaillé réalisé par "
            "des professionnels."
        )
    else:
        st.write(
            "Cette section permet de déterminer la rentabilité de l'installation d'une pompe "
            "à chaleur géothermique pour un nouveau bâtiment. Il propose une aide à la décision "
            "rapide, mais ne remplace pas un devis détaillé réalisé par des professionnels."
        )

    postcode_valid = False
    postcode_meta = None
    canton = "Vaud"  # fallback interne, mais le calcul sera bloqué si postcode_valid = False

    # ========================================================
    # BÂTIMENT
    # ========================================================

    st.header("1. Caractéristiques du bâtiment")

    # Le code postal est placé dans le corps principal de l'application,
    # car c'est une information centrale pour l'utilisateur : il détermine
    # automatiquement le canton et la station climatique utilisée.
    postcode = st.text_input("Code postal du bâtiment", value="1000")

    if postcode.strip():
        try:
            postcode_meta = postcode_info(postcode)
            canton = postcode_meta["canton_name"]
            postcode_valid = True

            st.success(
                f"Localisation reconnue : {postcode_meta['locality']} — "
                f"{postcode_meta['canton_abbr']} "
                f"({postcode_meta['canton_name']})"
            )

        except ValueError as e:
            st.error(str(e))
            postcode_valid = False

        except Exception as e:
            st.error(
                "Erreur lors de la lecture de la base des codes postaux. "
                f"Détail : {e}"
            )
            postcode_valid = False
    else:
        st.error("Code postal invalide. Veuillez vérifier votre saisie.")
        postcode_valid = False

    st.session_state["postcode_valid"] = postcode_valid
    st.session_state["postcode_meta"] = postcode_meta

    building_type_label = st.selectbox(
        "Typologie du bâtiment",
        options=[label for _, label in TYPOLOGIES_MENU],
    )
    building_type_key = next(key for key, label in TYPOLOGIES_MENU if label == building_type_label)

    date_construction_label = st.selectbox(
        "Date de construction",
        options=DATE_CONSTRUCTION_CHOICES,
        index=2,
    )
    st.caption(
        "Si l'isolation du bâtiment a été rénovée, prenez la date de ces travaux "
        "plutôt que la date de construction initiale."
    )
    ubat = float(DATE_CONSTRUCTION_UBAT[date_construction_label])

    # Le mémoire retient une valeur standard fixe R = 0.20 W/m³K.
    # On supprime donc le choix utilisateur de la ventilation et son affichage.
    ventilation_r = DEFAULT_VENTILATION_R

    # ========================================================
    # GÉOMÉTRIE
    # ========================================================

    st.header("2. Géométrie et données thermiques")
    st.write(
        "Cette section permet de déterminer la surface de déperdition du bâtiment "
        "ainsi que le volume habitable. La surface de déperdition représente "
        "l'ensemble des parois en contact avec l'extérieur. Plusieurs méthodes "
        "de saisie sont disponibles, en fonction de vos connaissances des "
        "paramètres du bâtiment."
    )

    sdep_mode = st.radio(
        "Méthode de saisie de la géométrie du bâtiment",
        options=["Estimation par typologie", "Estimation par périmètre", "Saisie directe"],
        horizontal=True,
    )

    sdep_m2 = None
    vh_m3 = None
    shab_m2 = None
    niveaux = None
    perimetre_m = None
    hauteur_m = DEFAULT_CEILING_HEIGHT_M
    toiture_exposee = True
    plancher_expose = True
    forme_generale = None
    mitoyennete = None
    longueur_m = None
    largeur_m = None

    if sdep_mode == "Estimation par typologie":
        defaults = DEFAULT_GEOMETRY_BY_BUILDING_TYPE[building_type_key]

        st.caption(
            "Méthode la plus générale, à privilégier si vous ne connaissez pas "
            "les paramètres techniques du bâtiment."
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            shab_m2 = st.number_input(
                "Surface habitable (m²)",
                min_value=1.0,
                value=180.0,
                step=10.0,
            )

        with c2:
            niveaux = st.number_input(
                "Niveaux chauffés",
                min_value=1,
                max_value=20,
                value=2,
                step=1,
            )

        with c3:
            hauteur_m = st.number_input(
                "Hauteur sous plafond (m)",
                min_value=1.8,
                value=float(defaults["hauteur_m"]),
                step=0.1,
            )

        c4, c5, c6 = st.columns(3)

        with c4:
            mitoyennete = st.selectbox(
                "Mitoyenneté",
                options=["isole", "1_cote", "2_cotes", "3_cotes"],
                index=["isole", "1_cote", "2_cotes", "3_cotes"].index(
                    defaults["mitoyennete"]
                ),
                format_func=lambda x: MITOYENNETE_LABELS[x],
            )

        with c5:
            toiture_exposee = st.checkbox(
                "Toiture en contact avec l'extérieur",
                value=bool(defaults["toiture_exposee"]),
            )

        with c6:
            plancher_expose = st.checkbox(
                "Plancher bas sur extérieur / volume non chauffé",
                value=bool(defaults["plancher_expose"]),
            )

        sdep_est, vh_est, geo_meta = estimate_sdep_vh_typology_with_exposure(
            shab_m2=float(shab_m2),
            niveaux=int(niveaux),
            hauteur_m=float(hauteur_m),
            building_type_key=building_type_key,
            mitoyennete=mitoyennete,
            toiture_exposee=bool(toiture_exposee),
            plancher_expose=bool(plancher_expose),
        )

        st.caption(
            "Voici les paramètres estimés de votre bâtiment. Vous pouvez les "
            "modifier si vous disposez de données plus précises."
        )

        c7, c8 = st.columns(2)

        with c7:
            sdep_m2 = st.number_input(
                "Surface de déperdition estimée (m²)",
                min_value=1.0,
                value=float(sdep_est),
                step=10.0,
            )

        with c8:
            vh_m3 = st.number_input(
                "Volume habitable (m³)",
                min_value=1.0,
                value=float(vh_est),
                step=10.0,
            )

    elif sdep_mode == "Estimation par périmètre":
        st.caption(
            "Méthode intermédiaire si vous connaissez le périmètre extérieur du bâtiment."
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            shab_m2 = st.number_input(
                "Surface habitable (m²)",
                min_value=1.0,
                value=180.0,
                step=10.0,
            )

        with c2:
            niveaux = st.number_input(
                "Niveaux chauffés",
                min_value=1,
                max_value=20,
                value=2,
                step=1,
            )

        with c3:
            perimetre_m = st.number_input(
                "Périmètre extérieur (m)",
                min_value=1.0,
                value=40.0,
                step=1.0,
            )

        with c4:
            hauteur_m = st.number_input(
                "Hauteur sous plafond (m)",
                min_value=1.8,
                value=2.5,
                step=0.1,
            )

        toiture_exposee = st.checkbox(
            "Toiture en contact avec l'extérieur",
            value=True,
        )

        plancher_expose = st.checkbox(
            "Plancher bas sur extérieur / volume non chauffé",
            value=True,
        )

        sdep_est, vh_est = estimate_sdep_vh_from_perimeter(
            shab_m2,
            int(niveaux),
            perimetre_m,
            hauteur_m,
            toiture_exposee,
            plancher_expose,
        )

        st.caption(
            "Voici les paramètres estimés de votre bâtiment. Vous pouvez les "
            "modifier si vous disposez de données plus précises."
        )

        c5, c6 = st.columns(2)

        with c5:
            sdep_m2 = st.number_input(
                "Surface de déperdition estimée (m²)",
                min_value=1.0,
                value=float(sdep_est),
                step=10.0,
            )

        with c6:
            vh_m3 = st.number_input(
                "Volume habitable (m³)",
                min_value=1.0,
                value=float(vh_est),
                step=10.0,
            )

    else:
        st.caption(
            "Méthode la plus précise si vous connaissez les paramètres techniques du bâtiment."
        )

        c1, c2 = st.columns(2)

        with c1:
            sdep_m2 = st.number_input(
                "Surface de déperdition (m²)",
                min_value=1.0,
                value=350.0,
                step=10.0,
            )

        with c2:
            vh_m3 = st.number_input(
                "Volume habitable (m³)",
                min_value=1.0,
                value=600.0,
                step=10.0,
            )

    # ========================================================
    # SYSTÈME ACTUEL
    # ========================================================

    current_energy = None
    current_efficiency = None
    has_existing_ac = False
    eer_current_ac = None

    if project_type == "replacement":
        st.header("3. Système actuel")
        st.write(
            "Cette section concerne les paramètres de votre système de chauffage "
            "actuellement installé."
        )

        c1, c2 = st.columns(2)

        with c1:
            current_energy = st.selectbox(
                "Énergie du système actuel",
                options=list(ENERGY_CHOICES.keys()),
            )

        with c2:
            current_efficiency = st.number_input(
                "Rendement actuel",
                min_value=0.1,
                max_value=1.2,
                value=float(RENDEMENTS_STANDARDS[current_energy]),
                step=0.01,
            )
            st.caption(
                "Si vous ne connaissez pas le rendement de votre système de chauffage, "
                "veuillez laisser la valeur associée par défaut."
            )

        has_existing_ac = st.checkbox(
            "Le bâtiment a déjà une climatisation",
            value=False,
        )

        if has_existing_ac:
            eer_current_ac = st.number_input(
                "Rendement de la clim actuelle",
                min_value=0.5,
                value=EER_CLIM_ACTUEL_DEFAULT,
                step=0.1,
            )

    # ========================================================
    # RAFRAÎCHISSEMENT
    # ========================================================

    cooling_section_number = 4 if project_type == "replacement" else 3
    st.header(f"{cooling_section_number}. Rafraîchissement")
    st.write(
        "Cette section est consacrée à l'utilisation de la pompe à chaleur pour la climatisation."
    )

    want_cooling = st.checkbox(
        "Je souhaite utiliser la pompe à chaleur en mode climatisation",
        value=True,
    )

    surface_climatisee_m2 = 0.0
    cooling_mode = "no_cooling"
    vitrage_level = "moyen"
    solar_protection_level = "moyenne"
    usage_level = "normal"
    night_ventilation = False

    if want_cooling:
        cooling_mode_descriptions = {
            "no_cooling": "Aucun besoin de climatisation n'est ajouté au calcul.",
            "free_cooling": "La fraîcheur du sol est utilisée au maximum, avec une consommation électrique très limitée.",
            "hybrid": "Le système combine le refroidissement passif et un appoint actif lorsque les besoins sont plus importants.",
            "active_cooling": "La pompe à chaleur fonctionne activement en mode climatisation, avec une consommation électrique plus élevée.",
        }

        vitrage_descriptions = {
            "faible": "Peu de surfaces vitrées : les apports solaires restent limités.",
            "moyen": "Surface vitrée standard pour ce type de bâtiment.",
            "fort": "Grandes surfaces vitrées : les apports solaires peuvent augmenter les besoins de climatisation.",
        }

        solar_descriptions = {
            "bonne": "Protections efficaces, par exemple stores extérieurs, volets ou brise-soleil.",
            "moyenne": "Protections partielles ou utilisées de manière irrégulière.",
            "faible": "Peu ou pas de protections solaires efficaces.",
        }

        usage_descriptions = {
            "faible": "Occupation limitée et peu d'appareils produisant de la chaleur.",
            "normal": "Usage standard du bâtiment.",
            "eleve": "Occupation importante ou nombreux apports internes, par exemple équipements, éclairage ou informatique.",
        }

        c1, c2 = st.columns(2)

        with c1:
            default_surface = shab_m2 if shab_m2 is not None else 100.0

            surface_climatisee_m2 = st.number_input(
                "Surface climatisée (m²)",
                min_value=0.0,
                value=float(default_surface),
                step=10.0,
            )

        with c2:
            cooling_mode = st.selectbox(
                "Mode de refroidissement",
                options=["no_cooling", "free_cooling", "hybrid", "active_cooling"],
                format_func=lambda x: {
                    "no_cooling": "Pas de climatisation",
                    "free_cooling": "Free cooling / passif",
                    "hybrid": "Hybride",
                    "active_cooling": "Refroidissement actif",
                }[x],
            )
            st.caption(cooling_mode_descriptions[cooling_mode])

        c1, c2, c3 = st.columns(3)

        with c1:
            vitrage_level = st.selectbox(
                "Surface vitrée",
                ["faible", "moyen", "fort"],
            )
            st.caption(vitrage_descriptions[vitrage_level])

        with c2:
            solar_protection_level = st.selectbox(
                "Protections solaires",
                ["bonne", "moyenne", "faible"],
            )
            st.caption(solar_descriptions[solar_protection_level])

        with c3:
            usage_level = st.selectbox(
                "Occupation / apports",
                ["faible", "normal", "eleve"],
            )
            st.caption(usage_descriptions[usage_level])

        night_ventilation = st.checkbox(
            "Ventilation nocturne possible",
            value=False,
        )

    # ========================================================
    # OBJET FINAL
    # ========================================================

    return ProjectInputs(
        project_type=project_type,
        canton=canton,
        postcode=postcode,
        building_type_key=building_type_key,
        ubat=ubat,
        ventilation_r=ventilation_r,
        sdep_mode=sdep_mode,
        sdep_m2=sdep_m2,
        vh_m3=vh_m3,
        shab_m2=shab_m2,
        niveaux=int(niveaux) if niveaux is not None else None,
        perimetre_m=perimetre_m,
        hauteur_m=hauteur_m,
        toiture_exposee=toiture_exposee,
        plancher_expose=plancher_expose,
        forme_generale=forme_generale,
        mitoyennete=mitoyennete,
        longueur_m=longueur_m,
        largeur_m=largeur_m,
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



def render_deterministic_comparison_block(central: dict) -> None:
    """Affiche le scénario déterministe comme comparaison secondaire."""
    st.subheader("Comparaison : scénario déterministe sans chocs fossiles")
    st.caption(
        "Ce scénario utilise les trajectoires centrales de prix et ne simule pas les chocs "
        "historiques liés au gaz ou au mazout. Il sert de point de comparaison, mais la "
        "valeur de référence pour la décision est la médiane Monte Carlo affichée plus haut."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coût actuel année 1", format_chf(central.get("cout_actuel_total_year0"), decimals=0))
    c2.metric("Coût PAC année 1", format_chf(central.get("cout_pac_total_year0"), decimals=0))
    c3.metric("Économies année 1", format_chf(central.get("economies_annuelles_year0"), decimals=0))
    c4.metric("Temps de retour déterministe", format_payback(central.get("payback")))

    c5, c6 = st.columns(2)
    c5.metric("VAN déterministe", format_chf(central.get("npv"), decimals=0))
    c6.metric(
        "Hypothèse prix déterministe",
        f"Élec. {central.get('electricity_scenario', 'neutral')} / "
        f"référence {central.get('fuel_scenario', 'neutral')}",
    )



def safe_series_sum(value: object) -> float | None:
    """Somme une série ou une liste numérique en évitant de casser l'interface."""
    try:
        s = pd.Series(value).astype(float)
        if s.empty:
            return None
        return float(s.sum())
    except Exception:
        return None


def compute_co2_reduction_percent(central: dict) -> float | None:
    """Calcule la réduction cumulée d'émissions de CO₂ du cas PAC par rapport au système actuel."""
    emissions_actuel = central.get("emissions_actuel_series")
    emissions_pac = central.get("emissions_pac_series")

    total_actuel = safe_series_sum(emissions_actuel)
    total_pac = safe_series_sum(emissions_pac)

    if total_actuel is None or total_pac is None or total_actuel <= 0:
        return None

    return 100.0 * (total_actuel - total_pac) / total_actuel


def render_public_simulation_block(label: str, data: dict) -> None:
    """Affiche l'analyse avec incertitudes dans un langage plus accessible."""
    unc = data.get("uncertainty", {})

    if not isinstance(unc, dict) or not unc.get("available"):
        reason = unc.get("reason", "non_disponible") if isinstance(unc, dict) else "non_disponible"
        st.info(
            "Le calcul avec variations possibles n'est pas disponible pour ce cas. "
            f"Détail : {reason}."
        )
        return

    st.subheader("Résultats avec variations possibles")
    st.write(
        "Dans cette partie, l'outil refait le calcul un grand nombre de fois en faisant varier "
        "raisonnablement les paramètres incertains. De plus, cette partie tient notamment compte "
        "de possibles hausses fortes du gaz ou du mazout, qui ne sont pas incluses dans le "
        "récapitulatif présenté plus haut."
    )

    pb50 = unc.get("payback_p50")
    prob_25 = unc.get("amortization_probability_life", unc.get("amortization_probability"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Temps de retour médian", format_payback(pb50))
    c2.metric("Chance d'être rentabilisé en 25 ans", format_percent(prob_25, decimals=0))
    c3.metric("VAN médiane", format_chf(unc.get("npv_p50"), decimals=0))

    if pb50 is not None and not bool(unc.get("payback_median_representative", False)):
        st.warning(
            "Dans beaucoup de calculs, l'installation n'est pas rentabilisée dans la période étudiée. "
            "Le temps de retour affiché doit donc être lu avec prudence : regardez surtout la chance "
            "d'être rentabilisé."
        )

    with st.expander("Voir les graphiques des variations possibles"):
        st.caption(
            "Ces graphiques montrent la dispersion des résultats obtenus lorsque les hypothèses varient."
        )
        render_payback_simulation_chart(
            unc=unc,
            deterministic_payback=data.get("payback"),
        )
        render_npv_simulation_chart(
            unc=unc,
            deterministic_npv=data.get("npv"),
        )

# ============================================================
# RÉSULTATS
# ============================================================

def render_results(results: dict) -> None:
    chauffage = results["chauffage"]
    froid = results["froid"]
    pac = results["pac"]
    inputs = results["inputs"]
    station = results["station_climatique"]

    central = results.get("central")
    scenarios = results.get("scenarios", {})
    if central is None and "Central" in scenarios:
        central = scenarios["Central"]

    results_section_number = 5 if inputs.project_type == "replacement" else 4
    st.header(f"{results_section_number}. Résultats")

    # ========================================================
    # RÉCAPITULATIF RAPIDE
    # ========================================================

    st.subheader("Récapitulatif rapide")
    st.write(
        "Voici les principaux ordres de grandeur estimés pour votre bâtiment et pour "
        "l'installation d'une pompe à chaleur géothermique."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Puissance estimée de la pompe à chaleur",
        f"{chauffage['p_pointe_kw']:.1f} kW",
    )
    c2.metric(
        "Besoin annuel de chauffage",
        f"{chauffage['energie_annuelle_kwh']:.0f} kWh/an",
    )
    c3.metric(
        "Besoin annuel de climatisation",
        f"{froid['q_froid_utile_kwh_an']:.0f} kWh/an",
    )

    if inputs.project_type != "replacement":
        c4, c5 = st.columns(2)
        c4.metric(
            "CAPEX estimé",
            format_chf(pac["capex_brut"], decimals=0),
            help=(
                "Le CAPEX représente l'investissement initial estimé, comprenant "
                "le prix de la pompe à chaleur et de son installation."
            ),
        )
        c5.metric(
            "Coût de fonctionnement de la PAC — année 1",
            format_chf(
                central.get("cout_pac_total_year0") if central is not None else None,
                decimals=0,
            ),
            help=(
                "Coût estimé de l'électricité et de la maintenance de la pompe à chaleur "
                "pendant la première année."
            ),
        )
        return

    c4, c5, c6 = st.columns(3)
    c4.metric(
        "CAPEX brut estimé",
        format_chf(pac["capex_brut"], decimals=0),
        help=(
            "Le CAPEX représente l'investissement initial total, comprenant "
            "le prix de la pompe et de son installation, avant déduction des subventions."
        ),
    )
    c5.metric(
        "Subventions estimées",
        format_chf(pac["subvention"], decimals=0),
        help="Aides financières estimées selon le canton et le type de projet.",
    )
    c6.metric(
        "CAPEX net estimé",
        format_chf(pac["capex_net"], decimals=0),
        help="Coût de l'investissement initial estimé après déduction des subventions.",
    )

    if central is None:
        st.error("Résultat central introuvable. Vérifie que calcul_projet.py retourne bien la clé 'central'.")
        return

    # ========================================================
    # CAS SIMPLE / DÉTERMINISTE
    # ========================================================

    st.subheader("Résultat avec les hypothèses principales")
    st.write(
        "Ce premier résultat compare la pompe à chaleur géothermique avec votre système actuel "
        "en utilisant les hypothèses principales de prix et de performance."
    )
    st.warning(
        "Ce cas ne prend pas en compte les fortes hausses possibles du gaz ou du mazout. "
        "Ces variations sont étudiées dans la partie suivante."
    )

    d1, d2, d3 = st.columns(3)
    d1.metric(
        "Temps de retour estimé",
        format_payback(central.get("payback")),
        help="Durée nécessaire pour que les économies réalisées compensent le coût d'installation.",
    )
    d2.metric(
        "VAN estimée",
        format_chf(central.get("npv"), decimals=0),
        help="Bilan économique global sur la durée étudiée, en tenant compte du coût d'installation et des économies futures.",
    )
    d3.metric(
        "Économies estimées la première année",
        format_chf(central.get("economies_annuelles_year0"), decimals=0),
        help="Différence estimée entre le coût annuel du système actuel et celui de la pompe à chaleur.",
    )

    st.markdown("#### Coûts cumulés dans le temps")
    st.write(
        "Le graphique ci-dessous compare le coût cumulé du système actuel avec celui de la pompe "
        "à chaleur. Vous pouvez tester rapidement l'effet d'une hausse ou d'une baisse des prix."
    )

    if "interactive_adjustments" not in st.session_state:
        st.session_state["interactive_adjustments"] = default_interactive_adjustments()

    adjustments = st.session_state["interactive_adjustments"]
    if "electricity_price_pct" not in adjustments:
        adjustments["electricity_price_pct"] = int(adjustments.get("pac_annual_pct", 0))
    adjustments.pop("pac_annual_pct", None)

    interactive_metrics = render_interactive_cumulative_cost_chart(
        central=central,
        pac=pac,
        adjustments=adjustments,
    )
    render_interactive_adjustment_controls(results)

    m1, m2, m3 = st.columns(3)
    m1.metric("Temps de retour recalculé", format_payback(interactive_metrics.get("payback")))
    m2.metric("VAN recalculée", format_chf(interactive_metrics.get("van_recalculee"), decimals=0))
    m3.metric("CAPEX recalculé", format_chf(interactive_metrics.get("capex_net"), decimals=0))

    st.markdown("#### Réduction estimée des émissions de CO₂")
    st.write(
        "Une pompe à chaleur géothermique permet de réduire grandement les émissions de CO2, "
        "ce graphique représente la réduction estimée des émissions. Il ne prend pas en compte "
        "les émissions liées à la fabrication et au transport du système de chauffage, seulement "
        "les émissions liées au fonctionnement."
    )

    if central.get("emissions_actuel_series") is not None and central.get("emissions_pac_series") is not None:
        fig_co2 = build_cumulative_emissions_bar_chart(
            central["emissions_actuel_series"],
            central["emissions_pac_series"],
            "Cas principal",
        )
        st.pyplot(fig_co2)

        co2_reduction = compute_co2_reduction_percent(central)
        if co2_reduction is not None:
            st.success(
                f"Réduction cumulée estimée des émissions : **{co2_reduction:.0f} %** "
                "par rapport au système actuel."
            )
        else:
            st.caption("La réduction des émissions n'a pas pu être calculée automatiquement.")
    else:
        st.info("Les séries d'émissions ne sont pas disponibles pour ce scénario.")

    # ========================================================
    # RÉSULTATS AVEC VARIATIONS POSSIBLES
    # ========================================================

    render_public_simulation_block("Central", central)

# ============================================================
# MAIN
# ============================================================

def init_session_state() -> None:
    if "results" not in st.session_state:
        st.session_state["results"] = None

    if "interactive_adjustments" not in st.session_state:
        st.session_state["interactive_adjustments"] = default_interactive_adjustments()


def main() -> None:
    st.title("Rentabilité d'une PAC géothermique en Suisse")

    init_session_state()

    try:
        inputs = build_inputs_from_form()

        if st.button("Lancer le calcul", type="primary"):
            if not st.session_state.get("postcode_valid", False):
                st.error("Le calcul ne peut pas être lancé tant que le code postal n'est pas valide.")
            else:
                st.session_state["results"] = evaluer_projet(inputs)
                reset_interactive_adjustments()

        if st.session_state["results"] is not None:
            render_results(st.session_state["results"])

    except Exception as e:
        st.error(f"Erreur : {e}")
        st.info(
            "Vérifie en priorité : le code postal, le canton, les fichiers CSV de scénarios, "
            "les données climatiques, le fichier de calibration du froid et l'installation de Plotly."
        )
        with st.expander("Détail technique de l'erreur"):
            st.exception(e)


if __name__ == "__main__":
    main()
