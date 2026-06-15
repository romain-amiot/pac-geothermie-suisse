""" 
cd "C:\Travail Master\CodePAC"
py -m streamlit run app.py 
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from Fonctions.puissance_pointe import (
    calcul_pointe_et_energie_annuelle,
    UBAT_CHOICES,
    VENTILATION_CHOICES,
    ask_positive_float,
    ask_choice,
)
from Fonctions.rentabilite import (
    gshp_capex_net_after_subsidy,
    couts_annuels_gshp_zuberi,
    load_price_path_electricity_ttc_by_canton,
    load_price_path_fuel,
    annual_cost_series_systeme_actuel,
    annual_cost_series_gshp,
    payback_discounted_from_cashflows,
    project_requires_reference_system,
    annual_om_cost_chf,
)
from Fonctions.climat import (
    cooling_hours_for_postcode,
    cooling_degree_hours_for_postcode,
)

# ============================================================
#         PARAMÈTRES MODÈLE FROID SEMI-EMPIRIQUE
# ============================================================

CDH_REF = 1600.0  # Genève comme référence suisse simple

COOLING_QREF_BY_TYPE_AND_MODE = {
    "maison_individuelle": {
        "no_cooling": 0.0,
        "free_cooling": 8.0,
        "hybrid": 14.0,
        "active_cooling": 20.0,
    },
    "residentiel_collectif": {
        "no_cooling": 0.0,
        "free_cooling": 5.0,
        "hybrid": 7.0,
        "active_cooling": 10.0,
    },
    "grand_batiment_compact": {
        "no_cooling": 0.0,
        "free_cooling": 10.0,
        "hybrid": 16.0,
        "active_cooling": 21.0,
    },
    "mixte": {
        "no_cooling": 0.0,
        "free_cooling": 12.0,
        "hybrid": 16.0,
        "active_cooling": 18.0,
    },
    "activites": {
        "no_cooling": 0.0,
        "free_cooling": 9.0,
        "hybrid": 12.0,
        "active_cooling": 14.0,
    },
    "equipement_collectif": {
        "no_cooling": 0.0,
        "free_cooling": 7.0,
        "hybrid": 10.0,
        "active_cooling": 14.0,
    },
}

COOLING_SPF_BY_MODE = {
    "free_cooling": 20.0,
    "hybrid": 10.0,
    "active_cooling": 5.2,
}

F_VITRAGE = {
    "faible": 0.85,
    "moyen": 1.00,
    "fort": 1.20,
}

F_SOLAIRE = {
    "bonne": 0.75,
    "moyenne": 1.00,
    "faible": 1.25,
}

F_INERTIE = {
    "forte": 0.90,
    "normale": 1.00,
    "faible": 1.10,
}

F_USAGE = {
    "faible": 0.90,
    "normal": 1.00,
    "eleve": 1.15,
}

F_NIGHT = {
    True: 0.85,
    False: 1.00,
}

# ============================================================
#                 CHEMINS CSV SCENARIOS (25 ans)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SCEN_ELEC_CSV = str(BASE_DIR / "Data" / "processed" / "scenarios_electricity_ttc_by_canton.csv")
SCEN_GAS_CSV = str(BASE_DIR / "Data" / "processed" / "scenarios_gas.csv")
SCEN_OIL_CSV = str(BASE_DIR / "Data" / "processed" / "scenarios_mazout.csv")

# ============================================================
#                     PARAMÈTRES FIXES NON-PRIX
# ============================================================

RENDEMENTS_STANDARDS: dict[str, float] = {
    "Électricité (réseau)": 1.00,
    "Gaz naturel": 0.90,
    "Mazout (fioul)": 0.90,
}

CO2_FACTORS_G_PER_KWH: dict[str, float] = {
    "Électricité (réseau)": 90,
    "Gaz naturel": 230,
    "Mazout (fioul)": 324,
}

ENERGY_CHOICES = {
    "1": "Électricité (réseau)",
    "2": "Gaz naturel",
    "3": "Mazout (fioul)",
}

T_BASE_COOL_C = 24.0
EER_CLIM_ACTUEL_DEFAULT = 3.0
EER_GSHP_COOL_DEFAULT = 4.0
DEFAULT_CEILING_HEIGHT_M: float = 2.5

# ============================================================
#      TYPOLOGIES BÂTIMENT - VERSION FINALE SIMPLIFIÉE
# ============================================================

K_TYPO_SIMPLE = {
    "maison_individuelle": 2.10,
    "residentiel_collectif": 1.92,
    "grand_batiment_compact": 1.71,
    "mixte": 1.69,
    "activites": 1.99,
    "equipement_collectif": 2.15,
}

TYPOLOGIES_MENU = [
    (
        "maison_individuelle",
        "Maison individuelle",
        "Villa, pavillon ou maison occupée par un seul ménage. Bâtiment résidentiel peu compact, généralement 1 à 3 niveaux.",
        2.10,
    ),
    (
        "residentiel_collectif",
        "Immeuble résidentiel collectif",
        "Bâtiment d’habitation comprenant plusieurs logements : maison à 2 logements, petit immeuble, immeuble d’habitation classique.",
        1.92,
    ),
    (
        "grand_batiment_compact",
        "Grand bâtiment compact",
        "Grand bâtiment avec plusieurs niveaux et une forme compacte : grand immeuble, résidence importante, grand bâtiment tertiaire compact.",
        1.71,
    ),
    (
        "mixte",
        "Bâtiment mixte",
        "Bâtiment combinant plusieurs usages, par exemple logements avec commerces, bureaux ou autres activités.",
        1.69,
    ),
    (
        "activites",
        "Bâtiment d’activités / tertiaire",
        "Bureaux, ateliers, locaux professionnels, bâtiment artisanal, bâtiment logistique ou entrepôt chauffé.",
        1.99,
    ),
    (
        "equipement_collectif",
        "Équipement collectif / bâtiment public",
        "École, crèche, EMS, bâtiment communal, salle de sport, bâtiment administratif public ou assimilé.",
        2.15,
    ),
]

# ============================================================
#  Correspondance typologie outil -> building_type tarifaire
# ============================================================

BUILDING_TYPE_PRICE_MAP = {
    "maison_individuelle": "maison_individuelle",
    "residentiel_collectif": "appartement",
    "grand_batiment_compact": "appartement",
    "mixte": "appartement",
    "activites": "maison_individuelle",
    "equipement_collectif": "maison_individuelle",
}

# ============================================================
#      SCÉNARIOS RELATIFS À LA FAVORABILITÉ POUR LA PAC
# ============================================================

PAC_SCENARIO_MAP = {
    "optimistic": {
        "electricity": "optimistic",
        "fuel": "pessimistic",
        "label": "Favorable à la PAC",
    },
    "neutral": {
        "electricity": "neutral",
        "fuel": "neutral",
        "label": "Neutre",
    },
    "pessimistic": {
        "electricity": "pessimistic",
        "fuel": "optimistic",
        "label": "Défavorable à la PAC",
    },
}


# ============================================================
#                     OUTILS D'INPUT
# ============================================================

def ask_float_default(prompt: str, default: float, min_value: float | None = None) -> float:
    while True:
        raw = input(f"{prompt} [{default}] : ").strip().replace(",", ".")
        if raw == "":
            val = float(default)
        else:
            try:
                val = float(raw)
            except ValueError:
                print("Entrée invalide. Exemple attendu : 0.85 ou 250.5")
                continue

        if min_value is not None and val < min_value:
            print(f"Veuillez entrer une valeur >= {min_value}.")
            continue
        return val


def ask_yes_no_default(prompt: str, default: bool) -> bool:
    d = "o" if default else "n"
    while True:
        raw = input(f"{prompt} (o/n) [{d}] : ").strip().lower()
        if raw == "":
            return default
        if raw in ("o", "oui", "y", "yes"):
            return True
        if raw in ("n", "non", "no"):
            return False
        print("Réponse invalide. Répondre par o/n.")


def ask_project_type() -> str:
    while True:
        print("\n=== Type de projet ===")
        print("  1) Nouveau bâtiment")
        print("  2) Remplacer un ancien système de chauffage")
        choice = input("Votre choix : ").strip()
        if choice == "1":
            return "new_building"
        if choice == "2":
            return "replacement"
        print("Choix invalide, recommencez.")


def ask_energy_choice() -> str:
    while True:
        print("\n=== Choix de l'énergie du système actuel ===")
        print("  1) Électricité (réseau)")
        print("  2) Gaz naturel")
        print("  3) Mazout (fioul)")
        choice = input("Votre choix : ").strip()

        if choice in ENERGY_CHOICES:
            return ENERGY_CHOICES[choice]

        print("Choix invalide, recommencez.")


def ask_rendement_for_energy(energie_label: str) -> float:
    std = RENDEMENTS_STANDARDS.get(energie_label)
    if std is None:
        return ask_float_default("Entrez le rendement de votre système (0–1)", 0.85, min_value=0.01)

    while True:
        print("\n=== Rendement du système actuel ===")
        print(f"  1) Rendement standard associé : {std:.2f}")
        print("  2) Rendement personnalisé (saisie)")
        choice = input("Votre choix : ").strip()
        if choice == "1":
            return std
        if choice == "2":
            return ask_float_default("Entrez le rendement de votre système (0–1)", std, min_value=0.01)
        print("Choix invalide, recommencez.")


def ask_co2_factor_for_energy_g_per_kwh(energie_label: str) -> float:
    f = CO2_FACTORS_G_PER_KWH.get(energie_label)
    if f is None:
        return ask_float_default("Facteur CO₂ (gCO2e/kWh)", 200.0, min_value=0.0)
    return float(f)


def g_per_kwh_to_kg_per_kwh(g_per_kwh: float) -> float:
    return g_per_kwh / 1000.0


def ask_int_in_choices(prompt: str, choices: list[int], default: int) -> int:
    choices_str = ", ".join(str(c) for c in choices)
    while True:
        raw = input(f"{prompt} ({choices_str}) [{default}] : ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
        except ValueError:
            print("Entrée invalide. Exemple : 2")
            continue
        if val not in choices:
            print("Valeur hors choix.")
            continue
        return val


def ask_scenario() -> str:
    while True:
        print("\n=== Scénario d'évolution des prix (25 ans) ===")
        print("Le scénario représente ici le niveau de favorabilité pour la PAC :")
        print("  1) Favorable à la PAC  → électricité optimiste / fossiles pessimistes")
        print("  2) Neutre              → électricité neutre / fossiles neutres")
        print("  3) Défavorable à la PAC→ électricité pessimiste / fossiles optimistes")
        c = input("Votre choix : ").strip()
        if c == "1":
            return "optimistic"
        if c == "2":
            return "neutral"
        if c == "3":
            return "pessimistic"
        print("Choix invalide.")


def ask_canton_for_electricity() -> str:
    return input("\nCanton pour le prix de l'électricité (ex: Vaud, Genève, Bern) : ").strip()


def choisir_typologie_batiment() -> dict:
    print("\n=== Type de bâtiment ===")
    print("Choisissez la catégorie la plus proche de votre bâtiment.\n")

    for i, (cle, label, desc, k) in enumerate(TYPOLOGIES_MENU, start=1):
        print(f"  {i}) {label}  → k = {k:.2f}")
        print(f"     {desc}\n")

    print("Conseil : basez-vous sur l’usage principal et la forme générale du bâtiment.")

    while True:
        saisie = input("Votre choix : ").strip()
        try:
            choix = int(saisie)
            if 1 <= choix <= len(TYPOLOGIES_MENU):
                cle, label, desc, k = TYPOLOGIES_MENU[choix - 1]
                print("\nTypologie sélectionnée :")
                print(f"  - Catégorie   : {label}")
                print(f"  - Description : {desc}")
                print(f"  - Coefficient k retenu : {k:.2f}\n")
                return {
                    "cle": cle,
                    "label": label,
                    "description": desc,
                    "k": float(k),
                }
            print("Choix invalide. Entrez un numéro de la liste.")
        except ValueError:
            print("Entrée invalide. Veuillez saisir un numéro.")


# ============================================================
#     ESTIMATION Sdép/Vh : AVEC périmètre (sinon via k)
# ============================================================

def estimate_sdep_vh_with_perimeter() -> tuple[float, float, dict]:
    print("\n=== Estimation via périmètre extérieur ===\n")
    shab = ask_positive_float("Surface chauffée (Shab) en m² (tous étages compris) : ")
    niveaux = ask_int_in_choices("Nombre de niveaux chauffés", [1, 2, 3, 4], default=1)
    perimetre = ask_positive_float("Périmètre extérieur du bâtiment (m) : ")
    h = ask_float_default("Hauteur sous plafond moyenne (m)", DEFAULT_CEILING_HEIGHT_M, min_value=1.8)

    a_foot = shab / niveaux

    toiture_exposee = ask_yes_no_default("La toiture est-elle en contact avec l'extérieur ?", True)
    plancher_expose = ask_yes_no_default(
        "Le plancher bas est-il sur extérieur/volume non chauffé (cave/garage) ?",
        True
    )

    s_murs = perimetre * (niveaux * h)
    s_toit = a_foot if toiture_exposee else 0.0
    s_plancher = a_foot if plancher_expose else 0.0

    sdep_est = s_murs + s_toit + s_plancher
    vh_est = shab * h

    print("\n--- Estimation ---")
    print(f"Murs extérieurs estimés       : {s_murs:.1f} m² (P × hauteur totale)")
    print(f"Toiture ajoutée               : {s_toit:.1f} m²")
    print(f"Plancher bas ajouté           : {s_plancher:.1f} m²")
    print(f"Sdép estimée                  : {sdep_est:.1f} m²")
    print(f"Vh estimé                     : {vh_est:.1f} m³")
    print("Vous pouvez corriger ces valeurs si besoin.\n")

    sdep = ask_float_default("Sdép finale (m²)", sdep_est, min_value=1.0)
    vh = ask_float_default("Vh final (m³)", vh_est, min_value=1.0)

    meta = {
        "mode": "perimetre",
        "shab_m2": shab,
        "niveaux": niveaux,
        "perimetre_m": perimetre,
        "h_m": h,
        "a_foot_m2": a_foot,
        "toiture_exposee": toiture_exposee,
        "plancher_expose": plancher_expose,
        "s_murs_m2": s_murs,
        "s_toit_m2": s_toit,
        "s_plancher_m2": s_plancher,
        "sdep_est_m2": sdep_est,
        "vh_est_m3": vh_est,
    }
    return sdep, vh, meta


def estimate_sdep_vh_with_k(typologie: dict) -> tuple[float, float, dict]:
    print("\n=== Estimation via typologie (k simplifié calibré) ===\n")

    shab = ask_positive_float("Surface chauffée (Shab) en m² (tous étages compris) : ")
    niveaux = ask_int_in_choices("Nombre de niveaux chauffés", [1, 2, 3, 4, 5, 6, 7, 8], default=2)
    h = ask_float_default("Hauteur sous plafond moyenne (m)", DEFAULT_CEILING_HEIGHT_M, min_value=1.8)

    k_typo = float(typologie["k"])
    sdep_est = k_typo * shab
    vh_est = shab * h

    print("\n--- Estimation ---")
    print(f"Typologie                    : {typologie['label']}")
    print(f"Description                  : {typologie['description']}")
    print(f"k typologique retenu         : {k_typo:.2f}")
    print(f"Nombre de niveaux saisi      : {niveaux}")
    print(f"Sdép estimée                 : {sdep_est:.1f} m² (= k × Shab)")
    print(f"Vh estimé                    : {vh_est:.1f} m³ (= Shab × h)")
    print("Le coefficient k est ici un coefficient global calibré par typologie.")
    print("Vous pouvez corriger les valeurs proposées si besoin.\n")

    sdep = ask_float_default("Sdép finale (m²)", sdep_est, min_value=1.0)
    vh = ask_float_default("Vh final (m³)", vh_est, min_value=1.0)

    meta = {
        "mode": "k_typologique",
        "building_type_key": typologie["cle"],
        "building_type_label": typologie["label"],
        "building_type_description": typologie["description"],
        "shab_m2": shab,
        "niveaux": niveaux,
        "h_m": h,
        "k_typo": k_typo,
        "sdep_est_m2": sdep_est,
        "vh_est_m3": vh_est,
    }
    return sdep, vh, meta


def ask_sdep_vh_inputs(typologie: dict) -> tuple[float, float, dict]:
    while True:
        print("\n=== Sdép & Vh ===")
        print("  1) Je connais Sdép et Vh (saisie directe)")
        print("  2) Je ne connais pas Sdép / Vh (estimation)")
        choice = input("Votre choix : ").strip()

        if choice == "1":
            sdep = ask_positive_float("Entrez la surface totale des parois Sdép (m²) : ")
            vh = ask_positive_float("Indiquez le volume habitable Vh (m³) : ")
            meta = {
                "mode": "direct",
                "building_type_key": typologie["cle"],
                "building_type_label": typologie["label"],
                "building_type_description": typologie["description"],
                "k_typo": typologie["k"],
            }
            return sdep, vh, meta

        if choice == "2":
            knows_p = ask_yes_no_default("Connaissez-vous le périmètre extérieur du bâtiment ?", False)
            if knows_p:
                sdep, vh, meta = estimate_sdep_vh_with_perimeter()
                meta["building_type_key"] = typologie["cle"]
                meta["building_type_label"] = typologie["label"]
                meta["building_type_description"] = typologie["description"]
                meta["k_typo"] = typologie["k"]
                return sdep, vh, meta
            return estimate_sdep_vh_with_k(typologie)

        print("Choix invalide, recommencez.")



def ask_cooling_mode() -> str:
    while True:
        print("\n=== Mode de refroidissement ===")
        print("  1) Pas de climatisation")
        print("  2) Free cooling / géocooling passif")
        print("  3) Système hybride (free cooling + actif)")
        print("  4) Refroidissement actif")
        c = input("Votre choix : ").strip()
        if c == "1":
            return "no_cooling"
        if c == "2":
            return "free_cooling"
        if c == "3":
            return "hybrid"
        if c == "4":
            return "active_cooling"
        print("Choix invalide.")


def ask_cooling_surface(default_surface: float | None = None) -> float:
    if default_surface is not None and default_surface > 0:
        return ask_float_default(
            "Surface réellement climatisée / rafraîchie (m²)",
            default_surface,
            min_value=0.0,
        )
    return ask_positive_float("Surface réellement climatisée / rafraîchie (m²) : ")


def ask_discrete_choice(prompt: str, mapping: dict[str, tuple[str, str]]) -> str:
    while True:
        print(prompt)
        for key, (label, code) in mapping.items():
            print(f"  {key}) {label}")
        c = input("Votre choix : ").strip()
        if c in mapping:
            return mapping[c][1]
        print("Choix invalide.")


def estimate_cooling_need_kwh(
    *,
    building_type_key: str,
    cooling_mode: str,
    surface_climatisee_m2: float,
    cooling_degree_hours: float,
    vitrage_level: str,
    solar_protection_level: str,
    inertia_level: str,
    usage_level: str,
    night_ventilation: bool,
) -> dict:
    if cooling_mode == "no_cooling" or surface_climatisee_m2 <= 0:
        return {
            "q_ref_kwh_m2a": 0.0,
            "f_climat": 0.0,
            "f_vitrage": 1.0,
            "f_solaire": 1.0,
            "f_inertie": 1.0,
            "f_usage": 1.0,
            "f_night": 1.0,
            "q_froid_utile_kwh_an": 0.0,
            "spf_froid": None,
            "conso_elec_clim_kwh_an": 0.0,
        }

    q_ref = COOLING_QREF_BY_TYPE_AND_MODE[building_type_key][cooling_mode]
    f_climat = cooling_degree_hours / CDH_REF if CDH_REF > 0 else 1.0
    f_climat = min(max(f_climat, 0.15), 1.40)

    f_vitrage = F_VITRAGE[vitrage_level]
    f_solaire = F_SOLAIRE[solar_protection_level]
    f_inertie = F_INERTIE[inertia_level]
    f_usage = F_USAGE[usage_level]
    f_night = F_NIGHT[night_ventilation]

    q_froid_utile = (
        surface_climatisee_m2
        * q_ref
        * f_climat
        * f_vitrage
        * f_solaire
        * f_inertie
        * f_usage
        * f_night
    )

    spf_froid = COOLING_SPF_BY_MODE[cooling_mode]
    conso_elec = q_froid_utile / spf_froid if spf_froid and spf_froid > 0 else 0.0

    return {
        "q_ref_kwh_m2a": q_ref,
        "f_climat": f_climat,
        "f_vitrage": f_vitrage,
        "f_solaire": f_solaire,
        "f_inertie": f_inertie,
        "f_usage": f_usage,
        "f_night": f_night,
        "q_froid_utile_kwh_an": q_froid_utile,
        "spf_froid": spf_froid,
        "conso_elec_clim_kwh_an": conso_elec,
    }



def describe_om_method() -> str:
    return "Interpolation log-log fondée sur ancrages JRC (proxy Allemagne)"

# ============================================================
#                        DEMO
# ============================================================

def run_demo() -> None:
    print("=== Entrées pour calcul chauffage (puissance_pointe) ===\n")

    project_type = ask_project_type()
    is_replacement = project_requires_reference_system(project_type)

    typologie = choisir_typologie_batiment()
    building_type_price = BUILDING_TYPE_PRICE_MAP[typologie["cle"]]
    postcode = input("Entrez le code postal (ex: 1200) : ").strip()
    ubat = ask_choice("\nSélectionnez le coefficient Ubat (W/m²K) :", UBAT_CHOICES)

    sdep_m2, vh_m3, sdep_meta = ask_sdep_vh_inputs(typologie)
    ventilation_r = ask_choice(
        "\nChoisissez le type de ventilation (coefficient R) :",
        VENTILATION_CHOICES,
        default="1",
    )

    res_ch = calcul_pointe_et_energie_annuelle(
        postcode=postcode,
        ubat=ubat,
        sdep_m2=sdep_m2,
        ventilation_r=ventilation_r,
        vh_m3=vh_m3,
    )

    besoin_total = res_ch["energie_annuelle_kwh"]
    P_max_kW = res_ch["p_pointe_kw"]
    hdd = res_ch["hdd"]
    dp = res_ch["dp_w_per_k"]

    print("\n=== Paramètres techno-économiques (CHF) ===\n")

    scenario = ask_scenario()
    canton_prix = ask_canton_for_electricity()

    scenario_cfg = PAC_SCENARIO_MAP[scenario]
    scenario_elec = scenario_cfg["electricity"]
    scenario_fuel = scenario_cfg["fuel"]
    scenario_label = scenario_cfg["label"]

    P_elec = load_price_path_electricity_ttc_by_canton(
        SCEN_ELEC_CSV,
        canton=canton_prix,
        building_type=building_type_price,
        scenario=scenario_elec,
    )
    P_gas = load_price_path_fuel(SCEN_GAS_CSV, scenario=scenario_fuel)
    P_oil = load_price_path_fuel(SCEN_OIL_CSV, scenario=scenario_fuel)

    # --------------------------------------------------------
    # Cas remplacement : on demande l'ancien système
    # --------------------------------------------------------
    energie_label = None
    rendement_actuel = None
    co2_factor_actuel_g_kwh = None
    co2_factor_actuel_kg_kwh = None
    P_actuel_heat = None
    scenario_actuel = None
    conso_energie_actuelle_kwh = None
    emissions_chauffage_actuel_kg = 0.0
    om_systeme_actuel_chf_per_year = 0.0

    if is_replacement:
        energie_label = ask_energy_choice()
        rendement_actuel = ask_rendement_for_energy(energie_label)

        co2_factor_actuel_g_kwh = ask_co2_factor_for_energy_g_per_kwh(energie_label)
        co2_factor_actuel_kg_kwh = g_per_kwh_to_kg_per_kwh(co2_factor_actuel_g_kwh)

        if energie_label == "Gaz naturel":
            P_actuel_heat = P_gas
            scenario_actuel = scenario_fuel
        elif energie_label == "Mazout (fioul)":
            P_actuel_heat = P_oil
            scenario_actuel = scenario_fuel
        elif energie_label == "Électricité (réseau)":
            P_actuel_heat = P_elec
            scenario_actuel = scenario_elec
        else:
            raise ValueError("Énergie non supportée par les scénarios.")

        om_systeme_actuel_chf_per_year = annual_om_cost_chf(energie_label, P_max_kW)

        conso_energie_actuelle_kwh = besoin_total / rendement_actuel
        emissions_chauffage_actuel_kg = conso_energie_actuelle_kwh * co2_factor_actuel_kg_kwh

    # --------------------------------------------------------
    # Besoins de froid - modèle semi-empirique
    # --------------------------------------------------------
    cooling_hours = cooling_hours_for_postcode(postcode)
    cooling_degree_hours = cooling_degree_hours_for_postcode(postcode)

    print("\n=== Paramètres bâtiment pour le froid ===")
    want_cooling_model = ask_yes_no_default(
        "Souhaitez-vous modéliser un besoin de rafraîchissement ?",
        True
    )

    cooling_mode = "no_cooling"
    surface_climatisee_m2 = 0.0
    vitrage_level = "moyen"
    solar_protection_level = "moyenne"
    inertia_level = "normale"
    usage_level = "normal"
    night_ventilation = False

    if want_cooling_model:
        default_surface = sdep_meta.get("shab_m2") if isinstance(sdep_meta, dict) else None
        surface_climatisee_m2 = ask_cooling_surface(default_surface=default_surface)
        cooling_mode = ask_cooling_mode()

        vitrage_level = ask_discrete_choice(
            "\nNiveau de surface vitrée :",
            {
                "1": ("Faible", "faible"),
                "2": ("Moyen", "moyen"),
                "3": ("Fort", "fort"),
            }
        )

        solar_protection_level = ask_discrete_choice(
            "\nQualité des protections solaires :",
            {
                "1": ("Bonne (stores extérieurs, brise-soleil, ombrage efficace)", "bonne"),
                "2": ("Moyenne", "moyenne"),
                "3": ("Faible / aucune", "faible"),
            }
        )

        inertia_level = ask_discrete_choice(
            "\nInertie thermique du bâtiment :",
            {
                "1": ("Forte", "forte"),
                "2": ("Normale", "normale"),
                "3": ("Faible", "faible"),
            }
        )

        usage_level = ask_discrete_choice(
            "\nNiveau d'apports internes / occupation estivale :",
            {
                "1": ("Faible", "faible"),
                "2": ("Normal", "normal"),
                "3": ("Élevé", "eleve"),
            }
        )

        night_ventilation = ask_yes_no_default(
            "Ventilation nocturne / rafraîchissement nocturne possible ?",
            False
        )

    cool_res = estimate_cooling_need_kwh(
        building_type_key=typologie["cle"],
        cooling_mode=cooling_mode,
        surface_climatisee_m2=surface_climatisee_m2,
        cooling_degree_hours=cooling_degree_hours,
        vitrage_level=vitrage_level,
        solar_protection_level=solar_protection_level,
        inertia_level=inertia_level,
        usage_level=usage_level,
        night_ventilation=night_ventilation,
    )

    energie_froid_utile_kwh = cool_res["q_froid_utile_kwh_an"]

    eer_actuel = None
    conso_elec_clim_actuelle_kwh = 0.0
    emissions_clim_actuelle_kg = 0.0

    co2_factor_elec_g_kwh = CO2_FACTORS_G_PER_KWH["Électricité (réseau)"]
    co2_factor_elec_kg_kwh = g_per_kwh_to_kg_per_kwh(co2_factor_elec_g_kwh)

    if is_replacement:
        has_ac = ask_yes_no_default("Avez-vous déjà une climatisation ?", False)

        if has_ac and energie_froid_utile_kwh > 0:
            eer_actuel = ask_float_default(
                "Performance clim actuelle (EER/COP froid)",
                EER_CLIM_ACTUEL_DEFAULT,
                min_value=0.5
            )
            conso_elec_clim_actuelle_kwh = energie_froid_utile_kwh / eer_actuel
            emissions_clim_actuelle_kg = conso_elec_clim_actuelle_kwh * co2_factor_elec_kg_kwh
    else:
        has_ac = False

    # --------------------------------------------------------
    # PAC géothermique
    # --------------------------------------------------------
    om_pac_chf_per_year = annual_om_cost_chf("PAC géothermique", P_max_kW)

    gshp = couts_annuels_gshp_zuberi(
        p_nom_kw=P_max_kW,
        chaleur_utile_kwh=besoin_total,
        prix_elec_chf_kwh=float(P_elec.iloc[0]),
        om_chf_per_year=om_pac_chf_per_year,
    )

    capex_brut, subvention_pac, capex_net = gshp_capex_net_after_subsidy(
        p_nom_kw=P_max_kW,
        canton=canton_prix,
        project_type=project_type,
        current_energy_label=energie_label,
    )

    cout_pac_chauffage_om = gshp.om_chf_per_year
    emissions_chauffage_pac_kg = gshp.conso_elec_kwh_per_year * co2_factor_elec_kg_kwh

    eer_gshp_cool = cool_res["spf_froid"]
    conso_elec_clim_pac_kwh = cool_res["conso_elec_clim_kwh_an"]
    emissions_clim_pac_kg = conso_elec_clim_pac_kwh * co2_factor_elec_kg_kwh

    want_gshp_cooling = cooling_mode != "no_cooling" and energie_froid_utile_kwh > 0

    # --------------------------------------------------------
    # Coûts PAC
    # --------------------------------------------------------
    cost_pac_heat_series = annual_cost_series_gshp(
        chaleur_utile_kwh=besoin_total,
        price_elec_path_chf_kwh=P_elec,
        spf=gshp.spf,
        om_chf_per_year=gshp.om_chf_per_year,
    )

    if want_gshp_cooling and energie_froid_utile_kwh > 0 and eer_gshp_cool is not None:
        cost_pac_cool_series = conso_elec_clim_pac_kwh * P_elec
    else:
        cost_pac_cool_series = pd.Series(0.0, index=P_elec.index)

    cost_pac_total_series = cost_pac_heat_series.add(cost_pac_cool_series, fill_value=0.0)

    cout_pac_chauffage_year0 = float(cost_pac_heat_series.iloc[0])
    cout_clim_pac_year0 = float(cost_pac_cool_series.iloc[0])
    cout_pac_total_year0 = float(cost_pac_total_series.iloc[0])

    # --------------------------------------------------------
    # Coûts système actuel uniquement si remplacement
    # --------------------------------------------------------
    cost_actuel_heat_series = None
    cost_actuel_cool_series = None
    cost_actuel_total_series = None
    cout_chauffage_actuel_year0 = None
    cout_clim_actuelle_year0 = None
    cout_actuel_total_year0 = None
    res_pb = None
    economies_annuelles_year0 = None

    if is_replacement:
        cost_actuel_heat_series = annual_cost_series_systeme_actuel(
            chaleur_utile_kwh=besoin_total,
            rendement=rendement_actuel,
            price_path_chf_kwh=P_actuel_heat,
            om_chf_per_year=om_systeme_actuel_chf_per_year,
        )

        if has_ac and energie_froid_utile_kwh > 0 and eer_actuel is not None:
            cost_actuel_cool_series = (energie_froid_utile_kwh / eer_actuel) * P_elec
        else:
            cost_actuel_cool_series = pd.Series(0.0, index=P_elec.index)

        cost_actuel_total_series = cost_actuel_heat_series.add(cost_actuel_cool_series, fill_value=0.0)

        cout_chauffage_actuel_year0 = float(cost_actuel_heat_series.iloc[0])
        cout_clim_actuelle_year0 = float(cost_actuel_cool_series.iloc[0])
        cout_actuel_total_year0 = float(cost_actuel_total_series.iloc[0])

        economies_annuelles_year0 = cout_actuel_total_year0 - cout_pac_total_year0

        res_pb = payback_discounted_from_cashflows(
            capex_chf=capex_net,
            cost_ref=cost_actuel_total_series,
            cost_new=cost_pac_total_series,
            discount_rate=0.03,
        )

    # --------------------------------------------------------
    # Émissions
    # --------------------------------------------------------
    emissions_pac_total_kg = emissions_chauffage_pac_kg + emissions_clim_pac_kg

    emissions_actuel_total_kg = None
    delta_emissions_kg = None
    if is_replacement:
        emissions_actuel_total_kg = emissions_chauffage_actuel_kg + emissions_clim_actuelle_kg
        delta_emissions_kg = emissions_actuel_total_kg - emissions_pac_total_kg

    # ============================================================
    # AFFICHAGE
    # ============================================================

    print("\n\n==================== RÉSULTATS ====================\n")

    print("=== Projet ===")
    print(f"Type de projet               : {'Remplacement ancien chauffage' if is_replacement else 'Nouveau bâtiment'}\n")

    print("=== Typologie bâtiment retenue ===")
    print(f"Catégorie                    : {typologie['label']}")
    print(f"Description                  : {typologie['description']}")
    print(f"k utilisé                    : {typologie['k']:.2f}\n")

    print("=== Sdép / Vh ===")
    print(f"Mode de saisie               : {sdep_meta.get('mode')}")
    print(f"Sdép utilisée                : {sdep_m2:.1f} m²")
    print(f"Vh utilisé                   : {vh_m3:.1f} m³")
    if "shab_m2" in sdep_meta:
        print(f"Shab saisie                  : {sdep_meta['shab_m2']:.1f} m²")
    if "niveaux" in sdep_meta:
        print(f"Niveaux                      : {sdep_meta['niveaux']}")
    if "k_typo" in sdep_meta:
        print(f"k typologique                : {sdep_meta['k_typo']:.2f}")
    print("")

    print("=== Calcul chauffage (via Dp + HDD) ===")
    print(f"Code postal                  : {postcode}")
    print(f"HDD (climat local)           : {hdd:.0f} °C·jours")
    print(f"Dp (déperditions)            : {dp:.1f} W/K")
    print(f"Températures (int/ext base)  : {res_ch['t_int']:.1f}°C / {res_ch['t_ext_base']:.1f}°C")
    print(f"Puissance de pointe chauffage: {P_max_kW:.1f} kW")
    print(f"Besoin annuel chauffage      : {besoin_total:.0f} kWh/an\n")

    print("=== Hypothèses O&M ===")
    print(f"Méthode O&M                  : {describe_om_method()}")
    print("")

    print("=== Scénario prix (25 ans) ===")
    print(f"Scénario PAC choisi          : {scenario_label}")
    print(f"Code scénario interne        : {scenario}")
    print(f"Scénario électricité         : {scenario_elec}")
    if is_replacement:
        print(f"Scénario énergie actuelle    : {scenario_actuel}")
    print(f"Canton (électricité TTC)     : {canton_prix}")
    print(f"Année début                  : {int(P_elec.index.min())}")
    print(f"Année fin                    : {int(P_elec.index.max())}")
    print(f"Prix électricité début       : {float(P_elec.iloc[0]):.3f} CHF/kWh")
    print(f"Prix électricité fin         : {float(P_elec.iloc[-1]):.3f} CHF/kWh")
    if is_replacement:
        print(f"Prix énergie actuelle début  : {float(P_actuel_heat.iloc[0]):.3f} CHF/kWh")
        print(f"Prix énergie actuelle fin    : {float(P_actuel_heat.iloc[-1]):.3f} CHF/kWh")
    print("")

    if is_replacement:
        print("=== Système actuel (chauffage) ===")
        print(f"Énergie                      : {energie_label}")
        print(f"Rendement utilisé            : {rendement_actuel:.2f}")
        print(f"Conso énergie (entrée)       : {conso_energie_actuelle_kwh:,.0f} kWh/an")
        print(f"CO₂ facteur                  : {co2_factor_actuel_g_kwh:.2f} gCO₂e/kWh")
        print(f"Émissions annuelles chauffage: {emissions_chauffage_actuel_kg:,.0f} kgCO₂e/an")
        print(f"Entretien annuel             : {om_systeme_actuel_chf_per_year:,.0f} CHF/an")
        print(f"Coût chauffage année départ  : {cout_chauffage_actuel_year0:,.0f} CHF/an\n")

    print("=== Climatisation (modèle semi-empirique) ===")
    print(f"Seuil clim (base info)       : {T_BASE_COOL_C:.1f} °C")
    print(f"Heures de clim (info)        : {cooling_hours:,.0f} h/an")
    print(f"Cooling Degree Hours (CDH)   : {cooling_degree_hours:,.0f} °C·h/an")
    print(f"CDH de référence             : {CDH_REF:,.0f} °C·h/an")
    print(f"Surface climatisée           : {surface_climatisee_m2:,.1f} m²")
    print(f"Mode de refroidissement      : {cooling_mode}")
    print(f"q_ref utilisé                : {cool_res['q_ref_kwh_m2a']:.1f} kWh/m².an")
    print(f"Facteur climat               : {cool_res['f_climat']:.2f}")
    print(f"Facteur vitrage              : {cool_res['f_vitrage']:.2f}")
    print(f"Facteur solaire              : {cool_res['f_solaire']:.2f}")
    print(f"Facteur inertie              : {cool_res['f_inertie']:.2f}")
    print(f"Facteur usage                : {cool_res['f_usage']:.2f}")
    print(f"Facteur ventilation nuit     : {cool_res['f_night']:.2f}")
    print(f"Froid utile estimé           : {energie_froid_utile_kwh:,.0f} kWh/an")

    if is_replacement:
        if has_ac and energie_froid_utile_kwh > 0:
            print("Clim existante               : Oui")
            print(f"EER clim actuelle            : {eer_actuel:.2f}")
            print(f"Conso élec clim actuelle     : {conso_elec_clim_actuelle_kwh:,.0f} kWh/an")
            print(f"Coût clim année départ       : {cout_clim_actuelle_year0:,.0f} CHF/an")
        else:
            print("Clim existante               : Non (ou besoin froid ~0)")
    else:
        print(f"PAC utilisée pour le froid   : {'Oui' if want_gshp_cooling and energie_froid_utile_kwh > 0 else 'Non'}")

    print("")

    print("=== PAC géothermique ===")
    print(f"CAPEX brut                   : {capex_brut:,.0f} CHF")
    print(f"Subvention estimée           : {subvention_pac:,.0f} CHF")
    print(f"CAPEX net                    : {capex_net:,.0f} CHF")
    print(f"CAPEX annualisé (brut)       : {gshp.annualized_capex_chf_per_year:,.0f} CHF/an")
    print(f"O&M                          : {cout_pac_chauffage_om:,.0f} CHF/an")
    print(f"SPF                          : {gshp.spf:.2f}")
    print(f"Conso élec chauffage         : {gshp.conso_elec_kwh_per_year:,.0f} kWh/an")
    print(f"Émissions annuelles PAC (ch) : {emissions_chauffage_pac_kg:,.0f} kgCO₂e/an")
    print(f"OPEX chauffage année départ  : {cout_pac_chauffage_year0:,.0f} CHF/an")

    print("=== PAC en mode clim ===")
    if want_gshp_cooling and energie_froid_utile_kwh > 0:
        print("Mode clim PAC                : Oui")
        print(f"Type de froid PAC            : {cooling_mode}")
        print(f"SPF froid PAC                : {eer_gshp_cool:.2f}")
        print(f"Conso élec clim PAC          : {conso_elec_clim_pac_kwh:,.0f} kWh/an")
        print(f"Coût clim PAC année départ   : {cout_clim_pac_year0:,.0f} CHF/an\n")
    else:
        print("Mode clim PAC                : Non (ou besoin froid ~0)\n")

    if is_replacement:
        print("=== Rentabilité (année de départ) ===")
        print(f"Coût annuel actuel total     : {cout_actuel_total_year0:,.0f} CHF/an")
        print(f"Coût annuel PAC total        : {cout_pac_total_year0:,.0f} CHF/an")
        print(f"O&M système actuel           : {om_systeme_actuel_chf_per_year:,.0f} CHF/an")
        print(f"O&M PAC géothermique         : {cout_pac_chauffage_om:,.0f} CHF/an")
        print(f"Économies annuelles          : {economies_annuelles_year0:,.0f} CHF/an")
        print(f"Investissement brut          : {capex_brut:,.0f} CHF")
        print(f"Subvention estimée           : {subvention_pac:,.0f} CHF")
        print(f"Investissement net           : {capex_net:,.0f} CHF")

        print("\n=== Rentabilité (sur 25 ans, prix scénario) ===")
        print(f"Année début                  : {res_pb.get('start_year')}")
        print(f"Année fin                    : {res_pb.get('end_year')}")
        if res_pb.get("payback_years") is None:
            print("Temps de retour actualisé    : Non rentable sur la période (économies ≤ 0)")
        else:
            print(f"Temps de retour actualisé    : {res_pb['payback_years']:.1f} ans")
        print(f"VAN (25 ans, actualisée)     : {res_pb['npv']:,.0f} CHF")

        print("\n=== CO₂ (comparaison : chauffage + clim) ===")
        print(f"Émissions actuelles totales  : {emissions_actuel_total_kg:,.0f} kgCO₂e/an")
        print(f"Émissions PAC totales        : {emissions_pac_total_kg:,.0f} kgCO₂e/an")
        if delta_emissions_kg >= 0:
            print(f"Réduction d'émissions        : {delta_emissions_kg:,.0f} kgCO₂e/an")
        else:
            print(f"Augmentation d'émissions     : {abs(delta_emissions_kg):,.0f} kgCO₂e/an")
    else:
        print("=== Coûts annuels PAC (année de départ) ===")
        print(f"Coût annuel PAC total        : {cout_pac_total_year0:,.0f} CHF/an")
        print(f"Investissement brut          : {capex_brut:,.0f} CHF")
        print(f"Subvention estimée           : {subvention_pac:,.0f} CHF")
        print(f"Investissement net           : {capex_net:,.0f} CHF")

        print("\n=== CO₂ PAC ===")
        print(f"Émissions PAC totales        : {emissions_pac_total_kg:,.0f} kgCO₂e/an")


if __name__ == "__main__":
    run_demo()