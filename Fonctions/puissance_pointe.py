"""Calcul thermique simplifié pour le chauffage :
- Dp (W/K) = Ubat * Sdép + R * Vh   ---> Dp <==> si la différence intérieur–extérieur augmente de 1°C, le bâtiment perd Dp watts de plus.
- P_pointe (kW) = Dp * (T_int - T_ext_base) / 1000
- E_annuelle (kWh/an) = Dp * HDD * 24 / 1000      --> on multiplie le nombre de degrés jours de chauffage par la puissance perdue par degré, puis par 24h/jour et on convertit en kWh.
"""

# Fonctions/puissance_pointe.py
from __future__ import annotations

from .climat import hdd_for_postcode, t_ext_design_for_postcode

# Température intérieure de référence 
T_INT_CONFORT = 20.0  # °C

UBAT_CHOICES = {
    "1": ("Isolation exceptionnelle", 0.30),
    "2": ("Excellente isolation sans ponts thermiques", 0.40),
    "3": ("RT 2000 (2001–2012)", 0.75),
    "4": ("Construction 1990–2000", 0.95),
    "5": ("Construction 1983–1989", 1.15),
    "6": ("Construction 1974–1982", 1.40),
    "7": ("Non isolée et simple vitrage", 1.80),
}

VENTILATION_CHOICES = {
    "1": ("Ventilation standard / inconnue", 0.20),
    "2": ("VMC hygroréglable A/B", 0.14),
}


# 1) Déperditions globales : transmission (Ubat*Sdép) + ventilation (R*Vh)
def dp_deperditions(ubat: float, sdep_m2: float, r: float, vh_m3: float) -> float:
    """Dp (W/K) = Ubat * Sdép + R * Vh"""
    if ubat <= 0:
        raise ValueError("Ubat doit être > 0.")
    if sdep_m2 <= 0:
        raise ValueError("Sdép doit être > 0.")
    if vh_m3 <= 0:
        raise ValueError("Vh doit être > 0.")
    if r <= 0:
        raise ValueError("R doit être > 0.")
    return ubat * sdep_m2 + r * vh_m3


# 2) Puissance de pointe : charge à la température extérieure de dimensionnement
def puissance_pointe_kw(dp_w_per_k: float, t_ext_base: float, t_int: float = T_INT_CONFORT) -> float:
    """Pointe (kW) = Dp * (Tint - Text_base) / 1000"""
    delta_t = t_int - t_ext_base
    if delta_t <= 0:
        raise ValueError("DeltaT doit être > 0 (T_int doit être > T_ext_base).")
    return (dp_w_per_k * delta_t) / 1000.0


# 3) Énergie annuelle : approximation basée sur les HDD (degrés-jours)
def besoin_annuel_chauffage_kwh(dp_w_per_k: float, hdd: float) -> float:
    """Eannuelle (kWh/an) = Dp * HDD * 24 / 1000"""
    if dp_w_per_k <= 0:
        raise ValueError("Dp doit être > 0.")
    if hdd <= 0:
        raise ValueError("HDD doit être > 0.")
    return dp_w_per_k * hdd * 24.0 / 1000.0


#  récupère le climat via le code postal puis applique les 3 calculs.
def calcul_pointe_et_energie_annuelle(
    postcode: str,
    ubat: float,
    sdep_m2: float,
    ventilation_r: float,
    vh_m3: float,
    *,
    t_ext_base: float | None = None,
    t_int: float = T_INT_CONFORT,
) -> dict:
    """
    Retourne : hdd, dp, p_pointe_kw, energie_annuelle_kwh, etc.
    t_ext_base:
      - si None => utilise la valeur t_ext_design du fichier climat (par postcode)
    """
    hdd = hdd_for_postcode(postcode)
    dp = dp_deperditions(ubat, sdep_m2, ventilation_r, vh_m3)

    if t_ext_base is None:
        t_ext_base = t_ext_design_for_postcode(postcode)

    p_kw = puissance_pointe_kw(dp, t_ext_base=t_ext_base, t_int=t_int)
    e_kwh = besoin_annuel_chauffage_kwh(dp, hdd)

    return {
        "postcode": postcode,
        "hdd": hdd,
        "dp_w_per_k": dp,
        "p_pointe_kw": p_kw,
        "energie_annuelle_kwh": e_kwh,
        "t_int": t_int,
        "t_ext_base": t_ext_base,
    }


def ask_positive_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip().replace(",", ".")
        try:
            val = float(raw)
            if val <= 0:
                print("Veuillez entrer une valeur strictement positive.")
                continue
            return val
        except ValueError:
            print("Entrée invalide. Exemple attendu : 250.5")


def ask_choice(prompt: str, choices: dict[str, tuple[str, float]], default: str | None = None) -> float:
    while True:
        print(prompt)
        for key, (label, value) in choices.items():
            suffix = " [défaut]" if default is not None and key == default else ""
            print(f"  {key}) {label}  → {value}{suffix}")

        choice = input("Votre choix : ").strip()
        if choice == "" and default is not None:
            choice = default

        if choice in choices:
            return choices[choice][1]

        print("Choix invalide, recommencez.\n")
