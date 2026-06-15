"""
commande: 
cd "C:\Travail Master\CodePAC" ; python Tools\transform_elcom_prices.py Data\elcom_raw.csv --out_dir Data\out

Transforme un CSV ElCom "brut" (tarifs électricité) en jeux de données propres, prêts à l'emploi
pour un modèle techno-éco (ex. PAC géothermie) + Monte Carlo.

Sorties principales :
1) electricity_clean_long.csv
   - format "long" : 1 ligne = (canton, année, profil, type_batiment, produit)
   - prix variables en CHF/kWh (hors TVA + option TTC)
   - décomposition : réseau, énergie, taxes/fees, KEV, comptage variable
   - part fixe : tarif de comptage annuel CHF/an (si dispo)

2) electricity_summary_by_canton_year_building.csv
   - agrégé par (canton, année, type_batiment, produit)
   - médiane / P10 / P90 / moyenne des prix variables + stats sur la part fixe

 Note importante:
Le fichier ElCom que tu montres contient des "Consumption profiles of typical households"
(C*, H*). Donc il couvre bien appartements/maisons (ménages).
Pour "grand bâtiment (école)" / tertiaire, il te faudra souvent une source tarifaire pro/MT
ou un dataset ElCom/DSO dédié aux clients non-ménages. Le script gère ça via un mapping modifiable
(et met "unknown" si un profil n’est pas classé).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Configuration / mapping
# -----------------------------

DEFAULT_PROFILE_TO_BUILDING = {
    # Profils "C*" : souvent ménages plus petits (plutôt appartements)
    # Profils "H*" : souvent maisons (plus gros volumes / consommation)
    # Ajuste librement si ton contexte est différent.
    r"^C\d+$": "appartement",
    r"^H\d+$": "maison_individuelle",

    # Si ton CSV contient d'autres profils (ex: "G1", "S2", etc.), ajoute-les ici.
    # r"^S\d+$": "immeuble",
    # r"^G\d+$": "grand_batiment",
}

# TVA Suisse (à adapter si besoin). 8.1% depuis 2024 (arrondi).
DEFAULT_VAT_RATE = 0.081


# -----------------------------
# Helpers
# -----------------------------

def parse_number(x: Any) -> float:
    """
    Convertit les nombres qui peuvent être:
    - des floats/ints
    - des strings avec '.' ou ',' (rare)
    - '-' ou vide => NaN
    """
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "" or s == "-" or s.lower() == "nan":
        return float("nan")
    # Remplace une virgule décimale éventuelle par un point
    s = s.replace(",", ".")
    # Supprime les espaces
    s = re.sub(r"\s+", "", s)
    try:
        return float(s)
    except ValueError:
        return float("nan")


def rp_per_kwh_to_chf_per_kwh(rp_per_kwh: float) -> float:
    """1 CHF = 100 Rp."""
    if math.isnan(rp_per_kwh):
        return float("nan")
    return rp_per_kwh / 100.0


def classify_building(consumption_profile: str,
                      regex_map: Dict[str, str]) -> str:
    """Retourne un type_batiment à partir d'un profil (C3, H5, etc.) selon des regex."""
    if consumption_profile is None:
        return "unknown"
    p = str(consumption_profile).strip()
    for pattern, building in regex_map.items():
        if re.match(pattern, p):
            return building
    return "unknown"


def quantile_safe(series: pd.Series, q: float) -> float:
    """Quantile robuste (ignore NaN)."""
    s = series.dropna()
    if s.empty:
        return float("nan")
    return float(s.quantile(q))


@dataclass
class TransformConfig:
    vat_rate: float = DEFAULT_VAT_RATE
    add_vat_columns: bool = True
    profile_regex_map: Dict[str, str] = None

    @staticmethod
    def from_json(path: Optional[Path]) -> "TransformConfig":
        cfg = TransformConfig(profile_regex_map=DEFAULT_PROFILE_TO_BUILDING.copy())
        if path is None:
            return cfg
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg.vat_rate = float(data.get("vat_rate", cfg.vat_rate))
        cfg.add_vat_columns = bool(data.get("add_vat_columns", cfg.add_vat_columns))
        profile_map = data.get("profile_regex_map")
        if isinstance(profile_map, dict):
            cfg.profile_regex_map = profile_map
        return cfg


# -----------------------------
# Core transformation
# -----------------------------

def transform_elcom_csv(input_csv: Path,
                        out_clean_long: Path,
                        out_summary: Path,
                        config: TransformConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Lit le CSV ElCom brut et produit:
    - df_long: données propres "long"
    - df_summary: agrégation par canton/année/type_batiment/produit
    """
    df = pd.read_csv(input_csv)

    # Normalisation des noms de colonnes attendus (selon ton exemple)
    # On rend tolérant aux variantes mineures d'intitulés.
    col_map_candidates = {
        "Canton": "canton",
        "Consumption profiles of typical households": "profile",
        "Period": "year",
        "Product": "product",
        "Total excl. VAT (Rp./kWH)": "total_ex_vat_rp_kwh",
        "Grid usage (Rp./kWH)": "grid_rp_kwh",
        "Energy supply costs (Rp./kWH)": "energy_supply_rp_kwh",
        "Community fees (Rp./kWH)": "community_fees_rp_kwh",
        "Feed-in remuneration at cost (KEV) (Rp./kWH)": "kev_rp_kwh",
        "Metering rate (Rp./kWH)": "metering_var_rp_kwh",
        "Tariff for metering in CHF per year": "metering_fixed_chf_year",
    }

    # Renommer uniquement les colonnes présentes
    rename_map = {k: v for k, v in col_map_candidates.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    required = ["canton", "profile", "year", "product", "total_ex_vat_rp_kwh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV: {missing}\n"
            f"Colonnes disponibles: {list(df.columns)}"
        )

    # Parsing numérique
    numeric_cols = [
        "total_ex_vat_rp_kwh",
        "grid_rp_kwh",
        "energy_supply_rp_kwh",
        "community_fees_rp_kwh",
        "kev_rp_kwh",
        "metering_var_rp_kwh",
        "metering_fixed_chf_year",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].map(parse_number)

    # Types
    df["year"] = df["year"].astype(str).str.strip()
    # Certains fichiers ont "2026" en int/float => on force int si possible
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    df["canton"] = df["canton"].astype(str).str.strip()
    df["profile"] = df["profile"].astype(str).str.strip()
    df["product"] = df["product"].astype(str).str.strip()

    # Classification bâtiment selon profil
    df["building_type"] = df["profile"].apply(lambda p: classify_building(p, config.profile_regex_map))

    # Conversion Rp/kWh -> CHF/kWh
    df["total_ex_vat_chf_kwh"] = df["total_ex_vat_rp_kwh"].apply(rp_per_kwh_to_chf_per_kwh)

    # Conversion des composantes si présentes
    comp_cols_rp = [
        ("grid_rp_kwh", "grid_chf_kwh"),
        ("energy_supply_rp_kwh", "energy_supply_chf_kwh"),
        ("community_fees_rp_kwh", "community_fees_chf_kwh"),
        ("kev_rp_kwh", "kev_chf_kwh"),
        ("metering_var_rp_kwh", "metering_var_chf_kwh"),
    ]
    for src, dst in comp_cols_rp:
        if src in df.columns:
            df[dst] = df[src].apply(rp_per_kwh_to_chf_per_kwh)
        else:
            df[dst] = np.nan

    # TVA (optionnel)
    if config.add_vat_columns:
        df["total_incl_vat_chf_kwh"] = df["total_ex_vat_chf_kwh"] * (1.0 + config.vat_rate)
        # Tu peux aussi appliquer TVA sur composantes si tu veux les afficher TTC
        for _, dst in comp_cols_rp:
            df[f"{dst}_incl_vat"] = df[dst] * (1.0 + config.vat_rate)

    # Colonnes propres finalisées (format long)
    keep_cols = [
        "canton", "year", "profile", "building_type", "product",
        "total_ex_vat_chf_kwh",
        "grid_chf_kwh", "energy_supply_chf_kwh", "community_fees_chf_kwh", "kev_chf_kwh", "metering_var_chf_kwh",
        "metering_fixed_chf_year",
    ]
    if config.add_vat_columns:
        keep_cols += ["total_incl_vat_chf_kwh"] + [f"{dst}_incl_vat" for _, dst in comp_cols_rp]

    df_long = df[keep_cols].copy()

    # Nettoyage léger : supprime lignes sans année ou prix total
    df_long = df_long.dropna(subset=["year", "total_ex_vat_chf_kwh"]).reset_index(drop=True)

    # Agrégation “prête modèle” par (canton, année, type_batiment, produit)
    group_cols = ["canton", "year", "building_type", "product"]

    def agg_block(g: pd.DataFrame) -> pd.Series:
        s = pd.Series(dtype="float64")
        price = g["total_ex_vat_chf_kwh"]
        s["n"] = len(g)
        s["price_median_chf_kwh"] = float(price.median(skipna=True))
        s["price_p10_chf_kwh"] = quantile_safe(price, 0.10)
        s["price_p90_chf_kwh"] = quantile_safe(price, 0.90)
        s["price_mean_chf_kwh"] = float(price.mean(skipna=True))

        # Part fixe : certains cantons/années peuvent avoir NaN si '-' dans la source
        fixed = g["metering_fixed_chf_year"]
        s["fixed_median_chf_year"] = float(fixed.median(skipna=True)) if fixed.notna().any() else float("nan")
        s["fixed_p10_chf_year"] = quantile_safe(fixed, 0.10)
        s["fixed_p90_chf_year"] = quantile_safe(fixed, 0.90)

        # Décomposition (moyennes) utile si tu veux simuler différemment les composantes
        for col in ["grid_chf_kwh", "energy_supply_chf_kwh", "community_fees_chf_kwh", "kev_chf_kwh", "metering_var_chf_kwh"]:
            s[f"{col}_mean"] = float(g[col].mean(skipna=True)) if col in g.columns else float("nan")

        if config.add_vat_columns:
            price_ttc = g["total_incl_vat_chf_kwh"]
            s["price_ttc_median_chf_kwh"] = float(price_ttc.median(skipna=True))
            s["price_ttc_p10_chf_kwh"] = quantile_safe(price_ttc, 0.10)
            s["price_ttc_p90_chf_kwh"] = quantile_safe(price_ttc, 0.90)

        return s

    df_summary = df_long.groupby(group_cols, dropna=False).apply(agg_block).reset_index()

    # Écriture
    out_clean_long.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(out_clean_long, index=False)
    df_summary.to_csv(out_summary, index=False)

    return df_long, df_summary


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Nettoie un CSV ElCom brut et produit des CSV prêts à l'emploi par type de bâtiment."
    )
    parser.add_argument("input_csv", type=str, help="Chemin vers le CSV brut ElCom")
    parser.add_argument("--out_dir", type=str, default="out", help="Dossier de sortie")
    parser.add_argument("--config", type=str, default=None,
                        help="Chemin JSON optionnel pour définir vat_rate / mapping profils->bâtiments")
    parser.add_argument("--no_vat", action="store_true", help="Ne pas ajouter de colonnes TTC (TVA)")

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    cfg = TransformConfig.from_json(Path(args.config) if args.config else None)
    if args.no_vat:
        cfg.add_vat_columns = False

    out_clean_long = out_dir / "electricity_clean_long.csv"
    out_summary = out_dir / "electricity_summary_by_canton_year_building.csv"

    df_long, df_summary = transform_elcom_csv(
        input_csv=input_csv,
        out_clean_long=out_clean_long,
        out_summary=out_summary,
        config=cfg
    )

    print(f"OK - écrit:\n- {out_clean_long}\n- {out_summary}")
    print("\nAperçu (clean_long):")
    print(df_long.head(5).to_string(index=False))
    print("\nAperçu (summary):")
    print(df_summary.head(5).to_string(index=False))

    # Alerte sur profils inconnus
    unknown_profiles = (df_long[df_long["building_type"] == "unknown"]["profile"]
                        .value_counts().head(20))
    if not unknown_profiles.empty:
        print("\n Profils non classés (building_type='unknown') — à mapper via --config :")
        print(unknown_profiles.to_string())


if __name__ == "__main__":
    main()
