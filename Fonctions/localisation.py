from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import re

ROOT_DIR = Path(__file__).resolve().parents[1]
POSTCODES_CSV = ROOT_DIR / "Data" / "localites_postcodes" / "ch_postcodes_weighted.csv"


CANTON_ABBR_TO_NAME = {
    "AG": "Argovie",
    "AI": "Appenzell Rhodes-Intérieures",
    "AR": "Appenzell Rhodes-Extérieures",
    "BE": "Berne",
    "BL": "Bâle-Campagne",
    "BS": "Bâle-Ville",
    "FR": "Fribourg",
    "GE": "Genève",
    "GL": "Glaris",
    "GR": "Grisons",
    "JU": "Jura",
    "LU": "Lucerne",
    "NE": "Neuchâtel",
    "NW": "Nidwald",
    "OW": "Obwald",
    "SG": "Saint-Gall",
    "SH": "Schaffhouse",
    "SO": "Soleure",
    "SZ": "Schwyz",
    "TG": "Thurgovie",
    "TI": "Tessin",
    "UR": "Uri",
    "VD": "Vaud",
    "VS": "Valais",
    "ZG": "Zoug",
    "ZH": "Zurich",
}


_POSTCODE_CACHE: pd.DataFrame | None = None

def suggest_nearest_postcode(postcode: str | int) -> dict[str, Any] | None:
    df = load_postcode_table()

    postcode_str = str(postcode).strip()

    if not postcode_str.isdigit():
        return None

    target = int(postcode_str)

    df_tmp = df.copy()
    df_tmp["postcode_int"] = pd.to_numeric(df_tmp["postcode"], errors="coerce")
    df_tmp = df_tmp.dropna(subset=["postcode_int"])

    if df_tmp.empty:
        return None

    df_tmp["distance_postcode"] = (df_tmp["postcode_int"] - target).abs()
    row = df_tmp.sort_values("distance_postcode").iloc[0]

    canton_abbr = str(row["canton"]).strip().upper()
    canton_name = canton_full_name(canton_abbr)

    return {
        "postcode": str(row["postcode"]),
        "e": float(row["e"]),
        "n": float(row["n"]),
        "canton_abbr": canton_abbr,
        "canton_name": canton_name,
        "locality": str(row["locality"]),
    }


def load_postcode_table(csv_path: Path = POSTCODES_CSV) -> pd.DataFrame:
    global _POSTCODE_CACHE

    if _POSTCODE_CACHE is not None:
        return _POSTCODE_CACHE

    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier localités/postcodes introuvable : {csv_path}")

    df = pd.read_csv(csv_path, sep=";", encoding="utf-8-sig")

    required_cols = {"postcode", "e", "n", "canton", "locality"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {csv_path.name} : {sorted(missing)}. "
            f"Colonnes attendues : {sorted(required_cols)}"
        )

    df["postcode"] = df["postcode"].astype(str).str.strip()
    df["canton"] = df["canton"].astype(str).str.strip().str.upper()
    df["locality"] = df["locality"].astype(str).str.strip()
    df["e"] = pd.to_numeric(df["e"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")

    df = df.dropna(subset=["postcode", "e", "n", "canton"])

    _POSTCODE_CACHE = df

    return _POSTCODE_CACHE


def canton_full_name(canton_abbr: str) -> str:
    abbr = str(canton_abbr).strip().upper()

    if abbr not in CANTON_ABBR_TO_NAME:
        raise ValueError(f"Canton abrégé inconnu : {abbr}")

    return CANTON_ABBR_TO_NAME[abbr]


def postcode_info(postcode: str | int) -> dict[str, Any]:
    df = load_postcode_table()

    postcode_str = str(postcode).strip()

    # Cas 1 : format invalide
    if not re.fullmatch(r"\d{4}", postcode_str):
        raise ValueError(
            "Code postal invalide. Veuillez vérifier votre saisie : "
            "un code postal suisse doit contenir exactement 4 chiffres."
        )

    rows = df[df["postcode"] == postcode_str]

    # Cas 2 : format correct, mais absent de la base
    if rows.empty:
        suggestion = suggest_nearest_postcode(postcode_str)

        if suggestion is not None:
            raise ValueError(
                f"Ce code postal n'est pas dans notre base de données. "
                f"Essayez avec une localité proche, par exemple "
                f"{suggestion['postcode']} — {suggestion['locality']} "
                f"({suggestion['canton_abbr']})."
            )

        raise ValueError(
            "Ce code postal n'est pas dans notre base de données. "
            "Essayez avec une localité proche de chez vous."
        )

    row = rows.iloc[0]

    canton_abbr = str(row["canton"]).strip().upper()
    canton_name = canton_full_name(canton_abbr)

    return {
        "postcode": postcode_str,
        "e": float(row["e"]),
        "n": float(row["n"]),
        "canton_abbr": canton_abbr,
        "canton_name": canton_name,
        "locality": str(row["locality"]),
    }