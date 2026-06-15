from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PARAMÈTRES
# ============================================================

INPUT_CSV = Path(r"C:\Travail Master\CodePAC\Data\climate_indicators_yearly.csv")

OUTPUT_STATION_REF = Path(r"C:\Travail Master\CodePAC\Data\climate_station_reference.csv")
OUTPUT_STATION_REF_DETAIL = Path(r"C:\Travail Master\CodePAC\Data\climate_station_reference_detail.csv")

MIN_VALID_HOURS = 8000

# période récente recommandée pour l'outil
YEAR_START = 2015
YEAR_END = 2024

# station de référence pour normaliser le climat
REFERENCE_STATION_NAME = "Lausanne"

# seuils de classes climatiques basés sur CDH26 moyen
def climate_class_from_cdh26(cdh26: float) -> str:
    if pd.isna(cdh26):
        return "tempere"
    if cdh26 < 50:
        return "froid"
    if cdh26 < 400:
        return "tempere"
    if cdh26 < 900:
        return "chaud"
    return "tres_chaud"


# ============================================================
# HELPERS
# ============================================================

def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    valid = (~series.isna()) & (~weights.isna())
    if valid.sum() == 0:
        return np.nan
    return float(np.average(series[valid], weights=weights[valid]))


def safe_round(x, ndigits=3):
    if pd.isna(x):
        return None
    return round(float(x), ndigits)


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Fichier introuvable : {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, sep=";", encoding="utf-8-sig")

    required_cols = {
        "station_abbr",
        "station_name",
        "year",
        "n_hours",
        "t_mean_annual_c",
        "t_mean_summer_c",
        "t_p95_summer_c",
        "t_p99_summer_c",
        "cdh_24",
        "cdh_26",
        "hours_above_24",
        "hours_above_26",
        "hours_above_28",
        "hours_above_30",
        "night_hours_below_18",
        "night_hours_below_20",
        "hot_nights_min_gt_18",
        "hot_nights_min_gt_20",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

    # conversion types
    numeric_cols = [
        "year", "n_hours",
        "t_mean_annual_c", "t_mean_summer_c", "t_p95_summer_c", "t_p99_summer_c",
        "cdh_24", "cdh_26",
        "hours_above_24", "hours_above_26", "hours_above_28", "hours_above_30",
        "night_hours_below_18", "night_hours_below_20",
        "hot_nights_min_gt_18", "hot_nights_min_gt_20",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["station_abbr"] = df["station_abbr"].astype(str).str.strip()
    df["station_name"] = df["station_name"].astype(str).str.strip()

    # --------------------------------------------------------
    # 1) Filtre années valides et période récente
    # --------------------------------------------------------
    df_valid = df[
        (df["n_hours"] >= MIN_VALID_HOURS) &
        (df["year"] >= YEAR_START) &
        (df["year"] <= YEAR_END)
    ].copy()

    if df_valid.empty:
        raise ValueError("Aucune donnée valide après filtrage.")

    # --------------------------------------------------------
    # 2) Synthèse détaillée par station
    # --------------------------------------------------------
    rows = []

    grouped = df_valid.groupby(["station_abbr", "station_name"], dropna=False)

    for (station_abbr, station_name), g in grouped:
        g = g.sort_values("year").copy()
        weights = g["n_hours"].fillna(0)

        row = {
            "station_abbr": station_abbr,
            "station_name": station_name,
            "year_start_used": int(g["year"].min()),
            "year_end_used": int(g["year"].max()),
            "n_years_used": int(g["year"].nunique()),
            "n_hours_total_used": int(g["n_hours"].sum()),

            "t_mean_annual_c_ref": safe_round(weighted_mean(g["t_mean_annual_c"], weights), 3),
            "t_mean_summer_c_ref": safe_round(weighted_mean(g["t_mean_summer_c"], weights), 3),
            "t_p95_summer_c_ref": safe_round(g["t_p95_summer_c"].mean(), 3),
            "t_p99_summer_c_ref": safe_round(g["t_p99_summer_c"].mean(), 3),

            "cdh_24_ref": safe_round(g["cdh_24"].mean(), 3),
            "cdh_26_ref": safe_round(g["cdh_26"].mean(), 3),

            "hours_above_24_ref": safe_round(g["hours_above_24"].mean(), 1),
            "hours_above_26_ref": safe_round(g["hours_above_26"].mean(), 1),
            "hours_above_28_ref": safe_round(g["hours_above_28"].mean(), 1),
            "hours_above_30_ref": safe_round(g["hours_above_30"].mean(), 1),

            "night_hours_below_18_ref": safe_round(g["night_hours_below_18"].mean(), 1),
            "night_hours_below_20_ref": safe_round(g["night_hours_below_20"].mean(), 1),

            "hot_nights_min_gt_18_ref": safe_round(g["hot_nights_min_gt_18"].mean(), 1),
            "hot_nights_min_gt_20_ref": safe_round(g["hot_nights_min_gt_20"].mean(), 1),
        }

        row["climate_class"] = climate_class_from_cdh26(row["cdh_26_ref"])
        rows.append(row)

    detail_df = pd.DataFrame(rows).sort_values(["station_name", "station_abbr"]).reset_index(drop=True)

    # --------------------------------------------------------
    # 3) Calcul des facteurs climatiques relatifs
    # --------------------------------------------------------
    ref_match = detail_df[
        detail_df["station_name"].str.lower() == REFERENCE_STATION_NAME.lower()
    ].copy()

    if ref_match.empty:
        raise ValueError(
            f"Station de référence '{REFERENCE_STATION_NAME}' introuvable dans les données."
        )

    cdh26_ref_station = float(ref_match["cdh_26_ref"].iloc[0])

    if cdh26_ref_station <= 0:
        raise ValueError("Le CDH26 de la station de référence doit être > 0.")

    detail_df["f_climat_cdh26_vs_ref"] = (
        detail_df["cdh_26_ref"] / cdh26_ref_station
    ).clip(lower=0.20, upper=3.00)

    # disponibilité relative pour ventilation nocturne
    night20_ref_station = float(ref_match["night_hours_below_20_ref"].iloc[0])

    if night20_ref_station > 0:
        detail_df["f_night_potential_vs_ref"] = (
            detail_df["night_hours_below_20_ref"] / night20_ref_station
        ).clip(lower=0.30, upper=1.50)
    else:
        detail_df["f_night_potential_vs_ref"] = 1.0

    # pénalité liée aux nuits chaudes
    hot20_ref_station = float(ref_match["hot_nights_min_gt_20_ref"].iloc[0])

    if hot20_ref_station > 0:
        detail_df["f_hot_nights_vs_ref"] = (
            detail_df["hot_nights_min_gt_20_ref"] / hot20_ref_station
        ).clip(lower=0.30, upper=3.00)
    else:
        detail_df["f_hot_nights_vs_ref"] = 1.0

    # arrondi
    for col in ["f_climat_cdh26_vs_ref", "f_night_potential_vs_ref", "f_hot_nights_vs_ref"]:
        detail_df[col] = detail_df[col].round(3)

    # --------------------------------------------------------
    # 4) CSV final simplifié pour l'application
    # --------------------------------------------------------
    app_df = detail_df[[
        "station_abbr",
        "station_name",
        "climate_class",
        "cdh_26_ref",
        "hours_above_26_ref",
        "hours_above_28_ref",
        "night_hours_below_20_ref",
        "hot_nights_min_gt_20_ref",
        "f_climat_cdh26_vs_ref",
        "f_night_potential_vs_ref",
        "f_hot_nights_vs_ref",
    ]].copy()

    # export
    detail_df.to_csv(OUTPUT_STATION_REF_DETAIL, sep=";", index=False, encoding="utf-8-sig")
    app_df.to_csv(OUTPUT_STATION_REF, sep=";", index=False, encoding="utf-8-sig")

    print(f"Créé : {OUTPUT_STATION_REF}")
    print(f"Créé : {OUTPUT_STATION_REF_DETAIL}")
    print()
    print(f"Station de référence : {REFERENCE_STATION_NAME}")
    print(f"CDH26 station de référence : {cdh26_ref_station:.3f}")
    print()
    print("Aperçu :")
    print(app_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()