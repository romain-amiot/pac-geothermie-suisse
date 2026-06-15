from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# ==============================
# PARAMÈTRES
# ==============================

INPUT_DIR = Path(r"C:\Travail Master\CodePAC\Data\Climat_Villes_Hourly")
OUTPUT_YEARLY = Path(r"C:\Travail Master\CodePAC\Data\climate_indicators_yearly.csv")
OUTPUT_STATION = Path(r"C:\Travail Master\CodePAC\Data\climate_indicators_station_summary.csv")

TEMP_COL = "tre200h0"          # température horaire principale
STATION_COL = "station_abbr"
TIME_COL = "reference_timestamp"

CDH_BASES = [24.0, 26.0]
HOT_THRESHOLDS = [24.0, 26.0, 28.0, 30.0]
NIGHT_COOL_THRESHOLDS = [18.0, 20.0]

SUMMER_MONTHS = [6, 7, 8]
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4, 5, 6]


# ==============================
# HELPERS
# ==============================

def parse_station_name_from_file(filepath: Path) -> str:
    """
    Extrait un nom de station lisible à partir du nom de fichier, ex:
    Geneve_hourly_1980.csv -> Geneve
    """
    name = filepath.stem
    if "_hourly_" in name:
        return name.split("_hourly_")[0]
    return name


def load_one_file(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep=";",
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {STATION_COL, TIME_COL, TEMP_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {filepath.name}: {sorted(missing)}")

    df = df[[STATION_COL, TIME_COL, TEMP_COL]].copy()
    df["station_name"] = parse_station_name_from_file(filepath)

    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL],
        format="%d.%m.%Y %H:%M",
        errors="coerce",
    )

    df[TEMP_COL] = pd.to_numeric(df[TEMP_COL], errors="coerce")

    df = df.dropna(subset=[TIME_COL, TEMP_COL]).copy()

    df["year"] = df[TIME_COL].dt.year
    df["month"] = df[TIME_COL].dt.month
    df["hour"] = df[TIME_COL].dt.hour
    df["date"] = df[TIME_COL].dt.date

    return df


def cooling_degree_hours(series_temp: pd.Series, base_temp: float) -> float:
    return float(np.maximum(series_temp - base_temp, 0.0).sum())


def hours_above(series_temp: pd.Series, threshold: float) -> int:
    return int((series_temp > threshold).sum())


def count_hot_nights(night_df: pd.DataFrame, threshold: float) -> int:
    """
    Nombre de nuits dont la température minimale nocturne reste > threshold.
    """
    if night_df.empty:
        return 0
    min_by_night = night_df.groupby("date")[TEMP_COL].min()
    return int((min_by_night > threshold).sum())


def build_yearly_indicators(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = df.groupby([STATION_COL, "station_name", "year"], dropna=False)

    for (station_abbr, station_name, year), g in grouped:
        g = g.sort_values(TIME_COL).copy()
        temps = g[TEMP_COL]

        summer = g[g["month"].isin(SUMMER_MONTHS)]
        night = summer[summer["hour"].isin(NIGHT_HOURS)]

        row = {
            "station_abbr": station_abbr,
            "station_name": station_name,
            "year": int(year),
            "n_hours": len(g),
            "t_mean_annual_c": round(float(temps.mean()), 3),
            "t_min_annual_c": round(float(temps.min()), 3),
            "t_max_annual_c": round(float(temps.max()), 3),
            "t_mean_summer_c": round(float(summer[TEMP_COL].mean()), 3) if not summer.empty else None,
            "t_p95_summer_c": round(float(summer[TEMP_COL].quantile(0.95)), 3) if not summer.empty else None,
            "t_p99_summer_c": round(float(summer[TEMP_COL].quantile(0.99)), 3) if not summer.empty else None,
        }

        for base in CDH_BASES:
            row[f"cdh_{int(base)}"] = round(cooling_degree_hours(summer[TEMP_COL], base), 3) if not summer.empty else 0.0

        for thr in HOT_THRESHOLDS:
            row[f"hours_above_{int(thr)}"] = hours_above(summer[TEMP_COL], thr) if not summer.empty else 0

        for thr in NIGHT_COOL_THRESHOLDS:
            if not night.empty:
                row[f"night_hours_below_{int(thr)}"] = int((night[TEMP_COL] < thr).sum())
                row[f"hot_nights_min_gt_{int(thr)}"] = count_hot_nights(night, thr)
            else:
                row[f"night_hours_below_{int(thr)}"] = 0
                row[f"hot_nights_min_gt_{int(thr)}"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    valid = (~series.isna()) & (~weights.isna())
    if valid.sum() == 0:
        return np.nan
    return float(np.average(series[valid], weights=weights[valid]))


def build_station_summary(yearly_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    grouped = yearly_df.groupby(["station_abbr", "station_name"], dropna=False)

    for (station_abbr, station_name), g in grouped:
        g = g.sort_values("year").copy()
        weights = g["n_hours"].fillna(0)

        row = {
            "station_abbr": station_abbr,
            "station_name": station_name,
            "year_start": int(g["year"].min()),
            "year_end": int(g["year"].max()),
            "n_years": int(g["year"].nunique()),
            "n_hours_total": int(g["n_hours"].sum()),
            "t_mean_annual_c": round(weighted_mean(g["t_mean_annual_c"], weights), 3),
            "t_mean_summer_c": round(weighted_mean(g["t_mean_summer_c"], weights), 3),
            "t_p95_summer_c_mean": round(float(g["t_p95_summer_c"].mean()), 3),
            "t_p99_summer_c_mean": round(float(g["t_p99_summer_c"].mean()), 3),
        }

        for base in CDH_BASES:
            col = f"cdh_{int(base)}"
            row[col] = round(float(g[col].mean()), 3)

        for thr in HOT_THRESHOLDS:
            col = f"hours_above_{int(thr)}"
            row[col] = round(float(g[col].mean()), 1)

        for thr in NIGHT_COOL_THRESHOLDS:
            col1 = f"night_hours_below_{int(thr)}"
            col2 = f"hot_nights_min_gt_{int(thr)}"
            row[col1] = round(float(g[col1].mean()), 1)
            row[col2] = round(float(g[col2].mean()), 1)

        rows.append(row)

    return pd.DataFrame(rows)


# ==============================
# MAIN
# ==============================

def main():
    files = sorted(INPUT_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {INPUT_DIR}")

    all_parts = []
    for f in files:
        try:
            part = load_one_file(f)
            all_parts.append(part)
            print(f"Lu: {f.name} ({len(part)} lignes utiles)")
        except Exception as e:
            print(f"Erreur sur {f.name}: {e}")

    if not all_parts:
        raise RuntimeError("Aucune donnée exploitable n'a pu être chargée.")

    full_df = pd.concat(all_parts, ignore_index=True)

    # Déduplication basique
    full_df = full_df.drop_duplicates(subset=[STATION_COL, TIME_COL]).copy()

    yearly_df = build_yearly_indicators(full_df)
    station_df = build_station_summary(yearly_df)

    yearly_df.to_csv(OUTPUT_YEARLY, sep=";", index=False, encoding="utf-8-sig")
    station_df.to_csv(OUTPUT_STATION, sep=";", index=False, encoding="utf-8-sig")

    print()
    print(f"Fichier annuel créé : {OUTPUT_YEARLY}")
    print(f"Fichier station créé : {OUTPUT_STATION}")
    print()
    print(f"Stations traitées : {station_df['station_abbr'].nunique()}")
    print(f"Années-stations traitées : {len(yearly_df)}")


if __name__ == "__main__":
    main()