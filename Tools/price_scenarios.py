from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PriceScenarioPaths:
    elec_csv: str = "Data/processed/scenarios_electricity_ttc_by_canton.csv"
    gas_csv: str = "Data/processed/scenarios_gas.csv"
    oil_csv: str = "Data/processed/scenarios_mazout.csv"


def load_electricity_path(
    paths: PriceScenarioPaths,
    canton: str,
    building_type: str,
    scenario: str,
) -> pd.Series:
    """
    Charge la trajectoire de prix de l'électricité (CHF/kWh) pour :
      - un canton
      - un type de bâtiment
      - un scénario

    Retourne une pandas.Series :
      index = year
      values = price_ttc_chf_kwh
    """
    df = pd.read_csv(paths.elec_csv)

    required_cols = {"canton", "building_type", "scenario", "year", "price_ttc_chf_kwh"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le fichier électricité : {sorted(missing)}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    df["canton"] = df["canton"].astype(str).str.strip()
    df["building_type"] = df["building_type"].astype(str).str.strip()
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["price_ttc_chf_kwh"] = pd.to_numeric(df["price_ttc_chf_kwh"], errors="coerce")

    df = df.dropna(subset=["year", "price_ttc_chf_kwh"]).copy()
    df["year"] = df["year"].astype(int)

    g = df[
        (df["canton"] == str(canton).strip()) &
        (df["building_type"] == str(building_type).strip()) &
        (df["scenario"] == str(scenario).strip())
    ].copy()

    if g.empty:
        raise ValueError(
            "Aucune trajectoire électricité trouvée pour "
            f"canton={canton!r}, building_type={building_type!r}, scenario={scenario!r}."
        )

    g = g.sort_values("year")

    return pd.Series(
        g["price_ttc_chf_kwh"].astype(float).values,
        index=g["year"].values,
        name="elec_chf_kwh",
    )


def load_fuel_path(csv_path: str, scenario: str, col_name: str) -> pd.Series:
    """
    Charge une trajectoire de prix combustible (gaz ou mazout) pour un scénario.

    Retourne une pandas.Series :
      index = year
      values = price_chf_kwh
    """
    df = pd.read_csv(csv_path)

    required_cols = {"scenario", "year", "price_chf_kwh"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le fichier combustible : {sorted(missing)}. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["price_chf_kwh"] = pd.to_numeric(df["price_chf_kwh"], errors="coerce")

    df = df.dropna(subset=["year", "price_chf_kwh"]).copy()
    df["year"] = df["year"].astype(int)

    g = df[df["scenario"] == str(scenario).strip()].copy()

    if g.empty:
        raise ValueError(f"Aucune trajectoire combustible trouvée pour scenario={scenario!r}.")

    g = g.sort_values("year")

    return pd.Series(
        g["price_chf_kwh"].astype(float).values,
        index=g["year"].values,
        name=col_name,
    )


def load_gas_path(paths: PriceScenarioPaths, scenario: str) -> pd.Series:
    """
    Charge la trajectoire de prix du gaz (CHF/kWh) pour un scénario.
    """
    return load_fuel_path(paths.gas_csv, scenario, "gas_chf_kwh")


def load_oil_path(paths: PriceScenarioPaths, scenario: str) -> pd.Series:
    """
    Charge la trajectoire de prix du mazout (CHF/kWh) pour un scénario.
    """
    return load_fuel_path(paths.oil_csv, scenario, "oil_chf_kwh")