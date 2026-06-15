"""
Commande d'exécution :

cd "C:\Travail Master\CodePAC"
py Tools\build_price_scenarios.py `
  --elec_csv "Data\Prix_electricite\Prix_electricite_canton.csv" `
  --gas_csv "Data\Prix_gaz_Energie360.csv" `
  --oil_csv "Data\Prix_mazout_Midland.csv" `
  --out_dir "Data\processed"
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ScenarioConfig:
    horizon_years: int = 25

    # Fenêtres d'historique
    window_lt_elec: int = 15
    window_lt_gas: int = 25
    window_lt_oil: int = 25

    # Lissage
    smooth_years: int = 3

    # Croissances minimales structurelles
    min_growth_elec_neutral: float = 0.015   # +1.5%/an
    min_growth_gas_neutral: float = 0.030    # +3.0%/an
    min_growth_oil_neutral: float = 0.025    # +2.5%/an

    # Largeur des scénarios autour du neutre
    spread_elec: float = 0.010
    spread_gas: float = 0.015
    spread_oil: float = 0.015

    # Bornes basses absolues pour l'optimiste
    floor_growth_elec_optimistic: float = 0.005
    floor_growth_gas_optimistic: float = 0.015
    floor_growth_oil_optimistic: float = 0.010

    # Caps annuels
    cap_elec_down: float = -0.10
    cap_elec_up: float = 0.10
    cap_fuel_down: float = -0.05
    cap_fuel_up: float = 0.10

    # Filtre produit électricité
    electricity_product_filter: str = "Standard product"


SCENARIOS = ("optimistic", "neutral", "pessimistic")


# ============================================================
# HELPERS
# ============================================================

def _read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def _find_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for cand in candidates:
        for cl in df.columns:
            if cand.lower() in cl.lower():
                return cl
    return None


def _to_annual_series(df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce", dayfirst=True)
    d = d.dropna(subset=[date_col])
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col])
    d = d.set_index(date_col).sort_index()

    annual = d[value_col].resample("YE").mean()
    annual.index = annual.index.year
    annual = annual.astype(float)
    return annual


def _cap_growth(g: float, cap_down: float, cap_up: float) -> float:
    return float(np.clip(g, cap_down, cap_up))


def _smooth_annual(series: pd.Series, smooth_years: int) -> pd.Series:
    s = series.dropna().sort_index().astype(float)
    if len(s) == 0:
        return s
    return s.rolling(window=smooth_years, min_periods=1).mean()


def _compute_cagr(prices: pd.Series) -> float:
    s = prices.dropna().sort_index().astype(float)
    s = s[s > 0]
    if len(s) < 2:
        raise ValueError("Pas assez de points pour calculer un CAGR.")
    y0 = int(s.index.min())
    y1 = int(s.index.max())
    n = y1 - y0
    if n <= 0:
        raise ValueError("Période invalide pour calcul CAGR.")
    return float((s.iloc[-1] / s.iloc[0]) ** (1.0 / n) - 1.0)


def _build_constant_growth_path(
    P0: float,
    growth: float,
    horizon: int,
    cap_down: float,
    cap_up: float,
) -> np.ndarray:
    if not np.isfinite(P0) or P0 <= 0:
        raise ValueError("P0 invalide")

    g = _cap_growth(growth, cap_down=cap_down, cap_up=cap_up)

    P = np.zeros(horizon + 1, dtype=float)
    P[0] = P0
    for t in range(horizon):
        P[t + 1] = P[t] * (1.0 + g)
    return P


# ============================================================
# LOADERS
# ============================================================

def load_electricity_canton_prices(path_csv: Path) -> pd.DataFrame:
    """
    Attend au minimum:
      canton, year, building_type, product, price_ttc_median_chf_kwh
    """
    df = _read_csv_flexible(path_csv)

    canton_col = _find_col(df, ("canton",))
    year_col = _find_col(df, ("year", "annee"))
    building_type_col = _find_col(df, ("building_type",))
    product_col = _find_col(df, ("product",))
    price_col = _find_col(df, ("price_ttc_median_chf_kwh",))

    required = [canton_col, year_col, building_type_col, product_col, price_col]
    if any(c is None for c in required):
        raise ValueError(
            "Colonnes requises non trouvées pour l'électricité. "
            f"Colonnes disponibles: {list(df.columns)}"
        )

    out = df[[canton_col, year_col, building_type_col, product_col, price_col]].copy()
    out.columns = ["canton", "year", "building_type", "product", "price_ttc_chf_kwh"]

    out["canton"] = out["canton"].astype(str).str.strip()
    out["building_type"] = out["building_type"].astype(str).str.strip()
    out["product"] = out["product"].astype(str).str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["price_ttc_chf_kwh"] = pd.to_numeric(out["price_ttc_chf_kwh"], errors="coerce")

    out = out.dropna(subset=["canton", "year", "building_type", "product", "price_ttc_chf_kwh"])
    out["year"] = out["year"].astype(int)

    return out


def load_gas_prices(path_csv: Path) -> pd.Series:
    """
    Format attendu :
      2000,0133676092544; 3,757620271759091

    => année_float ; prix_ct_kwh
    """
    df = pd.read_csv(
        path_csv,
        sep=";",
        header=None,
        names=["year_float", "price_ct_kwh"],
        decimal=",",
        engine="python",
    )

    df["year_float"] = pd.to_numeric(df["year_float"], errors="coerce")
    df["price_ct_kwh"] = pd.to_numeric(df["price_ct_kwh"], errors="coerce")
    df = df.dropna(subset=["year_float", "price_ct_kwh"])

    df["year"] = df["year_float"].astype(int)

    annual_ct = df.groupby("year")["price_ct_kwh"].mean().sort_index()
    annual_chf = annual_ct / 100.0

    return annual_chf


def load_oil_prices_midland(path_csv: Path) -> pd.Series:
    """
    Attend typiquement:
      date,price

    Si le prix est en CHF/100L, conversion approx:
      CHF/kWh = (CHF/100L) / 1000
    """
    df = _read_csv_flexible(path_csv)

    date_col = _find_col(df, ("date", "datum"))
    val_col = _find_col(df, ("price", "prix", "preis", "chf", "100l", "litre", "l"))

    if date_col is None or val_col is None:
        raise ValueError(
            "Impossible de détecter date/prix dans le CSV mazout. "
            f"Colonnes disponibles: {list(df.columns)}"
        )

    annual_raw = _to_annual_series(df, date_col=date_col, value_col=val_col)

    med = float(np.median(annual_raw.values)) if len(annual_raw) else float("nan")

    if np.isfinite(med) and med > 1.0:
        annual_chf_kwh = annual_raw / 1000.0
    else:
        annual_chf_kwh = annual_raw

    return annual_chf_kwh


# ============================================================
# SCÉNARIOS STRUCTURELS
# ============================================================

def build_structural_growth_rates(
    prices_annual: pd.Series,
    *,
    window_lt: int,
    smooth_years: int,
    neutral_floor: float,
    spread: float,
    optimistic_floor: float,
) -> Dict[str, float]:
    """
    Construit des taux de croissance annuels cohérents à partir d'une série historique.
    """
    s = prices_annual.dropna().sort_index()

    if len(s) < 2:
        raise ValueError("Pas assez de points pour construire des scénarios.")

    if len(s) > window_lt:
        s = s.iloc[-window_lt:]

    s_smooth = _smooth_annual(s, smooth_years=smooth_years)
    g_hist = _compute_cagr(s_smooth)

    g_neutral = max(g_hist, neutral_floor)
    g_optimistic = max(optimistic_floor, g_neutral - spread)
    g_pessimistic = g_neutral + spread

    return {
        "optimistic": g_optimistic,
        "neutral": g_neutral,
        "pessimistic": g_pessimistic,
    }


def build_paths_from_structural_growth(
    prices_annual: pd.Series,
    cfg: ScenarioConfig,
    *,
    window_lt: int,
    smooth_years: int,
    neutral_floor: float,
    spread: float,
    optimistic_floor: float,
    cap_down: float,
    cap_up: float,
) -> Tuple[int, float, Dict[str, np.ndarray], Dict[str, float]]:
    s = prices_annual.dropna().sort_index()
    base_year = int(s.index.max())
    P0 = float(s.loc[base_year])

    growth_rates = build_structural_growth_rates(
        s,
        window_lt=window_lt,
        smooth_years=smooth_years,
        neutral_floor=neutral_floor,
        spread=spread,
        optimistic_floor=optimistic_floor,
    )

    paths: Dict[str, np.ndarray] = {}
    for sc in SCENARIOS:
        paths[sc] = _build_constant_growth_path(
            P0=P0,
            growth=growth_rates[sc],
            horizon=cfg.horizon_years,
            cap_down=cap_down,
            cap_up=cap_up,
        )

    return base_year, P0, paths, growth_rates


# ============================================================
# ÉLECTRICITÉ PAR BUILDING_TYPE
# ============================================================

def build_electricity_option_B_by_building_type(
    df_elec: pd.DataFrame,
    cfg: ScenarioConfig,
) -> Tuple[int, pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Construit les scénarios électricité par building_type, en filtrant Standard product.

    Sortie:
      colonnes = canton, building_type, scenario, year, price_ttc_chf_kwh
    """
    df = df_elec.copy()

    # Filtre produit standard uniquement
    df = df[df["product"].str.lower() == cfg.electricity_product_filter.lower()].copy()
    if df.empty:
        raise ValueError(
            f"Aucune donnée électricité après filtre product='{cfg.electricity_product_filter}'."
        )

    all_records = []
    growth_by_building_type: Dict[str, Dict[str, float]] = {}
    base_years = []

    for building_type, df_bt in df.groupby("building_type"):
        df_bt = df_bt.copy()

        # Référence suisse médiane pour ce building_type
        ch = (
            df_bt.groupby("year")["price_ttc_chf_kwh"]
            .median()
            .dropna()
            .sort_index()
        )

        if len(ch) < 2:
            print(f"[WARN] building_type ignoré faute d'historique suffisant: {building_type}")
            continue

        base_year, _P0_ch, ch_paths, growth_rates = build_paths_from_structural_growth(
            ch,
            cfg,
            window_lt=min(cfg.window_lt_elec, len(ch)),
            smooth_years=cfg.smooth_years,
            neutral_floor=cfg.min_growth_elec_neutral,
            spread=cfg.spread_elec,
            optimistic_floor=cfg.floor_growth_elec_optimistic,
            cap_down=cfg.cap_elec_down,
            cap_up=cfg.cap_elec_up,
        )

        growth_by_building_type[building_type] = growth_rates
        base_years.append(base_year)

        years_hist = sorted([y for y in ch.index if y <= base_year])
        years_ref = years_hist[-5:] if len(years_hist) >= 1 else years_hist

        m_by_canton: Dict[str, float] = {}
        for canton, g in df_bt.groupby("canton"):
            g2 = g[g["year"].isin(years_ref)].copy()
            if g2.empty:
                continue

            ratios = []
            for _, row in g2.iterrows():
                y = int(row["year"])
                if y in ch.index and ch.loc[y] > 0:
                    ratios.append(float(row["price_ttc_chf_kwh"]) / float(ch.loc[y]))

            if len(ratios) == 0:
                continue

            m = float(np.median(ratios))
            m = float(np.clip(m, 0.6, 1.6))
            m_by_canton[canton] = m

        if not m_by_canton:
            print(f"[WARN] Aucun multiplicateur canton calculable pour building_type={building_type}")
            continue

        years_future = np.arange(base_year, base_year + cfg.horizon_years + 1, dtype=int)

        for sc in SCENARIOS:
            ch_prices = ch_paths[sc]
            for canton, m in m_by_canton.items():
                for i, y in enumerate(years_future):
                    all_records.append({
                        "canton": canton,
                        "building_type": building_type,
                        "scenario": sc,
                        "year": int(y),
                        "price_ttc_chf_kwh": float(m * ch_prices[i]),
                    })

    if not all_records:
        raise ValueError("Aucun scénario électricité n'a pu être construit.")

    df_out = pd.DataFrame.from_records(all_records)
    df_out = df_out.sort_values(["scenario", "building_type", "canton", "year"]).reset_index(drop=True)

    base_year_final = max(base_years) if base_years else None
    if base_year_final is None:
        raise ValueError("Impossible de déterminer une année de base pour l'électricité.")

    return base_year_final, df_out, growth_by_building_type


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Build realistic deterministic price scenarios (25 years).")
    ap.add_argument("--elec_csv", default="Data/Prix_electricite/Prix_electricite_canton.csv")
    ap.add_argument("--gas_csv", default="Data/Prix_gaz_Energie360.csv")
    ap.add_argument("--oil_csv", default="Data/Prix_mazout_Midland.csv")
    ap.add_argument("--out_dir", default="Data/processed")
    args = ap.parse_args()

    cfg = ScenarioConfig()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Electricity
    elec_path = Path(args.elec_csv)
    df_elec = load_electricity_canton_prices(elec_path)

    base_year_elec, df_elec_out, elec_growth = build_electricity_option_B_by_building_type(df_elec, cfg)

    elec_out_path = out_dir / "scenarios_electricity_ttc_by_canton.csv"
    df_elec_out.to_csv(elec_out_path, index=False, encoding="utf-8")
    print(f"[OK] Electricity scenarios -> {elec_out_path} (base_year={base_year_elec})")
    print("     growth rates electricity by building_type:")
    for bt, g in elec_growth.items():
        print(f"       - {bt}: {g}")

    # -------- Gas
    gas_path = Path(args.gas_csv)
    gas_annual = load_gas_prices(gas_path)

    base_year_gas, _P0_gas, gas_paths, gas_growth = build_paths_from_structural_growth(
        gas_annual,
        cfg,
        window_lt=min(cfg.window_lt_gas, len(gas_annual)),
        smooth_years=cfg.smooth_years,
        neutral_floor=cfg.min_growth_gas_neutral,
        spread=cfg.spread_gas,
        optimistic_floor=cfg.floor_growth_gas_optimistic,
        cap_down=cfg.cap_fuel_down,
        cap_up=cfg.cap_fuel_up,
    )

    years_gas = np.arange(base_year_gas, base_year_gas + cfg.horizon_years + 1, dtype=int)
    gas_records = []
    for sc in SCENARIOS:
        for i, y in enumerate(years_gas):
            gas_records.append({
                "scenario": sc,
                "year": int(y),
                "price_chf_kwh": float(gas_paths[sc][i]),
            })

    df_gas_out = pd.DataFrame(gas_records).sort_values(["scenario", "year"]).reset_index(drop=True)
    gas_out_path = out_dir / "scenarios_gas.csv"
    df_gas_out.to_csv(gas_out_path, index=False, encoding="utf-8")
    print(f"[OK] Gas scenarios -> {gas_out_path} (base_year={base_year_gas})")
    print("     growth rates gas:", gas_growth)

    # -------- Oil
    oil_path = Path(args.oil_csv)
    oil_annual = load_oil_prices_midland(oil_path)

    base_year_oil, _P0_oil, oil_paths, oil_growth = build_paths_from_structural_growth(
        oil_annual,
        cfg,
        window_lt=min(cfg.window_lt_oil, len(oil_annual)),
        smooth_years=cfg.smooth_years,
        neutral_floor=cfg.min_growth_oil_neutral,
        spread=cfg.spread_oil,
        optimistic_floor=cfg.floor_growth_oil_optimistic,
        cap_down=cfg.cap_fuel_down,
        cap_up=cfg.cap_fuel_up,
    )

    years_oil = np.arange(base_year_oil, base_year_oil + cfg.horizon_years + 1, dtype=int)
    oil_records = []
    for sc in SCENARIOS:
        for i, y in enumerate(years_oil):
            oil_records.append({
                "scenario": sc,
                "year": int(y),
                "price_chf_kwh": float(oil_paths[sc][i]),
            })

    df_oil_out = pd.DataFrame(oil_records).sort_values(["scenario", "year"]).reset_index(drop=True)
    oil_out_path = out_dir / "scenarios_mazout.csv"
    df_oil_out.to_csv(oil_out_path, index=False, encoding="utf-8")
    print(f"[OK] Mazout scenarios -> {oil_out_path} (base_year={base_year_oil})")
    print("     growth rates mazout:", oil_growth)

    print("\nDone.")


if __name__ == "__main__":
    main()