from pathlib import Path
from monte_carlo_electricity import (
    MCConfig,
    run_monte_carlo_for_group,
    simulate_electricity_cost_for_heat_pump,
)
import numpy as np

cfg = MCConfig(
    price_col="price_ttc_median_chf_kwh",
    base_year=2026,
    horizon_years=25,
    n_sims=5000,
    seed=1,
)
BASE_DIR = Path(__file__).resolve().parents[1]

summary_csv_path = BASE_DIR / "Data" / "out" / "electricity_summary_by_canton_year_building.csv"


res = run_monte_carlo_for_group(
    summary_csv_path=str(summary_csv_path),
    canton="Vaud",
    building_type="maison_individuelle",
    product="Cheapest product",
    config=cfg,
)


print("Stats prix élec final:", res["stats_final_price"])

# Besoin chauffage annuel (exemple) — ici on mettra celui calc avec les hdd
E_heat = 18000  # kWh/an
COP = 4.0

cost_paths = simulate_electricity_cost_for_heat_pump(
    price_paths_chf_kwh=res["price_paths"],
    fixed_fee_paths_chf_year=res["fixed_fee_paths"],
    heat_demand_kwh_per_year=E_heat,
    cop_annual=COP,
    include_fixed_fee=True,
)

final_cost = cost_paths[:, -1]
print("Coût final médian:", np.median(final_cost))
print("P10/P90:", np.quantile(final_cost, [0.1, 0.9]))
