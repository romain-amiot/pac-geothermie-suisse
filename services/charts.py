from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


DISCOUNT_RATE = 0.03


def _discount_series(cost_series: pd.Series, discount_rate: float = DISCOUNT_RATE) -> pd.Series:
    """
    Actualise une série annuelle de coûts à partir de la première année de la série.
    """
    years = cost_series.index.astype(int)
    y0 = int(years.min())

    discount_factors = pd.Series(
        [1.0 / ((1.0 + discount_rate) ** (int(y) - y0)) for y in years],
        index=cost_series.index,
    )

    return cost_series * discount_factors


def _find_crossover_year(cumul_actuel: pd.Series, cumul_pac: pd.Series) -> int | None:
    """
    Retourne la première année où le coût cumulé PAC devient
    inférieur ou égal au coût cumulé du système actuel.
    """
    diff = cumul_pac - cumul_actuel
    crossing_years = diff.index[diff <= 0]

    if len(crossing_years) == 0:
        return None

    return int(crossing_years[0])


def build_discounted_cumulative_cost_bar_chart(
    cost_actuel: pd.Series,
    cost_pac: pd.Series,
    capex_net: float,
    scenario_label: str,
    discount_rate: float = DISCOUNT_RATE,
):
    years = cost_actuel.index.astype(int)

    discounted_actuel = _discount_series(cost_actuel, discount_rate=discount_rate)
    discounted_pac = _discount_series(cost_pac, discount_rate=discount_rate)

    cumul_actuel = discounted_actuel.cumsum()
    cumul_pac = discounted_pac.cumsum() + float(capex_net)

    x = np.arange(len(years))
    width = 0.42

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, cumul_actuel.values, width=width, label="Système actuel")
    ax.bar(x + width / 2, cumul_pac.values, width=width, label="PAC géothermique (CAPEX inclus)")

    crossover_year = _find_crossover_year(cumul_actuel, cumul_pac)
    if crossover_year is not None:
        idx = list(years).index(crossover_year)
        y_cross = max(cumul_actuel.loc[crossover_year], cumul_pac.loc[crossover_year])

        ax.axvline(idx, linestyle="--", linewidth=1.5)
        ax.annotate(
            f"Croisement actualisé : {crossover_year}",
            xy=(idx, y_cross),
            xytext=(idx + 0.6, y_cross * 1.02),
            arrowprops=dict(arrowstyle="->", lw=1),
            fontsize=9,
        )

    ax.set_title(f"Coûts cumulés actualisés sur 25 ans — {scenario_label}")
    ax.set_xlabel("Année")
    ax.set_ylabel("Coût cumulé actualisé [CHF]")
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig


def build_cumulative_emissions_bar_chart(
    emissions_actuel: pd.Series,
    emissions_pac: pd.Series,
    scenario_label: str,
):
    years = emissions_actuel.index.astype(int)

    cumul_actuel = emissions_actuel.cumsum()
    cumul_pac = emissions_pac.cumsum()

    x = np.arange(len(years))
    width = 0.42

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - width / 2, cumul_actuel.values, width=width, label="Système actuel")
    ax.bar(x + width / 2, cumul_pac.values, width=width, label="PAC géothermique")

    ax.set_title(f"Émissions cumulées de CO₂ sur 25 ans — {scenario_label}")
    ax.set_xlabel("Année")
    ax.set_ylabel("Émissions cumulées [kgCO₂e]")
    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig