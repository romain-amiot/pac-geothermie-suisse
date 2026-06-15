from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np #besoin de numpy pour les calculs numériques et les simulations
import pandas as pd   #besoin de pandas pour lire et manipuler les données de prix d'électricité


# -------------------------
# ce scipt est un module de fonctions, qu'on importe depuis run_monte_carlo_demo.py. L'idée est de séparer la logique de simulation (ce fichier) de l'exemple d'utilisation (run_monte_carlo_demo.py).
# -------------------------

@dataclass
class MCConfig:
    """
    Configuration Monte Carlo.
    - price_col: quelle colonne de prix utiliser comme référence (TTC ou hors TVA)
    - base_year: année de départ (P0). Si None => dernière année dispo pour le groupe.
    - horizon_years: nombre d'années simulées
    - n_sims: nombre de simulations
    - seed: biais de randomisation pour reproductibilité
    - cap_up/down: plafonds de variation annuelle (ex: +0.4 / -0.3). None => pas de caps
    - floor/ceiling: bornes absolues en CHF/kWh (optionnel)
    """
    price_col: str = "price_ttc_median_chf_kwh"  # on utilise TTC pour coller aux factures, sinon "price_median_chf_kwh" (hors TVA)
    base_year: Optional[int] = None
    horizon_years: int = 25
    n_sims: int = 2000
    seed: int = 1  #on fixe la seed pour avoir les memes resultats à chaque exécution, ce qui permet de voir les effets des changements de paramètres

    cap_up: Optional[float] = 0.40     # +40% max/an
    cap_down: Optional[float] = -0.30  # -30% max/an

    floor_price: Optional[float] = 0.05     # pour pas avoir des prix négatifs ou irréalistes
    ceiling_price: Optional[float] = 1.50     # pour pas avoir des prix irréalistes (ex: >1.5 CHF/kWh)

    
    fixed_fee_growth: float = 0.0  # croissance annuelle de l'abonnement fixe (ex: 0.02 pour +2%/an pour l'inflation)


# -------------------------
# Chargement & préparation
# -------------------------

def load_elcom_summary(path_csv: str) -> pd.DataFrame:       #fonction pour charger les données de prix d'électricité à partir d'un fichier CSV. On s'assure que les types sont corrects et qu'on a une colonne "year" en int, ainsi que des colonnes de prix en float.
    df = pd.read_csv(path_csv) #on lit le fichier CSV avec pandas
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")  # année en int, NaN si conversion impossible
    for c in ["canton", "building_type", "product"]:     
        df[c] = df[c].astype(str).str.strip()     # s'assure que ce sont des strings sans espaces 
    return df


def _safe_log_returns(prices: pd.Series) -> np.ndarray:
    """ on utilise les log return pour calibrer mu/sigma, car c'est plus robuste et plus adapté à la nature multiplicative des prix."""
    p = prices.dropna().astype(float)     #on lit les prix, on ignore les NaN et on s'assure que c'est du float
    p = p[p > 0]    #on ignore les prix négatifs ou nuls, qui ne sont pas valides pour le calcul des log-returns
    if len(p) < 2:   #si on a moins de 2 points valides, on ne peut pas calculer de log-return, donc on retourne un array vide
        return np.array([])
    r = np.log(p.values[1:] / p.values[:-1])   #calcul des log-returns : ln(Pt/Pt-1) = ln(Pt) - ln(Pt-1)
    return r[np.isfinite(r)]      #on retourne seulement les log-returns qui sont des nombres finis (pas d'infini ou de NaN)


def _calibrate_mu_sigma_from_group(
    df_group: pd.DataFrame, price_col: str
) -> Tuple[float, float]:
    """
    Calibre mu et sigma à partir de l'historique du groupe
    mu = moyenne des log-returns annuels --> tendance centrale de la croissance annuelle du prix
    sigma = écart-type des log-returns annuels  --> volatilité annuelle du prix
    """
    g = df_group.sort_values("year") #on trie les données du groupe par année pour avoir une série temporelle ordonnée
    returns = _safe_log_returns(g[price_col])    #on calcule les log-returns à partir de la colonne de prix spécifiée, en utilisant la fonction précédente qui gère les cas problématiques
    if returns.size < 3:
        # Il faut assez d’années pour estimer une volatilité décente (sinon c’est instable): Trop peu de points -> signaler via NaN 
        return (float("nan"), float("nan"))
    mu = float(np.mean(returns))    #on calcule la moyenne des log-returns pour obtenir mu, qui représente la tendance centrale de la croissance annuelle du prix
    sigma = float(np.std(returns, ddof=1))      #on calcule l'écart-type des log-returns pour obtenir sigma, qui représente la volatilité annuelle du prix. On utilise ddof=1 afin d'avoir un estimateur non biaisé de l'écart-type (diviser par n-1 au lieu de n)
    sigma = max(sigma, 1e-6) # Evite sigma = 0 (sinon Monte Carlo dégénère)
    return (mu, sigma)


def calibrate_params_with_fallback(
    df: pd.DataFrame,
    canton: str,
    building_type: str,
    product: str,
    price_col: str,
) -> Tuple[float, float]:    
    """
    Calibre (mu, sigma) pour un groupe. un groupe est défini par la combinaison (canton, building_type, product)
    Fallback successifs si historique insuffisant :
      1) même building_type + product (tous cantons)
      2) tous groupes (Suisse globale)
    """
    group_mask = (
        (df["canton"] == canton)
        & (df["building_type"] == building_type)
        & (df["product"] == product)
    ) #on crée un masque pour filtrer les données du DataFrame en fonction du canton, du type de bâtiment et du produit spécifiés. Cela nous permet d'obtenir les données historiques de prix pour ce groupe spécifique.
    df_g = df[group_mask].copy()   #on applique le masque pour obtenir un DataFrame qui contient uniquement les données du groupe ciblé. 
    mu, sigma = _calibrate_mu_sigma_from_group(df_g, price_col)   #on tente de calibrer mu et sigma à partir de l'historique du groupe ciblé. Si on a suffisamment de données, on retourne ces valeurs.
    if np.isfinite(mu) and np.isfinite(sigma):         #si les valeurs de mu et sigma sont finies (pas de NaN ou d'infini), cela signifie que le calibrage a réussi, et on peut les retourner.
        return mu, sigma

    # Fallback 1: si pas assez de d'années pour le groupe ciblé, on essaie de calibrer à partir de tous les groupes qui ont le même building_type et product, indépendamment du canton. Cela nous donne une estimation plus générale basée sur des données similaires.
    fb1 = df[(df["building_type"] == building_type) & (df["product"] == product)].copy()
    mu, sigma = _calibrate_mu_sigma_from_group(fb1, price_col)
    if np.isfinite(mu) and np.isfinite(sigma):
        return mu, sigma

    # Fallback 2: si toujours pas assez de données, on calibre à partir de tous les groupes (Suisse globale), ce qui nous donne une estimation très générale basée sur l'ensemble des données disponibles.
    mu, sigma = _calibrate_mu_sigma_from_group(df.copy(), price_col)
    if np.isfinite(mu) and np.isfinite(sigma):
        return mu, sigma

    # Ultime fallback : si on n'a vraiment pas assez de données pour calibrer, on retourne des valeurs par défaut raisonnables (ex: mu=2%/an, sigma=10%/an), qui peuvent être ajustées 
    return (0.02, 0.10)


def get_P0_and_fixed_fee(
    df: pd.DataFrame,
    canton: str,
    building_type: str,
    product: str,
    price_col: str,
    base_year: Optional[int],
) -> Tuple[int, float, float]:
    """
    Retourne :
      - base_year_effective    --> année de base utilisée (peut différer de base_year si pas dispo, on prend la plus proche inférieure ou la dernière dispo)
      - P0 (CHF/kWh) --> prix de référence pour l'année de base
      - fixed_fee_0 (CHF/an) --> abonnement fixe pour l'année de base (si dispo, sinon 0)
    """
    mask = (
        (df["canton"] == canton)
        & (df["building_type"] == building_type)
        & (df["product"] == product)
    )
    g = df[mask].copy()
    if g.empty:
        raise ValueError(f"Aucune donnée pour {canton=} {building_type=} {product=}")

    g = g.sort_values("year")     #on trie les données du groupe par année pour pouvoir trouver l'année de base et les prix correspondants de manière ordonnée
    if base_year is None:
        base_year_effective = int(g["year"].dropna().max())   #si aucune année de base n'est spécifiée, on prend la dernière année disponible dans les données du groupe comme année de base
    else:
        base_year_effective = int(base_year) 

    row = g[g["year"] == base_year_effective]   #on essaie de trouver la ligne du DataFrame qui correspond à l'année de base effective. Si on trouve une ligne, c'est parfait, sinon on devra faire un fallback pour trouver une année de base proche.
    if row.empty:
        # si l’année demandée n’existe pas, prendre l’année la plus proche inférieure, sinon la dernière dispo
        older = g[g["year"] < base_year_effective]
        if not older.empty:
            row = older.iloc[[-1]]
            base_year_effective = int(row["year"].iloc[0])
        else:
            row = g.iloc[[-1]]
            base_year_effective = int(row["year"].iloc[0])

    P0 = float(row[price_col].iloc[0]) #on lit le prix de référence P0 à partir de la colonne spécifiée pour l'année de base effective. Si cette valeur n'est pas finie ou est négative, on considère que c'est une erreur, car on a besoin d'un prix de départ valide pour la simulation.
    if not np.isfinite(P0) or P0 <= 0:
        raise ValueError(f"P0 invalide pour {canton=} {building_type=} {product=} {base_year_effective=}")

    # Abonnement fixe : si NaN, on met 0 
    fixed_fee_0 = float(row["fixed_median_chf_year"].iloc[0]) if "fixed_median_chf_year" in row.columns else 0.0 #on lit l'abonnement fixe pour l'année de base effective à partir de la colonne "fixed_median_chf_year". Si cette colonne n'existe pas ou si la valeur n'est pas finie, on considère que l'abonnement fixe est de 0 CHF/an, ce qui signifie qu'on ne prend pas en compte un coût fixe dans la simulation.
    if not np.isfinite(fixed_fee_0):
        fixed_fee_0 = 0.0

    return base_year_effective, P0, fixed_fee_0


# -------------------------
# Simulation Monte Carlo
# -------------------------

def simulate_price_paths(
    P0: float,
    mu: float,
    sigma: float,
    horizon_years: int,
    n_sims: int,
    seed: int = 42,
    cap_up: Optional[float] = None,
    cap_down: Optional[float] = None,
    floor_price: Optional[float] = None,
    ceiling_price: Optional[float] = None,
) -> np.ndarray:
    """
    idée: simuler un prix qui évolue multiplicativement (en pourcentage) d’année en année, avec une tendance + une incertitude 
    simule des trajectoires annuelles de prix selon un random walk lognormal :
      ln(P_{t+1}) = ln(P_t) + mu + sigma * eps_t  equivalent a P_{t+1} = P_t * exp(mu + sigma * eps_t)
    on utilise le random walk lognormal car stable numériquement et rapide, et adapté à la nature multiplicative des prix d'électricité (les prix ont tendance à évoluer en pourcentage plutôt qu'en valeur absolue).
    mu et sigma sont calibrés à partir de l'historique du groupe ciblé, ou via des fallback si pas assez de données.
    eps_t sont des chocs aléatoires tirés d'une distribution normale standard (moyenne 0, écart-type 1), qui introduisent de la variabilité dans les trajectoires de prix simulées.
    """
    rng = np.random.default_rng(seed) #on crée un générateur de nombres aléatoires avec la seed spécifiée pour assurer la reproductibilité des simulations.
    eps = rng.normal(0.0, 1.0, size=(n_sims, horizon_years)) #on génère une matrice de nombres aléatoires suivant une distribution normale standard (moyenne 0, écart-type 1) avec une taille de (smulations x années ). Chaque élément de cette matrice représente un choc aléatoire pour la simulation du prix à chaque année et pour chaque simulation.
    logP = np.zeros((n_sims, horizon_years + 1), dtype=float)  #on initialise une matrice pour stocker les log-prix simulés, avec une taille de (n_sims, horizon_years+1) pour inclure l'année de base (année 0). On remplit la première colonne de cette matrice avec le log de P0, qui représente le prix de départ pour toutes les simulations.
    logP[:, 0] = math.log(P0) #on remplit la première colonne de logP avec le log de P0, ce qui signifie que toutes les simulations commencent avec le même prix de base P0 à l'année 0.

    for t in range(horizon_years):
        step = mu + sigma * eps[:, t]      #pour chaque année t, on calcule le changement de log-prix (step) pour chaque simulation en utilisant la formule du random walk lognormal.

        # caps en termes de variation relative annuelle
        if cap_up is not None or cap_down is not None:   #on applique des plafonds de variation annuelle pour éviter des changements de prix irréalistes d'une année à l'autre. Ces caps sont exprimés en termes de variation relative (ex: +40% max/an), et on les convertit en log pour les appliquer sur les log-returns.
            # approx : Δlog ≈ log(1+Δ)  --> pour une variation relative de +40%, le cap en log serait log(1+0.4) ≈ 0.336, ce qui signifie que le log-prix ne peut pas augmenter de plus de 0.336 d'une année à l'autre, ce qui correspond à une augmentation de 40% en termes de prix.
            if cap_up is not None:
                step = np.minimum(step, math.log(1.0 + cap_up))
            if cap_down is not None:
                step = np.maximum(step, math.log(1.0 + cap_down))

        logP[:, t + 1] = logP[:, t] + step

    P = np.exp(logP) #on exponentie les log-prix pour obtenir les prix simulés en CHF/kWh. La matrice P a la même taille que logP, soit (n_sims, horizon_years+1), et contient les trajectoires de prix pour chaque simulation et chaque année, incluant l'année de base avec P0.

    # bornes absolues
    if floor_price is not None:   # differents des caps qui sont juste d'une année à l'autre, là les bornes absolues s'assurent que les prix simulés restent dans une plage réaliste en CHF/kwh. 
        P = np.maximum(P, float(floor_price))
    if ceiling_price is not None:
        P = np.minimum(P, float(ceiling_price))

    return P


def simulate_fixed_fee_paths(
    fixed_fee_0: float,
    horizon_years: int,
    n_sims: int,
    growth: float = 0.0,
) -> np.ndarray:
    """
    Abonnement fixe : ici on le fait évoluer de manière déterministe (même pour toutes les sims)
    """
    years = np.arange(horizon_years + 1)
    fee = fixed_fee_0 * (1.0 + growth) ** years
    return np.tile(fee, (n_sims, 1)) #on crée une matrice où chaque ligne est la même série d'abonnement fixe évoluant selon la croissance spécifiée, et on répète cette série pour le nombre de simulations. La matrice résultante a une taille de (n_sims, horizon_years+1) et contient les coûts d'abonnement fixe pour chaque année et chaque simulation, même si dans ce cas ils sont identiques entre les simulations.


def summarize_simulations(values: np.ndarray) -> Dict[str, float]:
    """
    Résumés sur la valeur finale (dernière année) par défaut, ou une série.
    """
    if values.ndim == 2:  #si on a une matrice 2D, on suppose que les trajectoires sont dans les colonnes, et on prend la dernière colonne pour faire les statistiques sur la valeur finale. Si values est déjà 1D, on l'utilise directement.
        x = values[:, -1]
    else:
        x = values
    x = x[np.isfinite(x)]
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
    }


# -------------------------
# Couplage le modèle énergie (coûts chauffage PAC)
# -------------------------

def simulate_electricity_cost_for_heat_pump(
    price_paths_chf_kwh: np.ndarray,
    fixed_fee_paths_chf_year: np.ndarray,
    heat_demand_kwh_per_year: float,
    cop_annual: float,
    include_fixed_fee: bool = True,
) -> np.ndarray:
    """
    Coût annuel pour la PAC :
      conso_elec = E_chauffage / COP
      coût = conso_elec * prix_kWh + (abonnement si activé)

    Retour :
      coûts annuels shape (n_sims, horizon_years+1) (inclut année 0)
    """
    if cop_annual <= 0:
        raise ValueError("COP annuel invalide")
    elec_kwh = heat_demand_kwh_per_year / cop_annual

    costs = elec_kwh * price_paths_chf_kwh
    if include_fixed_fee:
        costs = costs + fixed_fee_paths_chf_year
    return costs


# -------------------------
# Exemple d'utilisation
# -------------------------

def run_monte_carlo_for_group(
    summary_csv_path: str,
    canton: str,
    building_type: str,
    product: str,
    config: MCConfig,
) -> Dict[str, object]:
    """
    Charge les données, calibre mu/sigma, récupère P0, simule des trajectoires.
    Renvoie un dict avec trajectoires et stats grace aux fonctions précédentes.
    """
    df = load_elcom_summary(summary_csv_path)

    # P0 et fixe
    base_year_eff, P0, fixed0 = get_P0_and_fixed_fee(
        df, canton, building_type, product, config.price_col, config.base_year
    )

    # calib mu/sigma
    mu, sigma = calibrate_params_with_fallback(
        df, canton, building_type, product, config.price_col
    )

    price_paths = simulate_price_paths(
        P0=P0,
        mu=mu,
        sigma=sigma,
        horizon_years=config.horizon_years,
        n_sims=config.n_sims,
        seed=config.seed,
        cap_up=config.cap_up,
        cap_down=config.cap_down,
        floor_price=config.floor_price,
        ceiling_price=config.ceiling_price,
    )

    fixed_paths = simulate_fixed_fee_paths(
        fixed_fee_0=fixed0,
        horizon_years=config.horizon_years,
        n_sims=config.n_sims,
        growth=config.fixed_fee_growth,
    )

    # Stats sur le prix final
    stats_final_price = summarize_simulations(price_paths)

    return {
        "base_year": base_year_eff,
        "P0": P0,
        "fixed_fee_0": fixed0,
        "mu": mu,
        "sigma": sigma,
        "price_paths": price_paths,  # (n_sims, horizon+1)
        "fixed_fee_paths": fixed_paths,
        "stats_final_price": stats_final_price,
    }
