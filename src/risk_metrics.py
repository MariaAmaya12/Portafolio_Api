from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import TRADING_DAYS


def parametric_var_cvar(returns: pd.Series, alpha: float = 0.95) -> dict:
    """
    VaR y CVaR paramétricos (normal).
    Se reportan como pérdidas positivas.
    """
    r = returns.dropna()
    if r.empty:
        return {}

    mu = r.mean()
    sigma = r.std(ddof=1)
    q = 1 - alpha
    z = norm.ppf(q)

    var_daily = -(mu + sigma * z)
    cvar_daily = -(mu - sigma * norm.pdf(z) / q)

    return {
        "VaR_diario": var_daily,
        "CVaR_diario": cvar_daily,
        "VaR_anualizado": var_daily * np.sqrt(TRADING_DAYS),
        "CVaR_anualizado": cvar_daily * np.sqrt(TRADING_DAYS),
    }


def historical_var_cvar(returns: pd.Series, alpha: float = 0.95) -> dict:
    """
    VaR y CVaR históricos.
    """
    r = returns.dropna()
    if r.empty:
        return {}

    cutoff = np.quantile(r, 1 - alpha)
    tail = r[r <= cutoff]

    var_daily = -cutoff
    cvar_daily = -tail.mean()

    return {
        "VaR_diario": var_daily,
        "CVaR_diario": cvar_daily,
        "VaR_anualizado": var_daily * np.sqrt(TRADING_DAYS),
        "CVaR_anualizado": cvar_daily * np.sqrt(TRADING_DAYS),
    }


def monte_carlo_var_cvar(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    n_sim: int = 10000,
    seed: int = 42,
) -> dict:
    """
    VaR y CVaR de Monte Carlo usando normal multivariada.
    """
    clean = returns_df.dropna()
    if clean.empty:
        return {}

    rng = np.random.default_rng(seed)
    mu = clean.mean().values
    cov = clean.cov().values

    sims = rng.multivariate_normal(mu, cov, size=n_sim)
    port_sim = sims @ weights

    cutoff = np.quantile(port_sim, 1 - alpha)
    tail = port_sim[port_sim <= cutoff]

    var_daily = -cutoff
    cvar_daily = -tail.mean()

    return {
        "VaR_diario": var_daily,
        "CVaR_diario": cvar_daily,
        "VaR_anualizado": var_daily * np.sqrt(TRADING_DAYS),
        "CVaR_anualizado": cvar_daily * np.sqrt(TRADING_DAYS),
        "simulated_returns": port_sim,
    }


def risk_comparison_table(
    portfolio_returns: pd.Series,
    asset_returns_df: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    n_sim: int = 10000,
) -> pd.DataFrame:
    """
    Tabla comparativa de VaR y CVaR.
    """
    p = parametric_var_cvar(portfolio_returns, alpha=alpha)
    h = historical_var_cvar(portfolio_returns, alpha=alpha)
    m = monte_carlo_var_cvar(asset_returns_df, weights=weights, alpha=alpha, n_sim=n_sim)

    rows = []
    for name, res in [("Paramétrico", p), ("Histórico", h), ("Monte Carlo", m)]:
        if res:
            rows.append(
                {
                    "método": name,
                    "VaR_diario": res["VaR_diario"],
                    "CVaR_diario": res["CVaR_diario"],
                    "VaR_anualizado": res["VaR_anualizado"],
                    "CVaR_anualizado": res["CVaR_anualizado"],
                }
            )

    return pd.DataFrame(rows)
