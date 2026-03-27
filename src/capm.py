from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.config import TRADING_DAYS


def to_daily_rf(rf_annual: float) -> float:
    """
    Convierte tasa anual nominal/aproximada a tasa diaria efectiva.
    """
    if pd.isna(rf_annual):
        return np.nan
    return (1 + rf_annual) ** (1 / TRADING_DAYS) - 1


def compute_beta_and_capm(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    rf_annual: float,
) -> dict:
    """
    Calcula beta, alpha y CAPM vía regresión lineal simple.
    """
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if df.empty:
        return {}

    df.columns = ["asset", "market"]
    rf_daily = to_daily_rf(rf_annual)

    asset_excess = df["asset"] - rf_daily
    market_excess = df["market"] - rf_daily

    reg = linregress(market_excess, asset_excess)
    beta = reg.slope
    alpha_daily = reg.intercept
    market_premium_annual = market_excess.mean() * TRADING_DAYS
    expected_return = rf_annual + beta * market_premium_annual

    if beta > 1.1:
        classification = "Agresivo"
    elif beta < 0.9:
        classification = "Defensivo"
    else:
        classification = "Neutro"

    return {
        "beta": beta,
        "alpha_diaria": alpha_daily,
        "r_value": reg.rvalue,
        "r_squared": reg.rvalue ** 2,
        "p_value_beta": reg.pvalue,
        "expected_return_capm_annual": expected_return,
        "classification": classification,
        "regression_line": {
            "x": market_excess.values,
            "y": alpha_daily + beta * market_excess.values,
        },
        "scatter_data": {
            "market_excess": market_excess.values,
            "asset_excess": asset_excess.values,
        },
    }


def jensen_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_annual: float,
) -> float:
    """
    Alpha de Jensen anualizado.
    """
    results = compute_beta_and_capm(
        asset_returns=portfolio_returns,
        market_returns=benchmark_returns,
        rf_annual=rf_annual,
    )
    if not results:
        return np.nan

    beta = results["beta"]
    realized_annual = portfolio_returns.mean() * TRADING_DAYS
    expected = rf_annual + beta * ((benchmark_returns.mean() * TRADING_DAYS) - rf_annual)
    return realized_annual - expected
