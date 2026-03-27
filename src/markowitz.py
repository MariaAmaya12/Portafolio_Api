from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS


def simulate_portfolios(
    returns_df: pd.DataFrame,
    rf_annual: float,
    n_portfolios: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simula portafolios aleatorios con pesos no negativos que suman 1.
    """
    clean = returns_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.empty or clean.shape[0] < 2 or clean.shape[1] < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    mean_returns = clean.mean().values * TRADING_DAYS
    cov_matrix = clean.cov().values * TRADING_DAYS
    n_assets = clean.shape[1]

    rows = []
    for _ in range(n_portfolios):
        weights = rng.random(n_assets)
        weights = weights / weights.sum()

        port_return = float(weights @ mean_returns)
        port_vol = float(np.sqrt(weights.T @ cov_matrix @ weights))

        if not np.isfinite(port_return) or not np.isfinite(port_vol) or port_vol <= 0:
            continue

        sharpe = (port_return - rf_annual) / port_vol

        row = {
            "return": port_return,
            "volatility": port_vol,
            "sharpe": sharpe,
        }
        for idx, col in enumerate(clean.columns):
            row[f"w_{col}"] = weights[idx]
        rows.append(row)

    sim_df = pd.DataFrame(rows)
    if sim_df.empty:
        return sim_df

    sim_df = sim_df.replace([np.inf, -np.inf], np.nan)
    sim_df = sim_df.dropna(subset=["return", "volatility", "sharpe"])

    return sim_df


def efficient_frontier(sim_df: pd.DataFrame, n_bins: int = 60) -> pd.DataFrame:
    """
    Aproximación de frontera eficiente a partir de portafolios simulados.
    """
    if sim_df.empty:
        return pd.DataFrame(columns=["volatility", "return"])

    df = sim_df[["volatility", "return"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["volatility", "return"])

    if df.empty:
        return pd.DataFrame(columns=["volatility", "return"])

    # Ajustar número de bins a la variabilidad realmente observada
    n_unique = int(df["volatility"].nunique())
    n_bins_eff = min(n_bins, max(2, n_unique))

    df["bin"] = pd.cut(df["volatility"], bins=n_bins_eff, duplicates="drop")
    df = df.dropna(subset=["bin"])

    if df.empty:
        return pd.DataFrame(columns=["volatility", "return"])

    frontier = (
        df.groupby("bin", observed=True, group_keys=False)
        .apply(lambda x: x.loc[x["return"].idxmax()])
        .reset_index(drop=True)
        .sort_values("volatility")
    )

    if frontier.empty:
        return pd.DataFrame(columns=["volatility", "return"])

    frontier["cummax_return"] = frontier["return"].cummax()
    frontier = frontier[frontier["return"] >= frontier["cummax_return"]]

    return frontier[["volatility", "return"]].dropna()


def minimum_variance_portfolio(sim_df: pd.DataFrame) -> pd.Series:
    if sim_df.empty:
        return pd.Series(dtype=float)

    valid = sim_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["volatility"])
    if valid.empty:
        return pd.Series(dtype=float)

    return valid.loc[valid["volatility"].idxmin()]


def maximum_sharpe_portfolio(sim_df: pd.DataFrame) -> pd.Series:
    if sim_df.empty:
        return pd.Series(dtype=float)

    valid = sim_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["sharpe"])
    if valid.empty:
        return pd.Series(dtype=float)

    return valid.loc[valid["sharpe"].idxmax()]


def weights_table(portfolio_row: pd.Series) -> pd.DataFrame:
    """
    Extrae pesos de una fila de resultados.
    """
    if portfolio_row.empty:
        return pd.DataFrame(columns=["activo", "peso"])

    items = []
    for key, value in portfolio_row.items():
        if key.startswith("w_"):
            items.append({"activo": key.replace("w_", ""), "peso": value})

    return pd.DataFrame(items).sort_values("peso", ascending=False).reset_index(drop=True)