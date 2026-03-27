from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TRADING_DAYS


def clean_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia precios:
    - elimina duplicados de índice,
    - ordena por fecha,
    - elimina filas completamente vacías.
    """
    if df.empty:
        return df
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out = out.dropna(how="all")
    return out


def align_close_prices(close: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Alinea la matriz de cierres.
    """
    if close.empty:
        return close
    out = clean_price_frame(close)
    if dropna:
        out = out.dropna()
    return out


def simple_returns(price_series: pd.Series) -> pd.Series:
    """
    Rendimiento simple diario.
    """
    return price_series.pct_change().dropna()


def log_returns(price_series: pd.Series) -> pd.Series:
    """
    Rendimiento logarítmico diario.
    """
    return np.log(price_series / price_series.shift(1)).dropna()


def equal_weight_vector(n_assets: int) -> np.ndarray:
    """
    Genera pesos iguales para n activos.
    """
    if n_assets <= 0:
        return np.array([])
    return np.repeat(1 / n_assets, n_assets)


def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    """
    Retorno de portafolio equiponderado.
    """
    if returns.empty:
        return pd.Series(dtype=float)
    clean = returns.dropna()
    weights = equal_weight_vector(clean.shape[1])
    port = clean @ weights
    port.name = "portfolio_equal_weight"
    return port


def annualize_return(daily_returns: pd.Series) -> float:
    """
    Anualiza el retorno medio diario.
    """
    if daily_returns.empty:
        return float("nan")
    return float((1 + daily_returns.mean()) ** TRADING_DAYS - 1)


def annualize_volatility(daily_returns: pd.Series) -> float:
    """
    Anualiza la volatilidad diaria.
    """
    if daily_returns.empty:
        return float("nan")
    return float(daily_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def base_100(series: pd.Series) -> pd.Series:
    """
    Convierte una serie a base 100.
    """
    if series.empty:
        return series
    return 100 * series / series.dropna().iloc[0]
