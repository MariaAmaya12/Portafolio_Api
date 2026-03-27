from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def plot_normalized_prices(close: pd.DataFrame) -> go.Figure:
    base = close / close.dropna().iloc[0] * 100
    fig = go.Figure()
    for col in base.columns:
        fig.add_trace(go.Scatter(x=base.index, y=base[col], mode="lines", name=col))
    fig.update_layout(title="Precios normalizados (base 100)", xaxis_title="Fecha", yaxis_title="Base 100")
    return fig


def plot_price_and_mas(df: pd.DataFrame, sma_col: str, ema_col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df[sma_col], mode="lines", name=sma_col))
    fig.add_trace(go.Scatter(x=df.index, y=df[ema_col], mode="lines", name=ema_col))
    fig.update_layout(title="Precio con medias móviles", xaxis_title="Fecha", yaxis_title="Precio")
    return fig


def plot_bollinger(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_up"], mode="lines", name="BB_up"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], mode="lines", name="BB_mid"))
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_low"], mode="lines", name="BB_low"))
    fig.update_layout(title="Bandas de Bollinger", xaxis_title="Fecha", yaxis_title="Precio")
    return fig


def plot_rsi(df: pd.DataFrame, rsi_col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[rsi_col], mode="lines", name=rsi_col))
    fig.add_hline(y=70, line_dash="dash")
    fig.add_hline(y=30, line_dash="dash")
    fig.update_layout(title="RSI", xaxis_title="Fecha", yaxis_title="RSI")
    return fig


def plot_macd(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], mode="lines", name="MACD_signal"))
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD_hist"))
    fig.update_layout(title="MACD", xaxis_title="Fecha")
    return fig


def plot_stochastic(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["%K"], mode="lines", name="%K"))
    fig.add_trace(go.Scatter(x=df.index, y=df["%D"], mode="lines", name="%D"))
    fig.add_hline(y=80, line_dash="dash")
    fig.add_hline(y=20, line_dash="dash")
    fig.update_layout(title="Oscilador Estocástico", xaxis_title="Fecha")
    return fig


def plot_histogram_with_normal(returns: pd.Series) -> go.Figure:
    r = returns.dropna()
    mu, sigma = r.mean(), r.std(ddof=1)
    x = np.linspace(r.min(), r.max(), 300)
    y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, histnorm="probability density", name="Histograma", nbinsx=40))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Normal teórica"))
    fig.update_layout(title="Histograma con curva normal", xaxis_title="Rendimiento", yaxis_title="Densidad")
    return fig


def plot_qq(qq_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=qq_df["theoretical_quantiles"],
            y=qq_df["sample_quantiles"],
            mode="markers",
            name="Q-Q",
        )
    )
    min_q = min(qq_df.min())
    max_q = max(qq_df.max())
    fig.add_trace(go.Scatter(x=[min_q, max_q], y=[min_q, max_q], mode="lines", name="45°"))
    fig.update_layout(title="Q-Q plot", xaxis_title="Cuantiles teóricos", yaxis_title="Cuantiles muestrales")
    return fig


def plot_box(returns: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Box(y=returns.dropna(), name="Rendimientos"))
    fig.update_layout(title="Boxplot de rendimientos", yaxis_title="Rendimiento")
    return fig


def plot_volatility(vol_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in vol_df.columns:
        fig.add_trace(go.Scatter(x=vol_df.index, y=vol_df[col], mode="lines", name=col))
    fig.update_layout(title="Volatilidad condicional estimada", xaxis_title="Fecha", yaxis_title="Volatilidad")
    return fig


def plot_forecast(forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_df["horizonte"],
            y=forecast_df["volatilidad_pronosticada"],
            mode="lines+markers",
            name="Forecast",
        )
    )
    fig.update_layout(title="Pronóstico de volatilidad", xaxis_title="Horizonte", yaxis_title="Volatilidad")
    return fig


def plot_scatter_regression(x: np.ndarray, y: np.ndarray, yhat: np.ndarray, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Observaciones"))
    sort_idx = np.argsort(x)
    fig.add_trace(go.Scatter(x=x[sort_idx], y=yhat[sort_idx], mode="lines", name="Regresión"))
    fig.update_layout(title=title, xaxis_title="Exceso benchmark", yaxis_title="Exceso activo")
    return fig


def plot_var_distribution(returns: pd.Series, table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns.dropna(), nbinsx=50, name="Rendimientos"))
    for _, row in table.iterrows():
        fig.add_vline(x=-row["VaR_diario"], line_dash="dash", annotation_text=f'VaR {row["método"]}')
        fig.add_vline(x=-row["CVaR_diario"], line_dash="dot", annotation_text=f'CVaR {row["método"]}')
    fig.update_layout(title="Distribución con líneas VaR / CVaR", xaxis_title="Rendimiento")
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriz de correlación")
    return fig


def plot_frontier(sim_df: pd.DataFrame, frontier_df: pd.DataFrame, min_var: pd.Series, max_sharpe: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sim_df["volatility"],
            y=sim_df["return"],
            mode="markers",
            marker=dict(size=5, color=sim_df["sharpe"], showscale=True),
            name="Portafolios",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=frontier_df["volatility"],
            y=frontier_df["return"],
            mode="lines",
            name="Frontera eficiente",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_var["volatility"]],
            y=[min_var["return"]],
            mode="markers",
            marker=dict(size=12, symbol="diamond"),
            name="Mínima varianza",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[max_sharpe["volatility"]],
            y=[max_sharpe["return"]],
            mode="markers",
            marker=dict(size=12, symbol="star"),
            name="Máximo Sharpe",
        )
    )
    fig.update_layout(title="Frontera eficiente de Markowitz", xaxis_title="Volatilidad", yaxis_title="Retorno")
    return fig


def plot_benchmark_base100(port: pd.Series, bench: pd.Series) -> go.Figure:
    df = pd.concat([port, bench], axis=1).dropna()
    df.columns = ["Portafolio", "Benchmark"]
    base = df / df.iloc[0] * 100

    fig = go.Figure()
    for col in base.columns:
        fig.add_trace(go.Scatter(x=base.index, y=base[col], mode="lines", name=col))
    fig.update_layout(title="Portafolio vs benchmark (base 100)", xaxis_title="Fecha", yaxis_title="Base 100")
    return fig
