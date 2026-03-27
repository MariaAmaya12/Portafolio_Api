from __future__ import annotations

import pandas as pd


def evaluate_signals(
    df: pd.DataFrame,
    rsi_overbought: float = 70,
    rsi_oversold: float = 30,
    stoch_overbought: float = 80,
    stoch_oversold: float = 20,
) -> dict:
    """
    Evalúa reglas técnicas usando el último dato disponible.
    """
    clean = df.dropna()
    if clean.empty:
        return {}

    last = clean.iloc[-1]
    prev = clean.iloc[-2] if len(clean) >= 2 else clean.iloc[-1]

    macd_buy = (prev["MACD"] <= prev["MACD_signal"]) and (last["MACD"] > last["MACD_signal"])
    macd_sell = (prev["MACD"] >= prev["MACD_signal"]) and (last["MACD"] < last["MACD_signal"])

    rsi_buy = last.filter(like="RSI_").iloc[0] < rsi_oversold
    rsi_sell = last.filter(like="RSI_").iloc[0] > rsi_overbought

    boll_buy = last["Close"] < last["BB_low"]
    boll_sell = last["Close"] > last["BB_up"]

    sma_col = [c for c in clean.columns if c.startswith("SMA_")][0]
    ema_col = [c for c in clean.columns if c.startswith("EMA_")][0]
    golden_cross = (prev[ema_col] <= prev[sma_col]) and (last[ema_col] > last[sma_col])
    death_cross = (prev[ema_col] >= prev[sma_col]) and (last[ema_col] < last[sma_col])

    stoch_buy = (prev["%K"] <= prev["%D"]) and (last["%K"] > last["%D"]) and (last["%K"] < stoch_oversold)
    stoch_sell = (prev["%K"] >= prev["%D"]) and (last["%K"] < last["%D"]) and (last["%K"] > stoch_overbought)

    # Normalizar a bool nativo de Python
    macd_buy = bool(macd_buy)
    macd_sell = bool(macd_sell)
    rsi_buy = bool(rsi_buy)
    rsi_sell = bool(rsi_sell)
    boll_buy = bool(boll_buy)
    boll_sell = bool(boll_sell)
    golden_cross = bool(golden_cross)
    death_cross = bool(death_cross)
    stoch_buy = bool(stoch_buy)
    stoch_sell = bool(stoch_sell)

    score_buy = int(macd_buy) + int(rsi_buy) + int(boll_buy) + int(golden_cross) + int(stoch_buy)
    score_sell = int(macd_sell) + int(rsi_sell) + int(boll_sell) + int(death_cross) + int(stoch_sell)

    if score_buy >= 3:
        recommendation = "Compra"
        color = "green"
    elif score_sell >= 3:
        recommendation = "Venta"
        color = "red"
    else:
        recommendation = "Mantener"
        color = "orange"

    reasons = []
    if macd_buy:
        reasons.append("MACD alcista")
    if macd_sell:
        reasons.append("MACD bajista")
    if rsi_buy:
        reasons.append("RSI en sobreventa")
    if rsi_sell:
        reasons.append("RSI en sobrecompra")
    if boll_buy:
        reasons.append("Precio por debajo de Bollinger inferior")
    if boll_sell:
        reasons.append("Precio por encima de Bollinger superior")
    if golden_cross:
        reasons.append("Golden cross")
    if death_cross:
        reasons.append("Death cross")
    if stoch_buy:
        reasons.append("Estocástico alcista en zona extrema")
    if stoch_sell:
        reasons.append("Estocástico bajista en zona extrema")

    return {
        "score_buy": score_buy,
        "score_sell": score_sell,
        "recommendation": recommendation,
        "color": color,
        "reasons": reasons,
        "details": {
            "macd_buy": macd_buy,
            "macd_sell": macd_sell,
            "rsi_buy": rsi_buy,
            "rsi_sell": rsi_sell,
            "boll_buy": boll_buy,
            "boll_sell": boll_sell,
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "stoch_buy": stoch_buy,
            "stoch_sell": stoch_sell,
        },
    }