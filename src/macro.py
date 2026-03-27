from __future__ import annotations

import io
import os
from typing import Dict

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import FRED_SERIES

load_dotenv()


def _build_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv,application/json,text/plain,*/*",
        }
    )
    return session


def _fred_csv_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def get_fred_series(series_id: str) -> pd.DataFrame:
    """
    Descarga serie desde FRED.
    Primero intenta API JSON con key.
    Si falla, usa CSV público.
    """
    session = _build_session()
    api_key = os.getenv("FRED_API_KEY", "").strip()

    # Intento 1: API JSON con key
    if api_key:
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
            }
            response = session.get(url, params=params, timeout=25)
            response.raise_for_status()
            obs = response.json()["observations"]
            df = pd.DataFrame(obs)
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            out = df[["date", "value"]].dropna()
            if not out.empty:
                return out
        except Exception as e:
            print(f"[FRED JSON] Error en {series_id}: {e}")

    # Intento 2: CSV público
    try:
        response = session.get(_fred_csv_url(series_id), timeout=25)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
        value_col = [c for c in df.columns if c != "date"][0]
        df["value"] = pd.to_numeric(df[value_col], errors="coerce")
        out = df[["date", "value"]].dropna()
        if not out.empty:
            return out
        print(f"[FRED CSV] {series_id} descargó, pero quedó vacío tras limpieza.")
        return pd.DataFrame(columns=["date", "value"])
    except Exception as e:
        print(f"[FRED CSV] Error en {series_id}: {e}")
        return pd.DataFrame(columns=["date", "value"])


def latest_value(df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float(df.dropna().iloc[-1]["value"])


def yoy_inflation(cpi_df: pd.DataFrame) -> float:
    if cpi_df.empty or len(cpi_df) < 13:
        return float("nan")
    c = cpi_df.set_index("date")["value"].sort_index()
    return float(c.pct_change(12).dropna().iloc[-1])


@st.cache_data(show_spinner=False, ttl=3600)
def macro_snapshot() -> Dict[str, float]:
    rf_df = get_fred_series(FRED_SERIES["risk_free_rate"])
    cpi_df = get_fred_series(FRED_SERIES["inflation"])
    cop_df = get_fred_series(FRED_SERIES["cop_usd"])

    return {
        "risk_free_rate_pct": latest_value(rf_df),
        "inflation_yoy": yoy_inflation(cpi_df),
        "cop_per_usd": latest_value(cop_df),
    }