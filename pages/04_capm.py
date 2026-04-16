import streamlit as st
import pandas as pd

from src.config import (
    ASSETS,
    DEFAULT_START_DATE,
    DEFAULT_END_DATE,
    get_ticker,
    get_local_benchmark,
    ensure_project_dirs,
)
from src.download import download_single_ticker
from src.returns_analysis import compute_return_series
from src.capm import compute_beta_and_capm
from src.api.macro import macro_snapshot
from src.plots import plot_scatter_regression

ensure_project_dirs()

st.title("Módulo 4 - CAPM y Beta")
st.caption("Evalúa sensibilidad al mercado, rendimiento esperado y riesgo sistemático del activo.")

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros CAPM")
    asset_name = st.selectbox("Activo", list(ASSETS.keys()), index=0)

    horizonte = st.selectbox(
        "Horizonte de análisis",
        [
            "1 mes",
            "Trimestre",
            "Semestre",
            "1 año",
            "3 años",
            "5 años",
            "Personalizado",
        ],
        index=3,
    )

    fecha_fin_ref = pd.to_datetime(DEFAULT_END_DATE)

    if horizonte == "1 mes":
        start_date = (fecha_fin_ref - pd.DateOffset(months=1)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "Trimestre":
        start_date = (fecha_fin_ref - pd.DateOffset(months=3)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "Semestre":
        start_date = (fecha_fin_ref - pd.DateOffset(months=6)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "1 año":
        start_date = (fecha_fin_ref - pd.DateOffset(years=1)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "3 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=3)).date()
        end_date = fecha_fin_ref.date()
    elif horizonte == "5 años":
        start_date = (fecha_fin_ref - pd.DateOffset(years=5)).date()
        end_date = fecha_fin_ref.date()
    else:
        start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="capm_start")
        end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="capm_end")

    st.divider()
    st.subheader("Modo de visualización")
    modo = st.radio(
        "Selecciona el nivel de detalle",
        ["General", "Estadístico"],
        index=0,
    )

    st.divider()
    st.subheader("Opciones de visualización")
    mostrar_tabla = st.checkbox("Mostrar tabla técnica", value=False)

    mostrar_interpretacion_tecnica = False
    if modo == "Estadístico":
        mostrar_interpretacion_tecnica = st.checkbox("Mostrar interpretación técnica", value=True)

# ==============================
# Datos
# ==============================
ticker = get_ticker(asset_name)
benchmark_ticker = get_local_benchmark(asset_name)

asset_df = download_single_ticker(ticker=ticker, start=str(start_date), end=str(end_date))
bench_df = download_single_ticker(ticker=benchmark_ticker, start=str(start_date), end=str(end_date))

if asset_df.empty or bench_df.empty:
    st.error("No se pudieron descargar los datos del activo o del benchmark.")
    st.stop()

asset_price = asset_df["Adj Close"] if "Adj Close" in asset_df.columns else asset_df["Close"]
bench_price = bench_df["Adj Close"] if "Adj Close" in bench_df.columns else bench_df["Close"]

asset_ret = compute_return_series(asset_price)["simple_return"]
bench_ret = compute_return_series(bench_price)["simple_return"]

macro = macro_snapshot()
rf_annual = (
    macro["risk_free_rate_pct"] / 100
    if macro["risk_free_rate_pct"] == macro["risk_free_rate_pct"]
    else 0.03
)

res = compute_beta_and_capm(asset_ret, bench_ret, rf_annual=rf_annual)

if not res:
    st.warning("No hay suficientes datos alineados para CAPM.")
    st.stop()

# ==============================
# Resumen
# ==============================
st.markdown("### Resumen del módulo")
if modo == "General":
    st.write(
        f"""
        Este módulo compara **{asset_name} ({ticker})** con su benchmark local **({benchmark_ticker})**
        para medir qué tan sensible es el activo a los movimientos del mercado y cuál sería su
        rendimiento esperado bajo CAPM.
        """
    )
else:
    st.write(
        f"""
        Este módulo estima la beta, el alpha diario, el ajuste de la regresión y el rendimiento esperado
        del activo **{asset_name} ({ticker})** frente al benchmark **{benchmark_ticker}**, bajo el marco
        del CAPM.
        """
    )

st.caption(f"Periodo analizado: {start_date} a {end_date}")

# ==============================
# KPIs
# ==============================
st.markdown("### KPIs CAPM")

beta = res.get("beta")
alpha_diaria = res.get("alpha_diaria")
r_squared = res.get("r_squared")
expected_return = res.get("expected_return_capm_annual")
classification = res.get("classification")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Beta", f"{beta:.4f}" if beta is not None else "N/D")
c2.metric("Alpha diaria", f"{alpha_diaria:.6f}" if alpha_diaria is not None else "N/D")
c3.metric("R²", f"{r_squared:.4f}" if r_squared is not None else "N/D")
c4.metric("Retorno esperado anual", f"{expected_return:.2%}" if expected_return is not None else "N/D")

st.markdown("### Clasificación del activo")
st.info(f"Clasificación: **{classification}**" if classification is not None else "Clasificación no disponible")

# ==============================
# Tabla técnica
# ==============================
summary_df = pd.DataFrame(
    {
        "metric": [
            "beta",
            "alpha_diaria",
            "r_squared",
            "p_value_beta",
            "expected_return_capm_annual",
            "classification",
        ],
        "value": [
            res["beta"],
            res["alpha_diaria"],
            res["r_squared"],
            res["p_value_beta"],
            res["expected_return_capm_annual"],
            res["classification"],
        ],
    }
)

summary_df["value"] = summary_df["value"].astype(str)

st.markdown("### Resumen técnico")
if mostrar_tabla:
    st.dataframe(summary_df, width="stretch")
else:
    with st.expander("Ver tabla técnica completa"):
        st.dataframe(summary_df, width="stretch")

# ==============================
# Gráfico
# ==============================
st.markdown("### Regresión CAPM")

fig = plot_scatter_regression(
    x=res["scatter_data"]["market_excess"],
    y=res["scatter_data"]["asset_excess"],
    yhat=res["regression_line"]["y"],
    title="Regresión CAPM",
)
st.plotly_chart(fig, width="stretch")

if modo == "General":
    st.info(
        """
        **Cómo leer este gráfico**

        - Cada punto representa una observación del activo frente al benchmark.
        - La línea resume la relación promedio entre ambos.
        - Si la pendiente es alta, el activo reacciona más fuerte a los movimientos del mercado.
        """
    )
else:
    with st.expander("Ver interpretación técnica del gráfico"):
        st.write(
            """
            El diagrama de dispersión muestra la relación entre el exceso de retorno del benchmark
            y el exceso de retorno del activo. La pendiente de la recta estimada corresponde a la beta,
            mientras que la dispersión alrededor de la recta se relaciona con el componente idiosincrático.
            """
        )

# ==============================
# Interpretación
# ==============================
st.markdown("### Interpretación")

if modo == "General":
    if beta is not None:
        if beta > 1:
            st.success(
                """
                **Lectura sencilla**
                - El activo se mueve con más intensidad que el mercado.
                - Cuando el mercado sube o baja, este activo tiende a amplificar ese movimiento.
                - Eso implica mayor sensibilidad y, en general, mayor riesgo sistemático.
                """
            )
        elif beta < 1:
            st.success(
                """
                **Lectura sencilla**
                - El activo se mueve con menor intensidad que el mercado.
                - Tiene un perfil más defensivo frente a cambios del benchmark.
                - Eso sugiere menor exposición al riesgo sistemático.
                """
            )
        else:
            st.success(
                """
                **Lectura sencilla**
                - El activo se mueve de forma parecida al mercado.
                - Su sensibilidad frente al benchmark es cercana a la del promedio del mercado.
                """
            )
else:
    if mostrar_interpretacion_tecnica:
        st.info(
            """
            **Interpretación económica del CAPM y la beta**

            - **Beta > 1**: el activo presenta mayor sensibilidad a los movimientos del mercado, por lo que su **riesgo sistemático** es superior al del benchmark.
            - **Beta < 1**: el activo muestra un comportamiento más defensivo y menor exposición al riesgo sistemático.
            - **Beta ≈ 1**: el activo tiende a moverse en línea con el mercado.
            - El **riesgo sistemático** es el componente del riesgo que no puede eliminarse mediante diversificación, porque depende de factores de mercado.
            - El **riesgo no sistemático** corresponde a factores propios del activo o de la firma y, en teoría, puede reducirse mediante diversificación.
            - En el CAPM, el rendimiento esperado remunera principalmente la exposición al **riesgo sistemático**, capturada por la beta.
            """
        )

# ==============================
# Contexto adicional
# ==============================
if modo == "Estadístico":
    with st.expander("Ver contexto de tasa libre de riesgo"):
        st.write(f"Tasa libre de riesgo anual usada en el cálculo: **{rf_annual:.2%}**")