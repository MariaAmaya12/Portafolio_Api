import streamlit as st
import numpy as np

from src.config import ASSETS, DEFAULT_START_DATE, DEFAULT_END_DATE, ensure_project_dirs
from src.download import load_market_bundle
from src.preprocess import equal_weight_vector, equal_weight_portfolio
from src.risk_metrics import risk_comparison_table, kupiec_test
from src.plots import plot_var_distribution

ensure_project_dirs()

st.title("Módulo 5 - VaR y CVaR")
st.caption(
    "Evalúa el riesgo extremo del portafolio mediante VaR y CVaR bajo distintos enfoques de estimación."
)

# ==============================
# Sidebar
# ==============================
with st.sidebar:
    st.header("Parámetros de riesgo")
    start_date = st.date_input("Fecha inicial", value=DEFAULT_START_DATE, key="var_start")
    end_date = st.date_input("Fecha final", value=DEFAULT_END_DATE, key="var_end")
    alpha = st.selectbox("Nivel de confianza", [0.95, 0.99], index=0)
    n_sim = st.slider(
        "Simulaciones Monte Carlo",
        min_value=5000,
        max_value=50000,
        value=10000,
        step=5000,
    )

    st.divider()
    modo = st.radio(
        "Modo de visualización",
        ["General", "Estadístico"],
        index=0,
    )

    mostrar_tablas = st.checkbox("Mostrar tablas completas", value=False)

    mostrar_fundamento = False
    mostrar_interpretacion_tecnica = False
    if modo == "Estadístico":
        mostrar_fundamento = st.checkbox("Mostrar fundamento teórico", value=False)
        mostrar_interpretacion_tecnica = st.checkbox("Mostrar interpretación técnica", value=False)

# ==============================
# Carga y preparación de datos
# ==============================
tickers = [meta["ticker"] for meta in ASSETS.values()]
bundle = load_market_bundle(tickers=tickers, start=str(start_date), end=str(end_date))
returns = bundle["returns"].replace([np.inf, -np.inf], np.nan).dropna()

if returns.empty or len(returns) < 30:
    st.error("No hay suficientes datos para calcular métricas de riesgo.")
    st.stop()

weights = equal_weight_vector(returns.shape[1])
portfolio_returns = equal_weight_portfolio(returns)

table = risk_comparison_table(
    portfolio_returns=portfolio_returns,
    asset_returns_df=returns,
    weights=weights,
    alpha=alpha,
    n_sim=n_sim,
)

if table.empty:
    st.error("No fue posible calcular VaR y CVaR con los datos disponibles.")
    st.stop()

# ==============================
# Resumen del módulo
# ==============================
st.markdown("### Resumen del módulo")
if modo == "General":
    st.write(
        f"""
        Este módulo estima cuánto podría perder el portafolio en escenarios adversos con un nivel de confianza
        del **{int(alpha * 100)}%**. El VaR indica una pérdida umbral, mientras que el CVaR muestra qué tan severas
        serían en promedio las pérdidas más extremas.
        """
    )
else:
    st.write(
        f"""
        Este módulo compara el **Value at Risk (VaR)** y el **Conditional Value at Risk (CVaR)** del portafolio
        equiponderado bajo enfoques **paramétrico**, **histórico** y **Monte Carlo**, usando la convención de pérdidas
        positivas para un nivel de confianza de **{int(alpha * 100)}%**.
        """
    )

# ==============================
# Fundamento teórico
# ==============================
if modo == "Estadístico" and mostrar_fundamento:
    st.markdown("### Fundamento teórico")

    st.markdown(
        rf"""
        Sea $R_{{p,t}}$ el rendimiento del portafolio en el periodo \(t\), definido como combinación lineal
        de los rendimientos individuales de los activos según sus pesos. En este análisis, la pérdida del portafolio se define como:
        """
    )
    st.latex(r"L_t = -R_{p,t}")

    st.markdown(
        r"""
        De esta forma:

        - valores **positivos** de \(L_t\) representan **pérdidas**
        - valores **negativos** de \(L_t\) representan **ganancias**

        El **Value at Risk (VaR)** al nivel de confianza \(\alpha\) corresponde al cuantil de la distribución
        de pérdidas. En términos prácticos, representa la pérdida máxima esperada que no se excede con probabilidad \(\alpha\).

        El **Conditional Value at Risk (CVaR)**, también llamado **Expected Shortfall**, mide la pérdida promedio
        en los escenarios más extremos, es decir, cuando la pérdida supera el VaR.

        **Convención usada en este proyecto:**
        - El **VaR** y el **CVaR** se reportan como **pérdidas positivas**.
        - Por ejemplo, un VaR diario de **0.025** significa una pérdida potencial de **2.5%**.
        """
    )

    st.info(
        """
        **Interpretación de métodos**
        - **Paramétrico**: supone normalidad de los rendimientos del portafolio.
        - **Histórico**: usa la distribución empírica observada, sin imponer una distribución teórica.
        - **Monte Carlo**: simula escenarios futuros a partir de la media y la matriz de covarianza de los activos.
        """
    )

# ==============================
# Portafolio
# ==============================
st.markdown("### Portafolio analizado")
if modo == "General":
    st.write("Se utiliza un portafolio equiponderado, es decir, todos los activos tienen el mismo peso.")
else:
    with st.expander("Ver pesos del portafolio"):
        st.write("Pesos:", dict(zip(returns.columns, np.round(weights, 4))))

# ==============================
# KPIs
# ==============================
st.markdown("### KPIs de riesgo")

var_hist_row = table.loc[table["método"] == "Histórico"]
var_param_row = table.loc[table["método"] == "Paramétrico"]
var_mc_row = table.loc[table["método"] == "Monte Carlo"]

var_h = float(var_hist_row["VaR_diario"].iloc[0]) if not var_hist_row.empty else None
cvar_h = float(var_hist_row["CVaR_diario"].iloc[0]) if not var_hist_row.empty else None
var_p = float(var_param_row["VaR_diario"].iloc[0]) if not var_param_row.empty else None
var_mc = float(var_mc_row["VaR_diario"].iloc[0]) if not var_mc_row.empty else None

col1, col2, col3, col4 = st.columns(4)
col1.metric(f"VaR histórico {int(alpha * 100)}%", f"{var_h:.2%}" if var_h is not None else "N/D")
col2.metric(f"CVaR histórico {int(alpha * 100)}%", f"{cvar_h:.2%}" if cvar_h is not None else "N/D")
col3.metric("VaR paramétrico", f"{var_p:.2%}" if var_p is not None else "N/D")
col4.metric("VaR Monte Carlo", f"{var_mc:.2%}" if var_mc is not None else "N/D")

# ==============================
# Gráfico
# ==============================
st.markdown("### Distribución y riesgo extremo")
st.plotly_chart(plot_var_distribution(portfolio_returns, table), width="stretch")

if modo == "General":
    st.info(
        """
        **Cómo leer este gráfico**

        - El gráfico muestra cómo se distribuyen los rendimientos o pérdidas del portafolio.
        - Las líneas asociadas al VaR marcan umbrales de pérdida bajo distintos métodos.
        - Un CVaR más alto indica que, cuando ocurren eventos extremos, las pérdidas promedio son más severas.
        """
    )
else:
    with st.expander("Ver interpretación técnica del gráfico"):
        st.write(
            """
            El gráfico permite visualizar la distribución empírica del portafolio y contrastar los niveles de VaR y CVaR
            obtenidos bajo diferentes metodologías. La ubicación relativa de los umbrales facilita comparar conservadurismo
            y sensibilidad de cada método frente al riesgo extremo.
            """
        )

# ==============================
# Tabla de resultados
# ==============================
st.markdown("### Comparación VaR / CVaR")
if mostrar_tablas:
    st.dataframe(table, width="stretch")

# ==============================
# Interpretación
# ==============================
st.markdown("### Interpretación")

mensajes = []

if not var_hist_row.empty:
    mensajes.append(
        f"Con {int(alpha * 100)}% de confianza, el **VaR histórico diario** del portafolio es **{var_h:.2%}**, "
        f"mientras que el **CVaR histórico diario** es **{cvar_h:.2%}**."
    )

    mensajes.append(
        "Esto implica que, en escenarios de pérdida extrema, el promedio de pérdidas severas supera el umbral del VaR, "
        "lo cual es consistente con la interpretación del CVaR como medida más sensible al riesgo de cola."
    )

if not var_param_row.empty and not var_hist_row.empty:
    if var_p < var_h:
        mensajes.append(
            "El VaR paramétrico es menor que el VaR histórico, lo que puede sugerir que el supuesto de normalidad "
            "subestima el riesgo extremo frente a la evidencia empírica."
        )
    elif var_p > var_h:
        mensajes.append(
            "El VaR paramétrico es mayor que el VaR histórico, lo que sugiere una estimación más conservadora "
            "bajo el supuesto normal."
        )
    else:
        mensajes.append(
            "El VaR paramétrico y el VaR histórico son muy similares para esta muestra."
        )

if not var_mc_row.empty:
    mensajes.append(
        f"El **VaR Monte Carlo diario** estimado es **{var_mc:.2%}**, útil para contrastar "
        "la sensibilidad del riesgo ante simulaciones probabilísticas."
    )

if modo == "General":
    st.success(
        f"""
        **Lectura sencilla**
        - Con {int(alpha * 100)}% de confianza, la pérdida diaria del portafolio no superaría aproximadamente **{var_h:.2%}**.
        - Si ocurre un evento extremo que supera ese umbral, la pérdida promedio podría acercarse a **{cvar_h:.2%}**.
        - El CVaR es más severo que el VaR porque se enfoca en los peores escenarios.
        """
    )
else:
    if mostrar_interpretacion_tecnica:
        for msg in mensajes:
            st.info(msg)

        st.warning(
            "El VaR paramétrico depende del supuesto de normalidad. Si la distribución de rendimientos presenta colas pesadas "
            "o asimetría, este método puede subestimar el riesgo extremo."
        )

# ==============================
# Backtesting VaR - Test de Kupiec
# ==============================
st.markdown("### Backtesting VaR - Test de Kupiec")

if not var_hist_row.empty:
    kupiec = kupiec_test(
        returns=portfolio_returns,
        var=var_h,
        alpha=alpha,
    )

    if kupiec:
        col1, col2, col3 = st.columns(3)
        col1.metric("Violaciones", kupiec["violations"])
        col2.metric("Observadas (%)", f"{kupiec['observed_fail_rate'] * 100:.2f}%")
        col3.metric("Esperadas (%)", f"{kupiec['expected_fail_rate'] * 100:.2f}%")

        st.write(f"**p-value:** {kupiec['p_value']:.4f}")
        st.write(f"**Conclusión:** {kupiec['conclusion']}")

        if kupiec["p_value"] > 0.05:
            st.success(
                "El VaR histórico es consistente con la frecuencia de pérdidas observadas en la muestra."
            )
        else:
            st.error(
                "El VaR histórico no es consistente con la frecuencia de pérdidas observadas. "
                "Esto sugiere que el modelo puede estar subestimando o sobreestimando el riesgo."
            )

        if modo == "Estadístico":
            with st.expander("Ver explicación del test de Kupiec"):
                st.info(
                    "El test de Kupiec compara la proporción esperada de violaciones del VaR con la proporción observada. "
                    "Es una forma de evaluar si el modelo de riesgo está calibrado de manera razonable."
                )
        else:
            st.info(
                "Este bloque verifica si el VaR estimado fue razonable frente a las pérdidas realmente observadas."
            )
    else:
        st.warning("No fue posible ejecutar el test de Kupiec.")
else:
    st.warning("No hay VaR histórico disponible para ejecutar el test de Kupiec.")