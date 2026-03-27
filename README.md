# riesgo_dashboard

Proyecto integrador de **Teoría del Riesgo** construido en **Python + Streamlit multipage** para analizar un portafolio internacional de 5 activos:

- Seven & i Holdings -> `3382.T`
- Alimentation Couche-Tard -> `ATD.TO`
- FEMSA -> `FEMSAUBD.MX`
- BP -> `BP.L`
- Carrefour -> `CA.PA`

## Objetivo

Desarrollar un tablero interactivo para análisis de riesgo financiero de un portafolio con al menos 5 activos, integrando APIs y aplicando:

- análisis técnico,
- rendimientos,
- modelos ARCH/GARCH,
- CAPM y beta,
- VaR y CVaR,
- optimización de Markowitz,
- señales automáticas,
- contexto macroeconómico y benchmark.

## Estructura del proyecto

```text
riesgo_dashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── config.py
│   ├── download.py
│   ├── preprocess.py
│   ├── indicators.py
│   ├── returns_analysis.py
│   ├── garch_models.py
│   ├── capm.py
│   ├── risk_metrics.py
│   ├── markowitz.py
│   ├── signals.py
│   ├── macro.py
│   ├── benchmark.py
│   └── plots.py
│
├── pages/
│   ├── 01_tecnico.py
│   ├── 02_rendimientos.py
│   ├── 03_garch.py
│   ├── 04_capm.py
│   ├── 05_var_cvar.py
│   ├── 06_markowitz.py
│   ├── 07_senales.py
│   └── 08_macro_benchmark.py
│
└── report/
    └── informe_articulo.tex
```

## Instalación

1. Crear entorno virtual.

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS
```bash
python -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias.

```bash
pip install -r requirements.txt
```

3. Crear variables de entorno.

```bash
cp .env.example .env
```

4. Ejecutar la aplicación.

```bash
streamlit run app.py
```

## APIs utilizadas

### 1. Yahoo Finance
Se usa mediante `yfinance` para descargar precios históricos OHLCV.

### 2. FRED
Se usa para:
- tasa libre de riesgo: `DGS3MO`
- inflación: `CPIAUCSL`
- tipo de cambio Colombia (promedio mensual): `COLCCUSMA02STM`

## Notas técnicas

- El proyecto usa **caching** con Streamlit para evitar descargas repetidas.
- Los datos descargados se guardan también en:
  - `data/raw/`
  - `data/processed/`
- El portafolio base es **equiponderado**.
- El benchmark global por defecto es `ACWI`.
- Para CAPM se usa benchmark local por activo.

## Módulos implementados

### 1. Técnico
SMA, EMA, RSI, MACD, Bollinger y Estocástico.

### 2. Rendimientos
Rendimientos simples y logarítmicos, descriptivos, histograma, QQ-plot, boxplot y pruebas de normalidad.

### 3. ARCH/GARCH
ARCH(1), GARCH(1,1) y EGARCH(1,1), con comparación AIC/BIC y forecast.

### 4. CAPM
Beta, regresión activo-mercado y rendimiento esperado CAPM.

### 5. VaR y CVaR
- Paramétrico
- Histórico
- Monte Carlo
- CVaR

### 6. Markowitz
Simulación de 10,000 portafolios, frontera eficiente, mínima varianza y máximo Sharpe.

### 7. Señales
Reglas automáticas con MACD, RSI, Bollinger, medias móviles y Estocástico.

### 8. Macro y benchmark
Indicadores macro vía FRED y comparación del portafolio vs. benchmark global.

## Recomendaciones de uso

- Si FRED falla por conectividad, la app usa fallback a CSV público cuando es posible.
- Si un ticker tiene pocos datos válidos en el rango seleccionado, amplía la fecha inicial.
- Para GARCH, selecciona activos con suficiente longitud de serie.

## Uso de IA

Este proyecto puede haber sido asistido por IA para:
- estructuración del código,
- documentación,
- refactorización,
- explicación metodológica.

La responsabilidad de validación final, interpretación y presentación es del estudiante.

## Autora / equipo

Maria Paula Amaya y Edward Mora
