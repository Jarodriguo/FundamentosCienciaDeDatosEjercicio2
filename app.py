"""
Dashboard de Energías Renovables — Inversión vs. Generación Diaria
Ejecutar con: streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Energías Renovables | Inversión vs. Generación",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta de color base (se asigna dinámicamente a las tecnologías que traiga
# el CSV cargado, así que funciona sin importar cómo se llamen o cuántas sean).
BASE_PALETTE = [
    "#1F6FB2", "#3FA796", "#F2B134", "#8C564B", "#C0392B",
    "#6A4C93", "#2E8B57", "#E07A5F", "#4C4C6D", "#B08968",
]


def build_tech_colors(tecnologias) -> dict:
    """Asigna un color estable a cada tecnología presente en los datos."""
    tecnologias = sorted(pd.unique(tecnologias))
    return {
        tech: BASE_PALETTE[i % len(BASE_PALETTE)]
        for i, tech in enumerate(tecnologias)
    }


sns.set_theme(style="whitegrid", font_scale=0.95)
plt.rcParams["axes.edgecolor"] = "#4a4a4a"
plt.rcParams["figure.facecolor"] = "white"

CUSTOM_CSS = """
<style>
    .stMetric {
        background-color: #f7f9fb;
        border: 1px solid #e6e9ec;
        border-radius: 10px;
        padding: 10px 6px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
    h1, h2, h3 { color: #163a5f; }
    .insight-box {
        background-color: #eef4fa;
        border-left: 5px solid #1F6FB2;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin: 0.6rem 0 1.1rem 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ────────────────────────────────────────────────────────────────────────────
REQUIRED_COLS = [
    "ID_Proyecto", "Tecnologia", "Operador", "Capacidad_Instalada_MW",
    "Generacion_Diaria_MWh", "Eficiencia_Planta_Pct", "Conectado_SIN",
    "Estado_Actual", "Inversion_Inicial_MUSD", "Fecha_Entrada_Operacion",
]


import os


@st.cache_data
def generate_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintético de proyectos de energía renovable en Colombia,
    con el esquema exacto solicitado. Se usa como dataset de ejemplo cuando el
    usuario no ha subido su propio CSV, y NO depende de ningún archivo externo,
    por lo que nunca falla en el despliegue (Streamlit Cloud, Docker, etc.).
    """
    rng = np.random.default_rng(seed)

    tech_params = {
        # (costo_musd_por_mw, factor_planta, participación en el portafolio)
        "Hidroeléctrica": {"costo_mw": (1.4, 1.9), "fp": (0.48, 0.62), "share": 0.28},
        "Eólica":         {"costo_mw": (1.2, 1.6), "fp": (0.28, 0.42), "share": 0.20},
        "Solar":          {"costo_mw": (0.7, 1.1), "fp": (0.16, 0.24), "share": 0.30},
        "Biomasa":        {"costo_mw": (2.0, 2.5), "fp": (0.52, 0.66), "share": 0.13},
        "Geotérmica":     {"costo_mw": (2.6, 3.2), "fp": (0.68, 0.80), "share": 0.09},
    }
    operadores = [
        "EPM", "Celsia", "ISAGEN", "AES Colombia", "Enel Colombia",
        "Ecopetrol Energía", "Air-e", "Emgesa", "Trina Solar Colombia",
        "Cubico Sustainable", "Grupo Energía Bogotá", "Vatia S.A.",
        "Fenoco Renovables", "Andina Green Power", "Zelestra Colombia",
    ]
    estados = ["Operativo", "En Construcción", "Mantenimiento", "Fuera de Servicio"]
    estado_probs = [0.68, 0.15, 0.11, 0.06]

    techs = list(tech_params.keys())
    tech_shares = [tech_params[t]["share"] for t in techs]
    tecnologia = rng.choice(techs, size=n, p=tech_shares)

    rows = []
    for i in range(n):
        tech = tecnologia[i]
        p = tech_params[tech]

        capacidad_mw = float(np.round(rng.lognormal(mean=np.log(35), sigma=0.75), 2))
        capacidad_mw = float(np.clip(capacidad_mw, 1.5, 400))

        costo_mw = max(rng.uniform(*p["costo_mw"]) * rng.normal(1.0, 0.12), 0.3)
        inversion_musd = round(capacidad_mw * costo_mw, 2)

        fp = np.clip(rng.normal(np.mean(p["fp"]), (p["fp"][1] - p["fp"][0]) / 3), 0.05, 0.92)
        generacion_diaria = round(capacidad_mw * 24 * fp, 2)
        eficiencia_pct = round(np.clip(fp * 100 * rng.normal(1.0, 0.05), 5, 98), 2)

        conectado_sin = bool(rng.choice([True, False], p=[0.86, 0.14]))
        estado = rng.choice(estados, p=estado_probs)
        operador = rng.choice(operadores)

        year, month, day = int(rng.integers(2005, 2026)), int(rng.integers(1, 13)), int(rng.integers(1, 28))

        rows.append({
            "ID_Proyecto": f"PROJ-{i+1:04d}",
            "Tecnologia": tech,
            "Operador": operador,
            "Capacidad_Instalada_MW": capacidad_mw,
            "Generacion_Diaria_MWh": generacion_diaria,
            "Eficiencia_Planta_Pct": eficiencia_pct,
            "Conectado_SIN": conectado_sin,
            "Estado_Actual": estado,
            "Inversion_Inicial_MUSD": inversion_musd,
            "Fecha_Entrada_Operacion": f"{year:04d}-{month:02d}-{day:02d}",
        })

    return pd.DataFrame(rows)


def find_bundled_csv() -> str | None:
    """Busca un CSV real incluido en el repo; si no existe, se usa el sintético."""
    for candidate in ("data/energia_renovable.csv", "energia_renovable.csv"):
        if os.path.exists(candidate):
            return candidate
    return None


TRUE_VALUES = {"true", "1", "1.0", "si", "sí", "yes", "y", "verdadero"}
FALSE_VALUES = {"false", "0", "0.0", "no", "n", "falso"}


def coerce_bool(series: pd.Series) -> pd.Series:
    """Convierte a booleano valores como 'Sí'/'No', '1'/'0', 'True'/'False', etc."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map(
        lambda v: True if v in TRUE_VALUES else (False if v in FALSE_VALUES else np.nan)
    )


@st.cache_data
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Fecha_Entrada_Operacion"] = pd.to_datetime(
        df["Fecha_Entrada_Operacion"], errors="coerce"
    )
    df["Anio_Entrada"] = df["Fecha_Entrada_Operacion"].dt.year
    df["Conectado_SIN"] = coerce_bool(df["Conectado_SIN"])
    # Métrica clave para la pregunta de negocio: cuántos MWh/día se generan
    # por cada millón de dólares invertido.
    df["Ratio_MWh_por_MUSD"] = (
        df["Generacion_Diaria_MWh"] / df["Inversion_Inicial_MUSD"]
    )
    df["Costo_por_MW_MUSD"] = df["Inversion_Inicial_MUSD"] / df["Capacidad_Instalada_MW"]
    return df


st.sidebar.title("⚡ Panel de control")
st.sidebar.markdown("**Paso 1 · Cargar datos**")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo CSV de proyectos", type=["csv"],
    help="Debe contener las columnas: " + ", ".join(REQUIRED_COLS),
)

data_source_msg = ""
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        missing = set(REQUIRED_COLS) - set(raw_df.columns)
        if missing:
            st.sidebar.error(f"Faltan columnas requeridas: {', '.join(missing)}")
            st.stop()
        data_source_msg = f"Usando archivo cargado: **{uploaded_file.name}**"
    except Exception as e:
        st.sidebar.error(f"No se pudo leer el archivo: {e}")
        st.stop()
else:
    bundled_csv = find_bundled_csv()
    if bundled_csv:
        raw_df = pd.read_csv(bundled_csv)
        data_source_msg = f"Usando dataset incluido en el repositorio (`{bundled_csv}`)."
    else:
        raw_df = generate_synthetic_data()
        data_source_msg = (
            "Usando dataset de ejemplo generado automáticamente (no se encontró ningún CSV "
            "en el repositorio). Sube tu archivo real desde el cargador de arriba."
        )

st.sidebar.caption(data_source_msg)

df = process_data(raw_df)

# Se construye a partir de las tecnologías reales del CSV cargado (propio o de ejemplo),
# así nunca falla por nombres de tecnología distintos a los del dataset de ejemplo.
TECH_COLORS = build_tech_colors(df["Tecnologia"])

# ────────────────────────────────────────────────────────────────────────────
# FILTROS (sidebar)
# ────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros**")

tecnologias_sel = st.sidebar.multiselect(
    "Tecnología", sorted(df["Tecnologia"].unique()), default=sorted(df["Tecnologia"].unique())
)
operadores_sel = st.sidebar.multiselect(
    "Operador", sorted(df["Operador"].unique()), default=sorted(df["Operador"].unique())
)
estado_sel = st.sidebar.multiselect(
    "Estado actual", sorted(df["Estado_Actual"].unique()), default=sorted(df["Estado_Actual"].unique())
)
conectado_sel = st.sidebar.radio(
    "Conexión al SIN", ["Todos", "Solo conectados", "Solo no conectados"], index=0
)

cap_min, cap_max = float(df["Capacidad_Instalada_MW"].min()), float(df["Capacidad_Instalada_MW"].max())
cap_range = st.sidebar.slider(
    "Capacidad instalada (MW)", min_value=round(cap_min, 1), max_value=round(cap_max, 1),
    value=(round(cap_min, 1), round(cap_max, 1)),
)

mask = (
    df["Tecnologia"].isin(tecnologias_sel)
    & df["Operador"].isin(operadores_sel)
    & df["Estado_Actual"].isin(estado_sel)
    & df["Capacidad_Instalada_MW"].between(cap_range[0], cap_range[1])
)
if conectado_sel == "Solo conectados":
    mask &= df["Conectado_SIN"] == True
elif conectado_sel == "Solo no conectados":
    mask &= df["Conectado_SIN"] == False

fdf = df.loc[mask].copy()

if fdf.empty:
    st.warning("No hay proyectos que coincidan con los filtros seleccionados. Ajusta los filtros en la barra lateral.")
    st.stop()

# ────────────────────────────────────────────────────────────────────────────
# ENCABEZADO
# ────────────────────────────────────────────────────────────────────────────
st.title("⚡ Energías Renovables: Inversión vs. Generación Diaria")
st.markdown(
    "##### Pregunta de negocio: *¿Qué tecnología tiene la mejor relación entre "
    "inversión inicial y generación diaria de energía?*"
)
st.markdown(
    f"Analizando **{len(fdf):,}** proyectos de energía renovable "
    f"(de un total de {len(df):,} en el dataset cargado)."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Proyectos analizados", f"{len(fdf):,}")
k2.metric("Capacidad total", f"{fdf['Capacidad_Instalada_MW'].sum():,.0f} MW")
k3.metric("Inversión total", f"${fdf['Inversion_Inicial_MUSD'].sum():,.0f} MUSD")
k4.metric("Generación diaria total", f"{fdf['Generacion_Diaria_MWh'].sum():,.0f} MWh")
k5.metric("Eficiencia media planta", f"{fdf['Eficiencia_Planta_Pct'].mean():,.1f}%")

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────
# 2. EDA
# ────────────────────────────────────────────────────────────────────────────
st.header("🔍 1. Exploración de los datos (EDA)")
st.markdown(
    "Antes de responder la pregunta de negocio, entendamos la estructura, calidad "
    "y distribución del dataset."
)

eda_tab1, eda_tab2, eda_tab3 = st.tabs(["Vista general", "Distribuciones", "Relaciones"])

with eda_tab1:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("**Muestra de los datos**")
        st.dataframe(fdf.drop(columns=["Anio_Entrada", "Ratio_MWh_por_MUSD", "Costo_por_MW_MUSD"]).head(10),
                     use_container_width=True)
    with c2:
        st.markdown("**Calidad de datos**")
        nulos = df[REQUIRED_COLS].isna().sum()
        calidad_df = pd.DataFrame({
            "Columna": nulos.index,
            "Valores nulos": nulos.values,
            "Tipo de dato": [df[c].dtype for c in nulos.index],
        })
        st.dataframe(calidad_df, use_container_width=True, hide_index=True)
        st.caption(f"Total de registros en el dataset original: {len(df):,} · Sin valores nulos detectados." if nulos.sum() == 0
                   else f"Se detectaron {int(nulos.sum())} valores nulos en total.")

    st.markdown("**Estadística descriptiva (variables numéricas)**")
    st.dataframe(
        fdf[["Capacidad_Instalada_MW", "Generacion_Diaria_MWh", "Eficiencia_Planta_Pct",
             "Inversion_Inicial_MUSD"]].describe().round(2),
        use_container_width=True,
    )

with eda_tab2:
    st.markdown("**Distribución de las variables numéricas clave**")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    dist_vars = [
        ("Capacidad_Instalada_MW", "Capacidad Instalada (MW)"),
        ("Inversion_Inicial_MUSD", "Inversión Inicial (MUSD)"),
        ("Generacion_Diaria_MWh", "Generación Diaria (MWh)"),
        ("Eficiencia_Planta_Pct", "Eficiencia de Planta (%)"),
    ]
    for ax, (col, label) in zip(axes.flat, dist_vars):
        sns.histplot(fdf[col], kde=True, ax=ax, color="#1F6FB2", edgecolor="white")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(
        "La capacidad, inversión y generación muestran asimetría positiva (pocos proyectos muy grandes "
        "elevan la cola derecha), típico de portafolios de infraestructura energética."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Cantidad de proyectos por tecnología**")
        fig2, ax2 = plt.subplots(figsize=(5.5, 4))
        order = fdf["Tecnologia"].value_counts().index
        sns.countplot(
            data=fdf, y="Tecnologia", order=order, ax=ax2,
            palette=[TECH_COLORS[t] for t in order],
        )
        ax2.set_xlabel("N.º de proyectos")
        ax2.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig2)
    with c2:
        st.markdown("**Capacidad instalada (MW) por tecnología**")
        fig3, ax3 = plt.subplots(figsize=(5.5, 4))
        sns.boxplot(
            data=fdf, x="Tecnologia", y="Capacidad_Instalada_MW", ax=ax3,
            hue="Tecnologia", palette=TECH_COLORS, legend=False,
        )
        ax3.set_xlabel("")
        ax3.set_ylabel("MW")
        ax3.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        st.pyplot(fig3)

with eda_tab3:
    st.markdown("**Matriz de correlación entre variables numéricas**")
    num_cols = ["Capacidad_Instalada_MW", "Generacion_Diaria_MWh", "Eficiencia_Planta_Pct",
                "Inversion_Inicial_MUSD"]
    corr = fdf[num_cols].corr()
    fig4, ax4 = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax4, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8})
    plt.tight_layout()
    st.pyplot(fig4)
    st.markdown(
        "<div class='insight-box'>📌 <b>Lectura rápida:</b> la capacidad instalada está fuertemente "
        "correlacionada con la inversión y con la generación diaria (a mayor tamaño de planta, mayor "
        "costo y mayor producción). Sin embargo, esta correlación <b>no nos dice qué tecnología es más "
        "eficiente por dólar invertido</b> — para eso necesitamos normalizar por inversión, lo que "
        "hacemos en la siguiente sección.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────
# 3. VISUALIZACIÓN CLAVE — Capacidad instalada por operador
# ────────────────────────────────────────────────────────────────────────────
st.header("🏭 2. Capacidad Instalada por Operador")
st.markdown(
    "Esta es la visualización central del dashboard: muestra cómo se distribuye "
    "la capacidad instalada (MW) entre los distintos operadores del sistema."
)

cap_operador = (
    fdf.groupby("Operador", as_index=False)["Capacidad_Instalada_MW"]
    .sum()
    .sort_values("Capacidad_Instalada_MW", ascending=True)
)

fig_operador = px.bar(
    cap_operador,
    x="Capacidad_Instalada_MW",
    y="Operador",
    orientation="h",
    text="Capacidad_Instalada_MW",
    color="Capacidad_Instalada_MW",
    color_continuous_scale="Blues",
    labels={"Capacidad_Instalada_MW": "Capacidad Instalada (MW)", "Operador": ""},
    title="Capacidad Instalada Total por Operador (MW)",
)
fig_operador.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
fig_operador.update_layout(
    height=560, coloraxis_showscale=False,
    title_font_size=16, margin=dict(l=10, r=10, t=60, b=10),
)
st.plotly_chart(fig_operador, use_container_width=True)

top_operador = cap_operador.iloc[-1]
st.markdown(
    f"<div class='insight-box'>📌 <b>{top_operador['Operador']}</b> lidera el sistema con "
    f"<b>{top_operador['Capacidad_Instalada_MW']:,.0f} MW</b> de capacidad instalada dentro de los "
    f"proyectos filtrados, seguido de cerca por otros operadores relevantes. Esta concentración importa "
    f"porque las decisiones de inversión en tecnología de estos operadores grandes tienen mayor peso "
    f"sobre la eficiencia global del sistema.</div>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────
# 4. RESPUESTA A LA PREGUNTA DE NEGOCIO
# ────────────────────────────────────────────────────────────────────────────
st.header("💡 3. ¿Qué tecnología tiene la mejor relación Inversión vs. Generación Diaria?")

st.markdown(
    "Para responder esto no basta con mirar la generación o la inversión por separado: "
    "necesitamos una métrica que las combine. Definimos:"
)
st.latex(r"\text{Ratio (MWh/día por MUSD)} = \frac{\text{Generación Diaria (MWh)}}{\text{Inversión Inicial (MUSD)}}")
st.markdown(
    "Cuanto **más alto** es este ratio, **más energía diaria produce cada millón de dólares invertido** "
    "— es decir, mejor retorno energético por dólar de capital."
)

c1, c2 = st.columns([1.15, 1])

with c1:
    st.markdown("**Inversión vs. Generación Diaria por proyecto**")
    fig_scatter = px.scatter(
        fdf, x="Inversion_Inicial_MUSD", y="Generacion_Diaria_MWh",
        color="Tecnologia", size="Capacidad_Instalada_MW",
        color_discrete_map=TECH_COLORS,
        hover_data=["Operador", "Estado_Actual", "Eficiencia_Planta_Pct"],
        labels={
            "Inversion_Inicial_MUSD": "Inversión Inicial (MUSD)",
            "Generacion_Diaria_MWh": "Generación Diaria (MWh)",
            "Tecnologia": "Tecnología",
        },
        opacity=0.75,
    )
    fig_scatter.update_layout(height=460, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption(
        "El tamaño de cada punto representa la capacidad instalada (MW). Las tecnologías cuyos puntos "
        "se ubican más 'arriba a la izquierda' generan más MWh por cada dólar invertido."
    )

with c2:
    st.markdown("**Ratio promedio (MWh/día por MUSD) por tecnología**")
    ratio_tech = (
        fdf.groupby("Tecnologia")
        .apply(lambda g: g["Generacion_Diaria_MWh"].sum() / g["Inversion_Inicial_MUSD"].sum())
        .reset_index(name="Ratio_MWh_por_MUSD")
        .sort_values("Ratio_MWh_por_MUSD", ascending=False)
    )
    fig_ratio = px.bar(
        ratio_tech, x="Ratio_MWh_por_MUSD", y="Tecnologia", orientation="h",
        color="Tecnologia", color_discrete_map=TECH_COLORS, text="Ratio_MWh_por_MUSD",
        labels={"Ratio_MWh_por_MUSD": "MWh/día por MUSD invertido", "Tecnologia": ""},
    )
    fig_ratio.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_ratio.update_layout(height=380, showlegend=False, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_ratio, use_container_width=True)

    fig_eff, axe = plt.subplots(figsize=(5.3, 3.2))
    order_eff = fdf.groupby("Tecnologia")["Eficiencia_Planta_Pct"].mean().sort_values(ascending=False)
    sns.barplot(x=order_eff.values, y=order_eff.index, ax=axe,
                hue=order_eff.index, palette=TECH_COLORS, legend=False)
    axe.set_xlabel("Eficiencia media de planta (%)")
    axe.set_ylabel("")
    axe.set_title("Eficiencia media por tecnología", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig_eff)

mejor_tech = ratio_tech.iloc[0]
segunda_tech = ratio_tech.iloc[1]
diferencia_pct = (mejor_tech["Ratio_MWh_por_MUSD"] / segunda_tech["Ratio_MWh_por_MUSD"] - 1) * 100

st.markdown(
    f"""
<div class='insight-box'>
✅ <b>Respuesta a la pregunta de negocio:</b> con los datos filtrados actuales, la tecnología con la mejor
relación inversión–generación es <b>{mejor_tech['Tecnologia']}</b>, con
<b>{mejor_tech['Ratio_MWh_por_MUSD']:.2f} MWh/día generados por cada MUSD invertido</b> — un
{diferencia_pct:.0f}% más eficiente en capital que la segunda mejor opción,
<b>{segunda_tech['Tecnologia']}</b> ({segunda_tech['Ratio_MWh_por_MUSD']:.2f} MWh/día por MUSD).
Esto se explica principalmente porque combina un costo de inversión por MW moderado con un
<b>factor de planta (eficiencia) alto y estable</b>, lo que le permite traducir cada dólar invertido en
más energía entregada de forma consistente.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────
# 5. ANÁLISIS COMPLEMENTARIO
# ────────────────────────────────────────────────────────────────────────────
st.header("📈 4. Análisis complementario")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Evolución de proyectos entrados en operación por año**")
    evo = fdf.dropna(subset=["Anio_Entrada"]).groupby(["Anio_Entrada", "Tecnologia"]).size().reset_index(name="Proyectos")
    fig_evo = px.area(
        evo, x="Anio_Entrada", y="Proyectos", color="Tecnologia",
        color_discrete_map=TECH_COLORS,
        labels={"Anio_Entrada": "Año de entrada en operación", "Proyectos": "N.º de proyectos"},
    )
    fig_evo.update_layout(height=380, legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig_evo, use_container_width=True)

with c2:
    st.markdown("**Conexión al Sistema Interconectado Nacional (SIN)**")
    sin_counts = fdf["Conectado_SIN"].map({True: "Conectado", False: "No conectado"}).value_counts().reset_index()
    sin_counts.columns = ["Estado", "Proyectos"]
    fig_sin = px.pie(
        sin_counts, names="Estado", values="Proyectos", hole=0.55,
        color="Estado", color_discrete_map={"Conectado": "#1F6FB2", "No conectado": "#C0392B"},
    )
    fig_sin.update_traces(textinfo="percent+label")
    fig_sin.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_sin, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown("**Estado actual de los proyectos**")
    fig_estado, axf = plt.subplots(figsize=(5.5, 3.6))
    estado_order = fdf["Estado_Actual"].value_counts().index
    sns.countplot(data=fdf, x="Estado_Actual", order=estado_order, ax=axf, color="#3FA796")
    axf.set_xlabel("")
    axf.set_ylabel("N.º de proyectos")
    plt.tight_layout()
    st.pyplot(fig_estado)

with c4:
    st.markdown("**Costo de inversión por MW instalado (MUSD/MW)**")
    fig_costo, axc = plt.subplots(figsize=(5.5, 3.6))
    order_costo = fdf.groupby("Tecnologia")["Costo_por_MW_MUSD"].mean().sort_values()
    sns.barplot(x=order_costo.values, y=order_costo.index, ax=axc,
                hue=order_costo.index, palette=TECH_COLORS, legend=False)
    axc.set_xlabel("MUSD por MW")
    axc.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig_costo)

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────
# 6. CONCLUSIONES
# ────────────────────────────────────────────────────────────────────────────
st.header("📝 5. Conclusiones")
st.markdown(
    f"""
- **{mejor_tech['Tecnologia']}** ofrece hoy la mejor relación entre inversión inicial y generación
  diaria de energía dentro del portafolio analizado, gracias a una combinación favorable de costo por
  MW y eficiencia de planta.
- La **capacidad instalada por operador** revela que el sistema está relativamente concentrado en pocos
  operadores grandes, lo que hace que sus decisiones de tecnología tengan un impacto desproporcionado
  en la eficiencia general del sistema.
- La correlación entre capacidad, inversión y generación es alta, pero **no es sinónimo de eficiencia**:
  una planta más grande no necesariamente genera más energía por dólar invertido, de ahí la importancia
  de mirar el ratio normalizado en lugar de los totales absolutos.
- Estos hallazgos pueden apoyar decisiones de **priorización de nuevos proyectos**, orientando el capital
  hacia las tecnologías que maximizan la generación por dólar invertido, sin dejar de lado consideraciones
  de diversificación de la matriz energética y disponibilidad de recursos naturales por región.
"""
)

st.caption(
    "Dashboard construido con Streamlit, Pandas, Matplotlib, Seaborn y Plotly · "
    "Usa el cargador de la barra lateral para analizar tu propio archivo CSV."
)
