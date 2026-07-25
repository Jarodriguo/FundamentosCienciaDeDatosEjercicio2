"""
Dashboard Inteligente de Energías Renovables
Motor de interpretación: Groq API + Llama 3.3 70B (llama-3.3-70b-versatile)

- Todas las gráficas se generan con Plotly.
- Las interpretaciones/narrativas de cada gráfica las escribe el modelo, con base
  en un resumen numérico real de los datos filtrados (no inventa cifras).
- Incluye un chat para conversar libremente con tus datos.

Ejecutar con: streamlit run app.py
La API Key de Groq se ingresa en la barra lateral (no se guarda en el código).
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from groq import Groq, APIError, AuthenticationError, RateLimitError

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ────────────────────────────────────────────────────────────────────────────
MODEL_ID = "llama-3.3-70b-versatile"

st.set_page_config(
    page_title="Dashboard Inteligente | Energías Renovables",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_PALETTE = [
    "#1F6FB2", "#3FA796", "#F2B134", "#8C564B", "#C0392B",
    "#6A4C93", "#2E8B57", "#E07A5F", "#4C4C6D", "#B08968",
]


def build_tech_colors(tecnologias) -> dict:
    tecnologias = sorted(pd.unique(tecnologias))
    return {t: BASE_PALETTE[i % len(BASE_PALETTE)] for i, t in enumerate(tecnologias)}




CUSTOM_CSS = """
<style>
    div[data-testid="stMetric"] {
        background-color: #f7f9fb;
        border: 1px solid #e6e9ec;
        border-radius: 10px;
        padding: 10px 6px;
    }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; color: #163a5f !important; }
    div[data-testid="stMetricLabel"] p { color: #4a4a4a !important; }
    div[data-testid="stMetricDelta"] { color: #163a5f !important; }

    .ai-box, .ai-box * { color: #12263f !important; }
    .ai-box {
        background-color: #eef7f0;
        border-left: 5px solid #2E8B57;
        padding: 0.9rem 1.1rem;
        border-radius: 6px;
        margin: 0.6rem 0 0.3rem 0;
    }
    .ai-tag {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        color: #2E8B57 !important;
        margin-bottom: 0.35rem;
        letter-spacing: 0.03em;
    }
    .warn-box, .warn-box * { color: #5c3d00 !important; }
    .warn-box {
        background-color: #fff6e5;
        border-left: 5px solid #F2B134;
        padding: 0.7rem 1rem;
        border-radius: 6px;
        margin: 0.4rem 0 1rem 0;
        font-size: 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS (sin depender de ningún archivo externo)
# ────────────────────────────────────────────────────────────────────────────
REQUIRED_COLS = [
    "ID_Proyecto", "Tecnologia", "Operador", "Capacidad_Instalada_MW",
    "Generacion_Diaria_MWh", "Eficiencia_Planta_Pct", "Conectado_SIN",
    "Estado_Actual", "Inversion_Inicial_MUSD", "Fecha_Entrada_Operacion",
]


@st.cache_data
def generate_synthetic_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Dataset de ejemplo generado en memoria (no depende de ningún CSV externo)."""
    rng = np.random.default_rng(seed)
    tech_params = {
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
        capacidad_mw = float(np.clip(np.round(rng.lognormal(mean=np.log(35), sigma=0.75), 2), 1.5, 400))
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
            "ID_Proyecto": f"PROJ-{i+1:04d}", "Tecnologia": tech, "Operador": operador,
            "Capacidad_Instalada_MW": capacidad_mw, "Generacion_Diaria_MWh": generacion_diaria,
            "Eficiencia_Planta_Pct": eficiencia_pct, "Conectado_SIN": conectado_sin,
            "Estado_Actual": estado, "Inversion_Inicial_MUSD": inversion_musd,
            "Fecha_Entrada_Operacion": f"{year:04d}-{month:02d}-{day:02d}",
        })
    return pd.DataFrame(rows)


def find_bundled_csv() -> str | None:
    for candidate in ("data/energia_renovable.csv", "energia_renovable.csv"):
        if os.path.exists(candidate):
            return candidate
    return None


TRUE_VALUES = {"true", "1", "1.0", "si", "sí", "yes", "y", "verdadero"}
FALSE_VALUES = {"false", "0", "0.0", "no", "n", "falso"}


def coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().map(
        lambda v: True if v in TRUE_VALUES else (False if v in FALSE_VALUES else np.nan)
    )


@st.cache_data
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Fecha_Entrada_Operacion"] = pd.to_datetime(df["Fecha_Entrada_Operacion"], errors="coerce")
    df["Anio_Entrada"] = df["Fecha_Entrada_Operacion"].dt.year
    df["Conectado_SIN"] = coerce_bool(df["Conectado_SIN"])
    df["Ratio_MWh_por_MUSD"] = df["Generacion_Diaria_MWh"] / df["Inversion_Inicial_MUSD"]
    df["Costo_por_MW_MUSD"] = df["Inversion_Inicial_MUSD"] / df["Capacidad_Instalada_MW"]
    return df


# ────────────────────────────────────────────────────────────────────────────
# BARRA LATERAL — datos, filtros y configuración de IA
# ────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Panel de control")

st.sidebar.markdown("**Datos**")
uploaded_file = st.sidebar.file_uploader(
    "Sube tu CSV de proyectos", type=["csv"],
    help="Debe contener las columnas: " + ", ".join(REQUIRED_COLS),
)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    missing = set(REQUIRED_COLS) - set(raw_df.columns)
    if missing:
        st.sidebar.error(f"Faltan columnas requeridas: {', '.join(missing)}")
        st.stop()
    data_source_msg = f"Archivo cargado: **{uploaded_file.name}**"
else:
    bundled_csv = find_bundled_csv()
    if bundled_csv:
        raw_df = pd.read_csv(bundled_csv)
        data_source_msg = f"Dataset incluido en el repo (`{bundled_csv}`)"
    else:
        raw_df = generate_synthetic_data()
        data_source_msg = "Dataset de ejemplo generado automáticamente"
st.sidebar.caption(data_source_msg)

df = process_data(raw_df)
TECH_COLORS = build_tech_colors(df["Tecnologia"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Filtros**")
tecnologias_sel = st.sidebar.multiselect("Tecnología", sorted(df["Tecnologia"].unique()), default=sorted(df["Tecnologia"].unique()))
operadores_sel = st.sidebar.multiselect("Operador", sorted(df["Operador"].unique()), default=sorted(df["Operador"].unique()))
estado_sel = st.sidebar.multiselect("Estado actual", sorted(df["Estado_Actual"].unique()), default=sorted(df["Estado_Actual"].unique()))

mask = (
    df["Tecnologia"].isin(tecnologias_sel)
    & df["Operador"].isin(operadores_sel)
    & df["Estado_Actual"].isin(estado_sel)
)
fdf = df.loc[mask].copy()
if fdf.empty:
    st.warning("No hay proyectos que coincidan con los filtros seleccionados.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 Motor de IA (Groq)**")
api_key = st.sidebar.text_input(
    "Groq API Key", type="password", placeholder="gsk_...",
    help="Consíguela gratis en https://console.groq.com/keys. No se guarda en ningún archivo.",
)
ai_temperature = st.sidebar.slider("Creatividad de las interpretaciones", 0.0, 1.0, 0.4, 0.1)
if st.sidebar.button("🔄 Regenerar interpretaciones de IA", use_container_width=True):
    st.session_state.pop("ai_cache", None)
if st.sidebar.button("🗑️ Reiniciar chat", use_container_width=True):
    st.session_state["chat_messages"] = []
st.sidebar.caption(
    f"Modelo: `{MODEL_ID}` (Llama 3.3 70B vía Groq). ⚠️ Groq anunció su retiro para el "
    "16/08/2026 → reemplazo recomendado: `openai/gpt-oss-120b`."
)


# ────────────────────────────────────────────────────────────────────────────
# CAPA DE IA: resumen de datos + generación de texto grounded en los números reales
# ────────────────────────────────────────────────────────────────────────────
def build_data_summary(full_df: pd.DataFrame, filt_df: pd.DataFrame) -> dict:
    ratio_tech = (
        filt_df.groupby("Tecnologia")
        .apply(lambda g: g["Generacion_Diaria_MWh"].sum() / g["Inversion_Inicial_MUSD"].sum())
        .sort_values(ascending=False)
    )
    cap_operador = filt_df.groupby("Operador")["Capacidad_Instalada_MW"].sum().sort_values(ascending=False)
    eficiencia_tech = filt_df.groupby("Tecnologia")["Eficiencia_Planta_Pct"].mean().sort_values(ascending=False)
    costo_tech = filt_df.groupby("Tecnologia")["Costo_por_MW_MUSD"].mean().sort_values()

    return {
        "n_proyectos_filtrados": int(len(filt_df)),
        "n_proyectos_total": int(len(full_df)),
        "capacidad_total_mw": round(float(filt_df["Capacidad_Instalada_MW"].sum()), 1),
        "inversion_total_musd": round(float(filt_df["Inversion_Inicial_MUSD"].sum()), 1),
        "generacion_diaria_total_mwh": round(float(filt_df["Generacion_Diaria_MWh"].sum()), 1),
        "eficiencia_media_pct": round(float(filt_df["Eficiencia_Planta_Pct"].mean()), 2),
        "pct_conectado_sin": round(float(filt_df["Conectado_SIN"].mean()) * 100, 1),
        "ratio_mwh_dia_por_musd_por_tecnologia": {k: round(v, 2) for k, v in ratio_tech.items()},
        "capacidad_mw_por_operador": {k: round(v, 1) for k, v in cap_operador.items()},
        "eficiencia_media_pct_por_tecnologia": {k: round(v, 2) for k, v in eficiencia_tech.items()},
        "costo_musd_por_mw_por_tecnologia": {k: round(v, 3) for k, v in costo_tech.items()},
        "conteo_por_estado": {k: int(v) for k, v in filt_df["Estado_Actual"].value_counts().items()},
    }


def summary_hash(summary: dict) -> str:
    return hashlib.md5(json.dumps(summary, sort_keys=True).encode()).hexdigest()


def get_groq_client() -> Groq:
    return Groq(api_key=api_key)


def ask_ai(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    """Llamada simple (no streaming) usada para las interpretaciones auto-generadas."""
    client = get_groq_client()
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


def render_ai_insight(section_key: str, system_prompt: str, user_prompt: str, summary: dict):
    """Genera (y cachea en session_state) una interpretación de IA para una sección del dashboard."""
    if "ai_cache" not in st.session_state:
        st.session_state["ai_cache"] = {}

    h = summary_hash(summary) + f"|{ai_temperature}"
    cached = st.session_state["ai_cache"].get(section_key)

    if not api_key:
        st.markdown(
            "<div class='warn-box'>🔒 Ingresa tu Groq API Key en la barra lateral para que la IA genere "
            "aquí una interpretación de esta gráfica basada en los datos filtrados.</div>",
            unsafe_allow_html=True,
        )
        return

    if cached and cached["hash"] == h:
        text = cached["text"]
    else:
        with st.spinner("🤖 Generando interpretación con Llama 3.3 70B..."):
            try:
                text = ask_ai(system_prompt, user_prompt, ai_temperature)
                st.session_state["ai_cache"][section_key] = {"hash": h, "text": text}
            except AuthenticationError:
                st.error("🔑 API Key inválida. Verifica que la copiaste correctamente.")
                return
            except RateLimitError:
                st.error("⏳ Límite de solicitudes de Groq alcanzado. Espera un momento e inténtalo de nuevo.")
                return
            except APIError as e:
                st.error(f"❌ Error de la API de Groq: {e}")
                return

    st.markdown(
        f"<div class='ai-box'><span class='ai-tag'>✨ INTERPRETACIÓN GENERADA POR IA (Llama 3.3 70B)</span>"
        f"<br>{text}</div>",
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# ENCABEZADO Y KPIs
# ────────────────────────────────────────────────────────────────────────────
st.title("🤖 Dashboard Inteligente de Energías Renovables")
st.markdown(
    "##### ¿Qué tecnología tiene la mejor relación entre inversión inicial y generación diaria? "
    "— con interpretaciones y chat generados en tiempo real por IA"
)
st.caption(f"Analizando **{len(fdf):,}** proyectos de {len(df):,} en total · {data_source_msg}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Proyectos", f"{len(fdf):,}")
k2.metric("Capacidad total", f"{fdf['Capacidad_Instalada_MW'].sum():,.0f} MW")
k3.metric("Inversión total", f"${fdf['Inversion_Inicial_MUSD'].sum():,.0f} MUSD")
k4.metric("Generación diaria", f"{fdf['Generacion_Diaria_MWh'].sum():,.0f} MWh")
k5.metric("Eficiencia media", f"{fdf['Eficiencia_Planta_Pct'].mean():,.1f}%")

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────
# 2. EDA RÁPIDO
# ────────────────────────────────────────────────────────────────────────────
st.header("🔍 Exploración de los datos")
c1, c2 = st.columns([1.3, 1])
with c1:
    st.markdown("**Muestra de los datos filtrados**")
    st.dataframe(fdf[REQUIRED_COLS].head(8), use_container_width=True)
with c2:
    st.markdown("**Estadística descriptiva**")
    st.dataframe(
        fdf[["Capacidad_Instalada_MW", "Generacion_Diaria_MWh", "Inversion_Inicial_MUSD"]].describe().round(1),
        use_container_width=True,
    )

st.markdown("**Distribuciones (Plotly)**")
dist_vars = [
    ("Capacidad_Instalada_MW", "Capacidad Instalada (MW)"),
    ("Inversion_Inicial_MUSD", "Inversión Inicial (MUSD)"),
    ("Generacion_Diaria_MWh", "Generación Diaria (MWh)"),
]
fig_dist = make_subplots(rows=1, cols=3, subplot_titles=[label for _, label in dist_vars])
for i, (col, label) in enumerate(dist_vars, start=1):
    fig_dist.add_trace(
        go.Histogram(x=fdf[col], marker_color="#1F6FB2", marker_line_color="white", marker_line_width=1, nbinsx=25),
        row=1, col=i,
    )
fig_dist.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────
# 3. VISUALIZACIÓN CLAVE — Capacidad Instalada por Operador (Plotly) + interpretación IA
# ────────────────────────────────────────────────────────────────────────────
st.header("🏭 Capacidad Instalada por Operador")

cap_operador = (
    fdf.groupby("Operador", as_index=False)["Capacidad_Instalada_MW"].sum()
    .sort_values("Capacidad_Instalada_MW", ascending=True)
)
fig_operador = px.bar(
    cap_operador, x="Capacidad_Instalada_MW", y="Operador", orientation="h",
    text="Capacidad_Instalada_MW", color="Capacidad_Instalada_MW", color_continuous_scale="Blues",
    labels={"Capacidad_Instalada_MW": "Capacidad Instalada (MW)", "Operador": ""},
)
fig_operador.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
fig_operador.update_layout(height=520, coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig_operador, use_container_width=True)

summary = build_data_summary(df, fdf)
render_ai_insight(
    section_key="operador",
    system_prompt=(
        "Eres un analista de datos del sector energético. Recibes un resumen JSON de un dataset de "
        "proyectos de energía renovable ya filtrado. Escribe una interpretación breve (3-4 frases), en "
        "español, sobre la distribución de capacidad instalada por operador: menciona el operador líder "
        "y su magnitud exacta, y qué implica esa concentración para el sistema. Usa SOLO los números que "
        "aparecen en el resumen, nunca inventes cifras. Responde en prosa natural, sin listas ni JSON."
    ),
    user_prompt=f"Resumen de datos filtrados (JSON):\n{json.dumps(summary, ensure_ascii=False)}",
    summary=summary,
)

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────
# 4. RESPUESTA A LA PREGUNTA DE NEGOCIO
# ────────────────────────────────────────────────────────────────────────────
st.header("💡 ¿Qué tecnología tiene la mejor relación Inversión vs. Generación Diaria?")

c1, c2 = st.columns([1.15, 1])
with c1:
    st.markdown("**Inversión vs. Generación Diaria por proyecto (Plotly)**")
    fig_scatter = px.scatter(
        fdf, x="Inversion_Inicial_MUSD", y="Generacion_Diaria_MWh", color="Tecnologia",
        size="Capacidad_Instalada_MW", color_discrete_map=TECH_COLORS,
        hover_data=["Operador", "Estado_Actual"],
        labels={"Inversion_Inicial_MUSD": "Inversión Inicial (MUSD)", "Generacion_Diaria_MWh": "Generación Diaria (MWh)"},
        opacity=0.75,
    )
    fig_scatter.update_layout(height=420, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_scatter, use_container_width=True)

with c2:
    st.markdown("**Ratio MWh/día por MUSD invertido, por tecnología (Plotly)**")
    ratio_tech = pd.Series(summary["ratio_mwh_dia_por_musd_por_tecnologia"]).sort_values(ascending=True)
    fig_ratio = px.bar(
        x=ratio_tech.values, y=ratio_tech.index, orientation="h",
        color=ratio_tech.index, color_discrete_map=TECH_COLORS,
        labels={"x": "MWh/día por MUSD invertido", "y": ""},
        text=ratio_tech.values,
    )
    fig_ratio.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_ratio.update_layout(height=340, showlegend=False, margin=dict(l=10, r=30, t=20, b=10))
    st.plotly_chart(fig_ratio, use_container_width=True)

render_ai_insight(
    section_key="pregunta_negocio",
    system_prompt=(
        "Eres un analista de datos experto en energías renovables. Recibes un resumen JSON con el ratio "
        "de generación diaria (MWh) por cada millón de dólares invertido (MUSD), desglosado por tecnología, "
        "además de la eficiencia media y el costo por MW de cada una. Responde, en español y en 4-6 frases, "
        "la pregunta de negocio: '¿Qué tecnología tiene la mejor relación entre inversión y generación "
        "diaria?'. Identifica la tecnología ganadora citando su valor exacto de ratio, compárala con la "
        "segunda mejor, y explica brevemente por qué ocurre (costo por MW, eficiencia/factor de planta). "
        "Usa SOLO los números del resumen, nunca inventes cifras. No repitas el JSON."
    ),
    user_prompt=f"Resumen de datos filtrados (JSON):\n{json.dumps(summary, ensure_ascii=False)}",
    summary=summary,
)

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────
# 5. CHATEA CON TUS DATOS
# ────────────────────────────────────────────────────────────────────────────
st.header("💬 Chatea con tus datos")
st.caption(
    "Pregúntale al modelo lo que quieras sobre los proyectos filtrados: comparaciones, prioridades de "
    "inversión, lecturas de las gráficas, etc. Sus respuestas están fundamentadas en el resumen numérico "
    "real de tus datos (no en las 500 filas completas)."
)

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

chat_prompt = st.chat_input("Ej: ¿qué operador debería priorizar inversión en hidroeléctrica?")

if chat_prompt:
    if not api_key:
        st.error("⚠️ Ingresa tu Groq API Key en la barra lateral antes de chatear.")
        st.stop()

    st.session_state["chat_messages"].append({"role": "user", "content": chat_prompt})
    with st.chat_message("user"):
        st.markdown(chat_prompt)

    chat_system_prompt = (
        "Eres un analista de datos conversacional, experto en el sector de energías renovables. "
        "Tienes acceso al siguiente resumen JSON del dataset de proyectos actualmente filtrado en el "
        "dashboard por el usuario:\n\n"
        f"{json.dumps(summary, ensure_ascii=False)}\n\n"
        "Responde las preguntas del usuario con precisión, usando SOLO los números de este resumen. "
        "Si te preguntan algo que no se puede responder con esta información (por ejemplo, detalles de "
        "un proyecto individual específico que no aparece en el resumen), dilo claramente en lugar de "
        "inventar datos. Puedes comparar tecnologías u operadores, calcular proporciones simples a partir "
        "de las cifras dadas, y dar recomendaciones de negocio fundamentadas."
    )
    api_messages = [{"role": "system", "content": chat_system_prompt}] + st.session_state["chat_messages"]

    try:
        client = get_groq_client()
        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=MODEL_ID, messages=api_messages, temperature=ai_temperature, stream=True,
            )

            def token_stream():
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta

            full_response = st.write_stream(token_stream())
        st.session_state["chat_messages"].append({"role": "assistant", "content": full_response})

    except AuthenticationError:
        st.error("🔑 API Key inválida. Verifica que la copiaste correctamente.")
        st.session_state["chat_messages"].pop()
    except RateLimitError:
        st.error("⏳ Límite de solicitudes de Groq alcanzado. Espera un momento e inténtalo de nuevo.")
        st.session_state["chat_messages"].pop()
    except APIError as e:
        st.error(f"❌ Error de la API de Groq: {e}")
        st.session_state["chat_messages"].pop()

st.caption(
    "Dashboard construido con Streamlit, Pandas, Plotly y Groq (Llama 3.3 70B)."
)
