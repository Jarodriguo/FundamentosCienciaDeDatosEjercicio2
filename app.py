"""
Chatbot de Trivia — Cultura General e Historia Mundial
Motor: Groq API + Llama 3.3 70B (llama-3.3-70b-versatile)

Ejecutar con: streamlit run app.py
La API Key de Groq se ingresa en la barra lateral (no se guarda en el código).
"""

import streamlit as st
from groq import Groq, APIError, AuthenticationError, RateLimitError

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ────────────────────────────────────────────────────────────────────────────
MODEL_ID = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "Eres un asistente experto en cultura general e historia mundial, especializado en trivia. "
    "Respondes preguntas de forma precisa y verificable, citando fechas, lugares, personajes y "
    "contexto relevante cuando aporte valor. Mantienes un tono ameno, cercano y didáctico, como un "
    "buen anfitrión de trivia. Si no estás seguro de un dato, dilo abiertamente en lugar de inventarlo. "
    "Cuando el usuario te pida preguntas de trivia, hazlas de una en una: primero solo la pregunta "
    "(con 4 opciones si aplica), espera la respuesta del usuario, y luego confírmale si acertó o no, "
    "explicando brevemente por qué. Varía los temas: historia antigua, historia moderna, geografía, "
    "arte, ciencia, mitología, grandes personajes y eventos mundiales."
)

WELCOME_MESSAGE = (
    "¡Hola! 🌍 Soy tu bot de trivia de **cultura general e historia mundial**. "
    "Puedes preguntarme lo que quieras, o pedirme: *\"hazme una pregunta de trivia\"* "
    "para empezar a jugar."
)

st.set_page_config(
    page_title="Trivia Bot | Cultura General e Historia Mundial",
    page_icon="🌍",
    layout="centered",
)

# ────────────────────────────────────────────────────────────────────────────
# BARRA LATERAL
# ────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuración")

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    placeholder="gsk_...",
    help="Consíguela gratis en https://console.groq.com/keys. No se guarda en ningún archivo.",
)

temperature = st.sidebar.slider(
    "Creatividad (temperature)", min_value=0.0, max_value=1.5, value=0.6, step=0.1,
    help="Valores bajos = respuestas más precisas y consistentes. Valores altos = más variadas.",
)

if st.sidebar.button("🗑️ Reiniciar conversación", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(f"**Modelo:** `{MODEL_ID}` (Llama 3.3 70B vía Groq)")
st.sidebar.caption(
    "⚠️ Groq anunció el retiro (deprecación) de este modelo para el **16 de agosto de 2026**, "
    "recomendando migrar a `openai/gpt-oss-120b` o `qwen/qwen3.6-27b`. Si después de esa fecha "
    "empiezas a ver errores de modelo no encontrado, cambia la constante `MODEL_ID` al inicio de "
    "`app.py` por uno de esos dos."
)

# ────────────────────────────────────────────────────────────────────────────
# ENCABEZADO
# ────────────────────────────────────────────────────────────────────────────
st.title("🌍 Trivia Bot: Cultura General e Historia Mundial")
st.caption(
    "Chatbot conversacional impulsado por Groq (inferencia ultrarrápida) y Llama 3.3 70B. "
    "Pregunta libremente o pide preguntas de trivia."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ────────────────────────────────────────────────────────────────────────────
# HISTORIAL DE CHAT
# ────────────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(WELCOME_MESSAGE)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Botón de inicio rápido para pedir una pregunta de trivia sin escribir
quick_trivia = st.button("🎲 Hazme una pregunta de trivia")

# ────────────────────────────────────────────────────────────────────────────
# ENTRADA DEL USUARIO
# ────────────────────────────────────────────────────────────────────────────
user_prompt = st.chat_input("Escribe tu pregunta, o pide 'hazme una pregunta de trivia'...")

prompt = user_prompt or ("Hazme una pregunta de trivia de cultura general o historia mundial." if quick_trivia else None)

if prompt:
    if not api_key:
        st.error("⚠️ Ingresa tu Groq API Key en la barra lateral antes de continuar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages

    try:
        client = Groq(api_key=api_key)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=api_messages,
                temperature=temperature,
                stream=True,
            )

            def token_stream():
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta

            full_response = st.write_stream(token_stream())

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except AuthenticationError:
        st.error("🔑 API Key inválida. Verifica que la copiaste correctamente desde console.groq.com/keys.")
        st.session_state.messages.pop()  # quita la pregunta del usuario que no pudo responderse
    except RateLimitError:
        st.error("⏳ Se alcanzó el límite de solicitudes de tu cuenta de Groq. Espera un momento e inténtalo de nuevo.")
        st.session_state.messages.pop()
    except APIError as e:
        st.error(f"❌ Error de la API de Groq: {e}")
        st.session_state.messages.pop()
    except Exception as e:
        st.error(f"❌ Ocurrió un error inesperado: {e}")
        st.session_state.messages.pop()
