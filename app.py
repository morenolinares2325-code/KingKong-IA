import streamlit as st
import ollama
import os
import pickle
import pypdf
import numpy as np

# Configuración de página
st.set_page_config(
    page_title="KingKong Suite IA",
    page_icon="🦍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos visuales
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 15% 15%, #181d26 0%, #0d1117 100%);
        color: #e6edf3;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    .hero-container {
        background: linear-gradient(135deg, rgba(255, 123, 0, 0.08) 0%, rgba(20, 25, 35, 0.6) 100%);
        border: 1px solid rgba(255, 123, 0, 0.25);
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff8c00, #ffb703, #fb8500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        margin-bottom: 0.8rem;
    }
    .badges-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .tech-badge {
        display: inline-flex;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-model {
        background: rgba(88, 166, 255, 0.12);
        color: #58a6ff;
        border: 1px solid rgba(88, 166, 255, 0.3);
    }
    .badge-rag {
        background: rgba(63, 185, 80, 0.12);
        color: #3fb950;
        border: 1px solid rgba(63, 185, 80, 0.3);
    }
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.8rem;
        padding: 0.8rem 1rem;
        background: rgba(22, 27, 34, 0.5) !important;
        border: 1px solid rgba(48, 54, 61, 0.6) !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0b0e14;
        border-right: 1px solid #21262d;
    }
    div.stButton > button {
        background: rgba(248, 81, 73, 0.1);
        color: #f85149;
        border: 1px solid rgba(248, 81, 73, 0.3);
        border-radius: 8px;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background: rgba(248, 81, 73, 0.2);
        border-color: #f85149;
        color: #ff7b72;
    }
</style>
""", unsafe_allow_html=True)

# Asegurar existencia de carpeta de vectores
os.makedirs("vector_db", exist_ok=True)

# Motor de indexación directa desde la web
def indexar_documento(uploaded_file, coleccion):
    texto = ""
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(uploaded_file)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                texto += t + "\n"
    elif uploaded_file.name.lower().endswith(".txt"):
        texto = uploaded_file.read().decode("utf-8", errors="ignore")

    if not texto.strip():
        return False, 0

    chunk_size = 1000
    overlap = 150
    chunks = []
    start = 0
    while start < len(texto):
        end = start + chunk_size
        chunks.append(texto[start:end])
        start += (chunk_size - overlap)

    nuevos_registros = []
    for chunk in chunks:
        res = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
        nuevos_registros.append({
            "doc": chunk,
            "embed": np.array(res["embedding"], dtype=np.float32)
        })

    archivo_db = os.path.join("vector_db", f"{coleccion}.pkl")
    registros_previos = []
    if os.path.exists(archivo_db):
        try:
            with open(archivo_db, "rb") as f:
                registros_previos = pickle.load(f)
        except Exception:
            registros_previos = []

    registros_previos.extend(nuevos_registros)
    with open(archivo_db, "wb") as f:
        pickle.dump(registros_previos, f)

    return True, len(chunks)

# Motor de búsqueda vectorial matricial
def buscar_contexto(pregunta, nombre_coleccion, top_k=2):
    archivo_db = os.path.join("vector_db", f"{nombre_coleccion}.pkl")
    if not os.path.exists(archivo_db):
        return None

    try:
        with open(archivo_db, "rb") as f:
            registros = pickle.load(f)

        if not registros:
            return None

        res_query = ollama.embeddings(model="nomic-embed-text", prompt=pregunta)
        q_vec = np.array(res_query["embedding"], dtype=np.float32)

        docs = [r["doc"] for r in registros]
        matrix = np.array([r["embed"] for r in registros])
        
        sims = np.dot(matrix, q_vec) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_vec) + 1e-10)
        top_indices = np.argsort(sims)[::-1][:top_k]

        coincidencias = [docs[i] for i in top_indices if sims[i] > 0.42]
        return "\n---\n".join(coincidencias) if coincidencias else None
    except Exception:
        return None

# Navegación lateral
with st.sidebar:
    st.markdown("### 🦍 KingKong Workspace")
    st.caption("Inferencia local optimizada")
    st.markdown("---")
    
    modulo = st.radio(
        "Selecciona el asistente:",
        [
            "💻 Coder (Pine Script / MQL5)",
            "📊 Analyst (Trading Cuantitativo)",
            "💬 Support (Atención & Redacción)"
        ],
        index=0
    )
    
    st.markdown("---")
    if st.button("🗑️ Vaciar conversación activa", use_container_width=True):
        st.session_state[f"messages_{modulo}"] = []
        st.rerun()

# Definición de modelo y base de datos según asistente
if "Coder" in modulo:
    modelo_activo = "kingkong-coder"
    coleccion_asociada = "documentacion_mql5"
    titulo = "💻 KingKong Coder"
    subtitulo = "Ingeniería algorítmica, desarrollo en Pine Script (v5/v6) y MQL5."
    prompt_placeholder = "Describe el indicador o script que necesitas..."
elif "Analyst" in modulo:
    modelo_activo = "kingkong-analyst"
    coleccion_asociada = "manual_trading"
    titulo = "📊 KingKong Analyst"
    subtitulo = "Evaluación de estructuras de mercado, gestión de riesgo y setups."
    prompt_placeholder = "Pregunta sobre estrategias, reglas operativas o métricas..."
else:
    modelo_activo = "kingkong-support"
    coleccion_asociada = "faq_soporte"
    titulo = "💬 KingKong Support"
    subtitulo = "Atención al cliente, soporte y redacción comercial."
    prompt_placeholder = "Escribe tu consulta de soporte o texto..."

# Cabecera
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">{titulo}</div>
    <div class="hero-subtitle">{subtitulo}</div>
    <div class="badges-row">
        <span class="tech-badge badge-model">🧠 Motor: {modelo_activo}</span>
        <span class="tech-badge badge-rag">📂 Base RAG: {coleccion_asociada}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sección de carga de información directa para cada IA
with st.expander(f"📁 Añadir información o documentos a {titulo}"):
    st.caption(f"Los documentos que subas aquí se guardarán exclusivamente en la base de **{coleccion_asociada}**.")
    archivo_subido = st.file_uploader(
        "Arrastra un archivo PDF o TXT:", 
        type=["pdf", "txt"], 
        key=f"upload_{coleccion_asociada}"
    )
    if archivo_subido is not None:
        if st.button("📥 Indexar y guardar documento", key=f"btn_{coleccion_asociada}", use_container_width=True):
            with st.spinner("Procesando y generando vectores con nomic-embed-text..."):
                ok, total_chunks = indexar_documento(archivo_subido, coleccion_asociada)
                if ok:
                    st.success(f"✅ ¡{archivo_subido.name} indexado correctamente! ({total_chunks} fragmentos añadidos)")
                else:
                    st.error("⚠️ No se pudo extraer texto del archivo.")

# Historial de conversación
session_key = f"messages_{modelo_activo}"
if session_key not in st.session_state:
    st.session_state[session_key] = []

for msg in st.session_state[session_key]:
    avatar = "👤" if msg["role"] == "user" else "🦍"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Generación de respuestas
if prompt := st.chat_input(prompt_placeholder):
    st.session_state[session_key].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🦍"):
        response_placeholder = st.empty()
        full_reply = ""
        
        contexto_rag = buscar_contexto(prompt, coleccion_asociada)
        mensajes_para_ollama = list(st.session_state[session_key])
        
        if contexto_rag:
            prompt_enriquecido = f"""Utiliza la siguiente información de referencia extraída del manual si es relevante:
[INFORMACIÓN EXCLUSIVA]
{contexto_rag}
[FIN INFORMACIÓN]

Consulta: {prompt}"""
            mensajes_para_ollama[-1] = {"role": "user", "content": prompt_enriquecido}

        texto_espera = "🦍 Generando código..." if "Coder" in modulo else "🦍 Generando respuesta..."
        with st.spinner(texto_espera):
            try:
                stream = ollama.chat(
                    model=modelo_activo,
                    messages=mensajes_para_ollama,
                    stream=True
                )
            except Exception as e:
                st.error(f"⚠️ Error al conectar con {modelo_activo}: {str(e)}")
                stream = None

        if stream:
            for chunk in stream:
                full_reply += chunk["message"]["content"]
                response_placeholder.markdown(full_reply + "▌")
            
            response_placeholder.markdown(full_reply)
            st.session_state[session_key].append({"role": "assistant", "content": full_reply})