import streamlit as st
import requests
import time

OLLAMA_API_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "kingkong-coder"

def dividir_codigo_en_chunks(codigo, lineas_por_chunk=400, solapamiento=50):
    lineas = codigo.splitlines()
    chunks = []
    for i in range(0, len(lineas), lineas_por_chunk - solapamiento):
        chunk = "\n".join(lineas[i:i + lineas_por_chunk])
        chunks.append(chunk)
    return chunks

def llamar_a_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres un ingeniero de software senior experto en multiples lenguajes. "
                    "Revisa, corrige y mejora el codigo que se te da. "
                    "Devuelve SIEMPRE: 1) el codigo corregido completo, 2) una lista breve de cambios. "
                    "Responde en espanol, claro y directo. No inventes funcionalidades."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0.1}
    }
    try:
        r = requests.post(OLLAMA_API_URL, json=payload, timeout=1800)
        r.raise_for_status()
        return r.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error de conexion con Ollama: {e}"
    except (KeyError, IndexError) as e:
        return f"Error inesperado: {e}"

st.set_page_config(page_title="KingKongIA", page_icon="🦍", layout="wide")
st.title("🦍 KingKongIA - Revisor de Codigo")
st.caption("Sube tu archivo. La IA lo divide, lo revisa y te lo devuelve corregido.")

archivo = st.file_uploader(
    "Sube tu archivo de codigo",
    type=["py", "js", "ts", "java", "cpp", "cs", "txt", "md", "html", "css", "sql"]
)

if archivo:
    codigo = archivo.read().decode("utf-8", errors="ignore")
    total_lineas = len(codigo.splitlines())
    st.info(f"Archivo cargado: {total_lineas} lineas.")

    if st.button("Iniciar revision"):
        chunks = dividir_codigo_en_chunks(codigo)
        total_chunks = len(chunks)
        st.write(f"El archivo se ha dividido en {total_chunks} fragmentos internamente.")

        barra = st.progress(0)
        estado = st.empty()
        resultado = st.empty()

        salidas = []
        tiempos = []
        inicio_total = time.time()

        for i, chunk in enumerate(chunks, 1):
            t0 = time.time()
            estado.info(f"Procesando fragmento {i}/{total_chunks}...")

            prompt = f"""Revisa este fragmento de codigo. Corrige errores, mejora rendimiento y aplica buenas practicas. Devuelve el codigo corregido y explica brevemente los cambios.

Codigo:
            respuesta = llamar_a_ollama(prompt)
            salidas.append(f"---\n### Fragmento {i}/{total_chunks}\n\n{respuesta}\n")

            t1 = time.time()
            tiempos.append(t1 - t0)
            promedio = sum(tiempos) / len(tiempos)
            restante = promedio * (total_chunks - i)

            barra.progress(i / total_chunks)
            estado.success(
                f"Fragmento {i}/{total_chunks} completado. "
                f"Tiempo estimado restante: {restante/60:.1f} min."
            )
            resultado.markdown("\n".join(salidas))

        duracion_total = (time.time() - inicio_total) / 60
        st.success(f"Revision completada en {duracion_total:.1f} minutos.")
        st.balloons()

        informe = "\n".join(salidas)
        st.download_button(
            "Descargar informe completo",
            data=informe,
            file_name=f"informe_kingkongia_{archivo.name}.md",
            mime="text/markdown"
        )
