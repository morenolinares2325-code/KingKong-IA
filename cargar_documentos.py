import os
import pickle
import pypdf
import ollama
import numpy as np

DB_DIR = "vector_db"
os.makedirs(DB_DIR, exist_ok=True)

def procesar_carpeta(carpeta_origen, nombre_coleccion):
    if not os.path.exists(carpeta_origen):
        print(f"⚠️ Carpeta inexistente: {carpeta_origen}")
        return

    archivos = [f for f in os.listdir(carpeta_origen) if f.lower().endswith((".pdf", ".txt"))]
    if not archivos:
        print(f"ℹ️ Sin archivos en {carpeta_origen}. Saltando.")
        return

    print(f"\n==================================================")
    print(f"Indexando: [{nombre_coleccion}] desde {carpeta_origen}")
    print(f"==================================================")

    base_datos = [] # Lista de diccionarios: {"doc": chunk, "embed": vector}

    for archivo in archivos:
        ruta = os.path.join(carpeta_origen, archivo)
        texto = ""

        print(f"-> Leyendo: {archivo}...")
        if archivo.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(ruta)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    texto += t + "\n"
        elif archivo.lower().endswith(".txt"):
            with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
                texto = f.read()

        if not texto.strip():
            continue

        chunk_size = 1000
        overlap = 150
        chunks = []
        start = 0

        while start < len(texto):
            end = start + chunk_size
            chunks.append(texto[start:end])
            start += (chunk_size - overlap)

        print(f"   Generando embeddings para {len(chunks)} fragmentos...")
        for chunk in chunks:
            res = ollama.embeddings(model="nomic-embed-text", prompt=chunk)
            base_datos.append({
                "doc": chunk,
                "embed": np.array(res["embedding"], dtype=np.float32)
            })

    # Guardar en disco
    archivo_db = os.path.join(DB_DIR, f"{nombre_coleccion}.pkl")
    with open(archivo_db, "wb") as f:
        pickle.dump(base_datos, f)

    print(f"✅ Colección [{nombre_coleccion}] guardada con éxito en {archivo_db}")

if __name__ == "__main__":
    procesar_carpeta("documentos/analyst", "manual_trading")
    procesar_carpeta("documentos/coder", "documentacion_mql5")
    procesar_carpeta("documentos/support", "faq_soporte")
    print("\n🚀 Indexación terminada.")