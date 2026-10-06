# KingKongIA

IA local privada para revision y correccion de codigo.

## Requisitos

- Windows 10/11
- Ollama instalado
- Python 3.11 o superior
- 16 GB de RAM minimo

## Instalacion

1. Clona el repositorio:
   git clone https://github.com/morenolinares2325-code/KingKong-IA.git

2. Instala Ollama y descarga el modelo base:
   ollama pull qwen2.5-coder:7b

3. Crea el modelo personalizado:
   ollama create kingkong-coder -f Modelfile

4. Instala las dependencias:
   pip install -r requirements.txt

5. Ejecuta la aplicacion:
   streamlit run app.py

## Privacidad

Todo el procesamiento ocurre localmente. Ningun dato sale de tu equipo.
