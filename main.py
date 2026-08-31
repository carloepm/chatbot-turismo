from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import csv
import requests
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
# --- TRUCO PARA RENDER Y CHROMADB ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ------------------------------------

# Importaciones de LangChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Configuración de Entorno
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("¡Atención! No se encontró la GOOGLE_API_KEY en el archivo .env")

app = FastAPI(title="Turismo Chilecito Web")

URL_GOOGLE_SHEETS = os.getenv("GOOGLE_SHEETS_WEBHOOK_URL")

def guardar_en_sheets(fecha: str, id_sesion: str, pregunta: str, respuesta: str):
    """Envía la interacción a Google Sheets de forma silenciosa"""
    if not URL_GOOGLE_SHEETS:
        return
    try:
        payload = {
            "fecha_hora": fecha,
            "id_sesion": id_sesion,
            "mensaje_turista": pregunta,
            "respuesta_ia": respuesta
        }
        requests.post(URL_GOOGLE_SHEETS, json=payload, timeout=5)
    except Exception as error:
        print(f"⚠️ No se pudo registrar en Google Sheets: {error}")

# Permitir cargar imágenes locales desde la carpeta "static"
RUTA_BASE = Path(__file__).parent
app.mount("/static", StaticFiles(directory=RUTA_BASE / "static"), name="static")
ARCHIVO_HTML = RUTA_BASE / "index.html"

# ---------------------------------------------------------
# SISTEMA DE ANALÍTICA Y RECOLECCIÓN DE DATOS
# ---------------------------------------------------------
ARCHIVO_CSV = RUTA_BASE / "documentos_turismo" / "registro_consultas.csv" 

# --- LÍNEA NUEVA: Crea la carpeta automáticamente si no existe ---
ARCHIVO_CSV.parent.mkdir(parents=True, exist_ok=True)
# ---------------------------------------------------------------

# Si el archivo no existe, lo creamos y escribimos las cabeceras
if not ARCHIVO_CSV.exists():
    with open(ARCHIVO_CSV, mode="w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["Fecha_Hora", "ID_Sesion", "Mensaje_Turista", "Respuesta_IA"])

# 2. Configuración de IA y Base de Datos (RAG)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
ia = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", 
    temperature=0.2,
    google_api_key=GOOGLE_API_KEY
)
motor_vectores = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
CARPETA_BASE_DATOS = "./chroma_db"

base_datos = Chroma(persist_directory=CARPETA_BASE_DATOS, embedding_function=motor_vectores)
buscador = base_datos.as_retriever(search_kwargs={"k": 4})

# ---------------------------------------------------------
# 3. EL ADMINISTRADOR DE SESIONES (MULTIUSUARIO)
# ---------------------------------------------------------
# Aquí guardaremos el historial de cada turista por separado
historiales_por_sesion = {}

def obtener_historial_sesion(id_sesion: str) -> BaseChatMessageHistory:
    if id_sesion not in historiales_por_sesion:
        historiales_por_sesion[id_sesion] = ChatMessageHistory()
    return historiales_por_sesion[id_sesion]

# 4. Modelos de Datos (Estructuras)
class SolicitudChat(BaseModel):
    texto: str
    id_sesion: str # Obliga al HTML a enviar de qué pestaña viene

class RespuestaChat(BaseModel):
    respuesta_ia: str
    status: str

# 5. Rutas y Lógica Web
@app.get("/")
async def interfaz_web():
    if not ARCHIVO_HTML.exists():
        raise HTTPException(status_code=404, detail="No se encontró el HTML")
    return FileResponse(ARCHIVO_HTML)

@app.post("/api/chat", response_model=RespuestaChat)
async def procesar_chat(solicitud: SolicitudChat, request: Request):
    ip_turista = request.headers.get("X-Forwarded-For")
    if ip_turista:
        ip_turista = ip_turista.split(",")[0].strip() # Tomar la primera IP si hay varias
    else:
        ip_turista = request.client.host # Respaldo por si se prueba en la computadora local
        
    id_memoria_ip = f"ip_{ip_turista}"
    try:
        print(f"\nTurista [{solicitud.id_sesion}] pregunta: {solicitud.texto}")

        
        
        # A. RAG: Buscar en la base de datos
        documentos = buscador.invoke(solicitud.texto)
        contexto_local = "\n\n".join([doc.page_content for doc in documentos])
        
      # B. RAG: Prompt Híbrido 
        instruccion = f"""
        Eres un guía turístico experto de Chilecito, La Rioja, Argentina. Responde de forma cálida y útil.
        
        A continuación, te proporciono información actualizada de nuestra base de datos local:
        --- INICIO DE DATOS LOCALES ---
        {contexto_local}
        --- FIN DE DATOS LOCALES ---
        
        REGLAS DE COMPORTAMIENTO:
        1. PREGUNTAS ESPECÍFICAS: Si el turista pregunta por precios, horarios, eventos u hoteles, utiliza los DATOS LOCALES.
        2. CHARLA GENERAL Y CONTEXTO: Si te saluda o hace preguntas generales sobre turismo, usa tu conocimiento general.
        3. EL LÍMITE Y DERIVACIÓN: Nunca inventes precios, fechas ni nombres que no estén en los DATOS LOCALES. Si no tienes la información exacta, recomiéndale comunicarse con un asesor turístico.
        IMPORTANTE: Cuando le des el contacto, debes usar EXACTAMENTE este formato Markdown para que el número sea un botón clickeable: 
        [+54 9 3825 67-5999](https://wa.me/5493825675999)
        """
        
        # C. Motor con Memoria Separada
        motor_con_memoria = RunnableWithMessageHistory(
            ia,
            obtener_historial_sesion
        )
        
        # D. Consultar a Gemini pasándole el ID de la sesión actual
        resultado = motor_con_memoria.invoke(
            [
                SystemMessage(content=instruccion),
                HumanMessage(content=solicitud.texto)
            ],
            config={"configurable": {"session_id": id_memoria_ip}}
        )
        
        texto_ia = resultado.content if not isinstance(resultado.content, list) else resultado.content[0].get("text", "")
        # D. Consultar a Gemini pasándole el ID de la sesión actual
        resultado = motor_con_memoria.invoke(
            [
                SystemMessage(content=instruccion),
                HumanMessage(content=solicitud.texto)
            ],
            config={"configurable": {"session_id": solicitud.id_sesion}}
        )
        
        texto_ia = resultado.content if not isinstance(resultado.content, list) else resultado.content[0].get("text", "")

        # Guardar en Google Sheets en tiempo real
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        guardar_en_sheets(fecha_actual, solicitud.id_sesion, solicitud.texto, texto_ia)
        
        # --- NUEVO: GUARDAR LA FILA EN EL DATASET ---
        fecha_actual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ARCHIVO_CSV, mode="a", newline="", encoding="utf-8") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow([fecha_actual, solicitud.id_sesion, solicitud.texto, texto_ia])
        # --------------------------------------------
        
        return RespuestaChat(respuesta_ia=texto_ia, status="exito")

                
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {str(e)}\n")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bajar")
async def descargar_registros():
    """Ruta oculta para descargar el CSV de analíticas"""
    if ARCHIVO_CSV.exists():
        # Usamos FileResponse (que ya importamos arriba) para enviar el archivo
        return FileResponse(
            path=ARCHIVO_CSV, 
            media_type="text/csv", 
            filename="consultas_turistas.csv"
        )
    else:
        return {"mensaje": "Todavía no hay datos guardados."}