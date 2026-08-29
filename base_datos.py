import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader, TextLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CARPETA_DOCUMENTOS = "./documentos_turismo"
CARPETA_BASE_DATOS = "./chroma_db"

def cargar_documentos():
    documentos_totales = []
    
    # Escaneamos todos los archivos en la carpeta
    for archivo in os.listdir(CARPETA_DOCUMENTOS):
        ruta_completa = os.path.join(CARPETA_DOCUMENTOS, archivo)
        cargador = None
        
        # Elegimos el lector según la extensión del archivo
        if archivo.endswith(".pdf"):
            cargador = PyPDFLoader(ruta_completa)
        elif archivo.endswith(".docx"):
            cargador = Docx2txtLoader(ruta_completa)
        elif archivo.endswith(".xlsx") or archivo.endswith(".xls"):
            cargador = UnstructuredExcelLoader(ruta_completa, mode="elements")
        elif archivo.endswith(".txt"):
            cargador = TextLoader(ruta_completa, encoding="utf-8")
            
        if cargador:
            print(f"Leyendo: {archivo}")
            documentos_totales.extend(cargador.load())
            
    return documentos_totales

def construir_base_datos():
    print("Iniciando lectura de carpeta...")
    documentos = cargar_documentos()
    
    # Cortamos los documentos en fragmentos más pequeños para que la IA los procese mejor
    cortador = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    fragmentos = cortador.split_documents(documentos)
    
    print(f"Se crearon {len(fragmentos)} fragmentos de texto. Guardando en ChromaDB...")
    
    # Usamos el motor de Google
    motor_vectores = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    # Creamos la base de datos y la guardamos en el disco (persist_directory)
    Chroma.from_documents(
        documents=fragmentos,
        embedding=motor_vectores,
        persist_directory=CARPETA_BASE_DATOS
    )
    print("¡Base de datos construida y guardada exitosamente!")

if __name__ == "__main__":
    construir_base_datos()