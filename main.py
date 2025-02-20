import fitz  # PyMuPDF
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Cargar el PDF y extraer texto
def extraer_texto_pdf(ruta_pdf):
    documento = fitz.open(ruta_pdf)
    texto = "\n".join([pagina.get_text() for pagina in documento])
    return texto

# Configurar el modelo de lenguaje
def preguntar_al_llm(texto, pregunta, api_key):
    llm = ChatOpenAI(model="gpt-4", openai_api_key=api_key)
    respuesta = llm([HumanMessage(content=f"Texto del documento:\n{texto}\n\nPregunta: {pregunta}")])
    return respuesta.content

# Uso
ruta_pdf = "documento.pdf"
api_key = "TU_OPENAI_API_KEY"
texto = extraer_texto_pdf(ruta_pdf)

pregunta = "¿Cuál es la fecha del contrato?"
respuesta = preguntar_al_llm(texto, pregunta, api_key)

print("Respuesta:", respuesta)
