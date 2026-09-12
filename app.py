import os
import json
import streamlit as st
from tools import consultar_estado_rut
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# 1. Configuración de página
st.set_page_config(page_title="Asistente FBC", page_icon="🤖", layout="centered")

# 2. Caché para el Pipeline RAG (se ejecuta 1 sola vez)
@st.cache_resource
def setup_rag():
    doc_path = os.path.join("data", "terminos_condiciones.txt")
    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# 3. Caché para la inicialización del LLM
@st.cache_resource
def setup_llm():
    llm = ChatOllama(
        model="qwen2.5:3b", 
        temperature=0.1,
        request_timeout=60.0
    )
    return llm, llm.bind_tools([consultar_estado_rut])

st.title("🤖 Asistente Virtual - Fundación Bienestar Corporativo")

# Inicialización de recursos
retriever = setup_rag()
llm, llm_with_tools = setup_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Historial de chat
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Interacción del usuario
if user_input := st.chat_input("Escribe tu consulta aquí..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    # 1. Recuperación de contexto RAG
    rag_docs = retriever.invoke(user_input)
    context_str = "\n\n".join([d.page_content for d in rag_docs])
    
    # 2. System Prompt estricto
    prompt_system = f"""Eres el Asistente Virtual Oficial de la Fundación Bienestar Corporativo (FBC).
Tu trato es empático, cercano y estrictamente profesional.

REGLA CRÍTICA Y OBLIGATORIA:
- PROHIBIDO pedir el RUT para responder preguntas generales sobre convenios, farmacias, salud dental, reembolsos o horarios. 
- Responde la consulta general INMEDIATAMENTE utilizando el siguiente Contexto RAG recuperado.

INSTRUCCIONES DE ATENCIÓN:
1. Para consultas generales, entrega la información del Contexto RAG sin solicitar datos personales.
2. Solicita o procesa el RUT ÚNICAMENTE si la consulta requiere verificar el estado individual, deudas o cargas del beneficiario.
3. Si la consulta incluye un RUT o requiere revisión de cuenta, invoca obligatoriamente la herramienta 'consultar_estado_rut'.
4. Si consultan por 'Legal Asesores', 'Giftcard' u otros convenios excluidos, responde exactamente:
   "Lamentamos informarle que revisando su información usted no se encuentra en condiciones de obtener el beneficio '[Nombre]', para más información contactar mediante call center."
5. Todos los pagos de beneficios se realizan mediante descuento por planilla.

Contexto RAG recuperado:
{context_str}"""

    with st.spinner("Procesando consulta..."):
        try:
            # 3. Invocación inicial con Ollama
            response = llm_with_tools.invoke([("system", prompt_system), ("human", user_input)])
            output = ""

            # 4. Evaluación de Tool Calls con validación de RUT real
            tool_ejecutada_con_exito = False
            
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call["name"] == "consultar_estado_rut":
                        rut_val = str(tool_call["args"].get("rut", "")).strip()
                        
                        # Solo ejecutar la tool si el modelo capturó un RUT real (no cadena vacía)
                        if rut_val and len(rut_val) >= 7:
                            tool_res = consultar_estado_rut.invoke({"rut": rut_val})
                            output = f"**Resultado de la consulta de estado (RUT {rut_val}):**\n\n{tool_res}"
                            tool_ejecutada_con_exito = True
                            break

            # 5. Si no hubo Tool Call o si intentó llamar a la tool sin un RUT válido (fallback a RAG puro)
            if not tool_ejecutada_con_exito:
                if response.content and response.content.strip():
                    output = response.content
                else:
                    # Forzar respuesta directa con RAG usando el modelo base si la respuesta tool vino vacía
                    fallback_response = llm.invoke([("system", prompt_system), ("human", user_input)])
                    output = fallback_response.content

            # 6. Control de resiliencia final
            if not output or output.strip() == "":
                output = "No se pudo recuperar información para esta consulta. Por favor, reintenta tu pregunta o proporciona tu RUT si deseas revisar tu estado personal."

        except Exception as e:
            output = f"Ocurrió un error al procesar la solicitud: {str(e)}"

    # 7. Renderizado de respuesta
    st.session_state.messages.append({"role": "assistant", "content": output})
    st.chat_message("assistant").write(output)