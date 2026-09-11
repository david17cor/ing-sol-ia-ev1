import os
import json
import streamlit as st
from tools import consultar_estado_rut
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# 1. Configurar RAG
@st.cache_resource
def setup_rag():
    doc_path = os.path.join("data", "terminos_condiciones.txt")
    loader = TextLoader(doc_path, encoding="utf-8")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

st.set_page_config(page_title="Asistente FBC", page_icon="🤖")
st.title("🤖 Asistente Virtual - Fundación Bienestar Corporativo")

retriever = setup_rag()
llm = ChatOllama(model="qwen2.5:3b", temperature=0.1)
llm_with_tools = llm.bind_tools([consultar_estado_rut])

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Escribe tu consulta aquí..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    rag_docs = retriever.invoke(user_input)
    context_str = "\n\n".join([d.page_content for d in rag_docs])
    
    prompt_system = f"""Eres el Asistente Virtual Oficial de la Fundación Bienestar Corporativo (FBC).
Tu trato es empático, cercano y estrictamente profesional.
- Solicita el RUT en la primera interacción. Usa la herramienta 'consultar_estado_rut' cuando recibas un RUT.
- Si el usuario consulta por 'Legal Asesores', 'Giftcard' u otros convenios excluidos, responde exactamente:
  "Lamentamos informarle que revisando su información usted no se encuentra en condiciones de obtener el beneficio '[Nombre]', para más información contactar mediante call center."
- Para pagos, recuerda que siempre son mediante descuento por planilla.
Contexto RAG recuperado: {context_str}"""

    response = llm_with_tools.invoke([("system", prompt_system), ("human", user_input)])
    
    if response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "consultar_estado_rut":
                rut_val = tool_call["args"].get("rut", "")
                tool_res = consultar_estado_rut.invoke({"rut": rut_val})
                
                final_response = llm.invoke([
                    ("system", prompt_system),
                    ("human", user_input),
                    ("assistant", f"Resultado de consulta BD: {tool_res}")
                ])
                output = final_response.content
    else:
        output = response.content

    st.session_state.messages.append({"role": "assistant", "content": output})
    st.chat_message("assistant").write(output)