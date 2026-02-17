import streamlit as st
import os
import functools
from typing import Annotated, TypedDict

# Importaciones de LangChain y LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, BaseMessage, AiMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="F1 Paddock AI", page_icon="🏎️", layout="wide")
st.title("🏎️ F1 Paddock Intelligence")
st.markdown("Investigación de Fórmula 1 en tiempo real con agentes de IA.")

# Barra lateral para llaves
with st.sidebar:
    st.header("🔑 Configuración")
    google_key = st.text_input("Google API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password", value="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")
    
    if google_key: 
        os.environ["GOOGLE_API_KEY"] = google_key.strip()
    if tavily_key: 
        os.environ["TAVILY_API_KEY"] = tavily_key.strip()

# --- DEFINICIÓN DEL ESTADO ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- FUNCIÓN DE PROCESAMIENTO DE TEXTO (SOLUCIÓN AL FORMATO) ---
def extraer_texto_markdown(mensaje):
    """Extrae el texto limpio de la respuesta del modelo para evitar el formato JSON."""
    if isinstance(mensaje.content, list):
        # Si es una lista de bloques, unimos el contenido de texto
        return "".join([block['text'] for block in mensaje.content if block.get('type') == 'text'])
    return mensaje.content

# --- LÓGICA DE NODOS ---
def call_model(state, llm, system_prompt, tools=None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt | (llm.bind_tools(tools) if tools else llm)
    try:
        response = chain.invoke(state)
        return {"messages": [response]}
    except Exception as e:
        st.error(f"❌ Error en el modelo: {str(e)}")
        st.stop()

# --- CONSTRUCCIÓN DEL GRAFO ---
if google_key and tavily_key:
    # Usamos Gemini 1.5 Flash por su alta estabilidad con herramientas
    llm = ChatGoogleGenerativeAI(model='gemini-flash-latest', temperature=0.1)
    search_tool = TavilySearchResults(max_results=3)
    
    workflow = StateGraph(AgentState)
    
    # Configuramos los agentes
    reporter_prompt = "Eres un reportero de F1. DEBES usar la herramienta de búsqueda para obtener noticias y rumores actuales antes de responder."
    editor_prompt = "Eres el editor jefe. Crea un resumen épico y profesional en formato Markdown basado en la info recibida."

    workflow.add_node("reporter", functools.partial(call_model, llm=llm, tools=[search_tool], system_prompt=reporter_prompt))
    workflow.add_node("tools", ToolNode([search_tool]))
    workflow.add_node("editor", functools.partial(call_model, llm=llm, system_prompt=editor_prompt))

    # Definimos las flechas (bordes)
    workflow.set_entry_point("reporter")
    workflow.add_conditional_edges("reporter", lambda x: "tools" if x["messages"][-1].tool_calls else "editor")
    workflow.add_edge("tools", "reporter")
    workflow.add_edge("editor", END)

    app = workflow.compile()

    # --- INTERFAZ DE USUARIO ---
    pregunta = st.text_input("¿Qué sucede en el Paddock?", placeholder="Ej: Rumores sobre el futuro de Max Verstappen")

    if st.button("🏁 Iniciar Investigación"):
        if not pregunta:
            st.warning("Escribe una pregunta para empezar.")
        else:
            with st.status("🛠️ Conectando con boxes...", expanded=True) as status:
                try:
                    # Ejecución
                    resultado_final = app.invoke({"messages": [HumanMessage(content=pregunta)]})
                    
                    status.update(label="✅ Reporte finalizado", state="complete")
                    
                    # Extracción y muestra de resultados
                    mensaje_final = resultado_final["messages"][-1]
                    texto_limpio = extraer_texto_markdown(mensaje_final)
                    
                    st.markdown("---")
                    st.subheader("📰 Reporte Oficial")
                    st.markdown(texto_limpio)
                    
                except Exception as e:
                    st.error(f"Fallo en la telemetría (Grafo): {e}")
else:
    st.info("⚠️ Introduce tus API Keys en la barra lateral para arrancar el motor.")
