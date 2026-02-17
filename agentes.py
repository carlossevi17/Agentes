import streamlit as st
import os
import functools
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, BaseMessage, AiMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="F1 AI Reporter", page_icon="🏎️", layout="wide")
st.title("🏎️ F1 Paddock Intelligence")

# Barra lateral para llaves
with st.sidebar:
    st.header("🔑 Configuración")
    google_key = st.text_input("Google API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password", value="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")
    
    if google_key: os.environ["GOOGLE_API_KEY"] = google_key.strip()
    if tavily_key: os.environ["TAVILY_API_KEY"] = tavily_key.strip()

# --- DEFINICIÓN DEL ESTADO ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- NODOS ---
def call_model(state, llm, system_prompt, tools=None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    chain = prompt | (llm.bind_tools(tools) if tools else llm)
    response = chain.invoke(state)
    return {"messages": [response]}

# --- CONSTRUCCIÓN DEL GRAFO ---
if google_key and tavily_key:
    # IMPORTANTE: Cambiado a gemini-1.5-flash para estabilidad total
    llm = ChatGoogleGenerativeAI(model='gemini-flash-latest', temperature=0.2)
    search_tool = TavilySearchResults(max_results=3)
    
    workflow = StateGraph(AgentState)
    
    # Reportero: Obligado a investigar
    reporter_node = lambda state: call_model(
        state, llm, 
        "Eres un reportero de F1. DEBES usar la herramienta de búsqueda para obtener datos reales y actuales antes de responder.", 
        [search_tool]
    )
    
    # Editor: Sintetiza el reporte final
    editor_node = lambda state: call_model(
        state, llm, 
        "Eres el editor jefe. Toma la información encontrada y escribe un artículo emocionante con titular."
    )

    workflow.add_node("reporter", reporter_node)
    workflow.add_node("tools", ToolNode([search_tool]))
    workflow.add_node("editor", editor_node)

    # Lógica de navegación
    workflow.set_entry_point("reporter")
    workflow.add_conditional_edges("reporter", lambda x: "tools" if x["messages"][-1].tool_calls else "editor")
    workflow.add_edge("tools", "reporter")
    workflow.add_edge("editor", END)

    app = workflow.compile()

    # --- FLUJO DE USUARIO ---
    pregunta = st.text_input("¿Qué quieres saber de la F1?", placeholder="Ej: ¿Cómo va el mundial de constructores?")

    if st.button("🏁 Arrancar Investigación"):
        if not pregunta:
            st.warning("Por favor, escribe una pregunta.")
        else:
            with st.status("🛠️ Investigando en el Paddock...", expanded=True) as status:
                try:
                    # Ejecutamos el grafo
                    final_state = app.invoke({"messages": [HumanMessage(content=pregunta)]})
                    
                    status.update(label="✅ Reporte listo", state="complete")
                    
                    # Buscamos el último mensaje del Editor para mostrarlo
                    st.markdown("---")
                    st.subheader("📊 Reporte Oficial del Editor")
                    st.write(final_state["messages"][-1].content)
                    
                except Exception as e:
                    st.error(f"Hubo un fallo en la carrera: {e}")
else:
    st.info("⚠️ Introduce tus API Keys en el menú de la izquierda.")
