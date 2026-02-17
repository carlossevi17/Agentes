import streamlit as st
import os
import functools
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="F1 AI Reporter", page_icon="🏎️", layout="wide")
st.title("🏎️ F1 Paddock Intelligence")
st.markdown("Investigación de Fórmula 1 en tiempo real con agentes IA.")

# Barra lateral para llaves
with st.sidebar:
    st.header("🔑 Configuración")
    google_key = st.text_input("Google API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password", value="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")
    
    if google_key: os.environ["GOOGLE_API_KEY"] = google_key
    if tavily_key: os.environ["TAVILY_API_KEY"] = tavily_key

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
    try:
        response = chain.invoke(state)
        return {"messages": [response]}
    except Exception as e:
        st.error(f"❌ Error en el modelo: {str(e)}")
        st.stop()

# --- CONSTRUCCIÓN DEL GRAFO ---
if google_key and tavily_key:
    # Usamos Gemini 1.5 Flash por estabilidad
    llm = ChatGoogleGenerativeAI(model='gemini-flash-latest', temperature=0.1)
    search_tool = TavilySearchResults(max_results=3)
    
    workflow = StateGraph(AgentState)
    
    # Definición de funciones de nodo
    reporter_func = functools.partial(call_model, llm=llm, tools=[search_tool], 
                                     system_prompt="Eres un reportero de F1. Busca noticias y rumores actuales.")
    editor_func = functools.partial(call_model, llm=llm, 
                                   system_prompt="Eres el editor jefe. Crea un resumen épico con la info recibida.")

    workflow.add_node("reporter", reporter_node := lambda state: reporter_func(state))
    workflow.add_node("tools", ToolNode([search_tool]))
    workflow.add_node("editor", editor_node := lambda state: editor_func(state))

    # Lógica de navegación
    workflow.set_entry_point("reporter")
    workflow.add_conditional_edges("reporter", lambda x: "tools" if x["messages"][-1].tool_calls else "editor")
    workflow.add_edge("tools", "reporter")
    workflow.add_edge("editor", END)

    app = workflow.compile()

    # --- FLUJO DE USUARIO ---
    pregunta = st.text_input("¿Qué quieres saber del Gran Circo?", placeholder="Ej: ¿Quién sustituirá a Hamilton en 2025?")

    if st.button("🏁 Arrancar Investigación"):
        with st.status("🛠️ Conectando con el Paddock...", expanded=True) as status:
            st.write("Iniciando flujo de agentes...")
            try:
                # Ejecución del grafo
                events = app.stream({"messages": [HumanMessage(content=pregunta)]}, stream_mode="values")
                for event in events:
                    node_name = event['messages'][-1].type
                    st.write(f"⚙️ Procesando: {node_name.capitalize()}")
                
                status.update(label="✅ Investigación finalizada", state="complete")
                
                # Resultado final
                st.markdown("---")
                st.subheader("📊 Reporte Oficial")
                st.write(event["messages"][-1].content)
                
            except Exception as e:
                st.error(f"Hubo un fallo en la carrera: {e}")
else:
    st.info("⚠️ Por favor, introduce tus API Keys en el menú de la izquierda para comenzar.")
