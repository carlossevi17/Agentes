import streamlit as st
import os
import functools
from typing import Annotated, Literal, TypedDict

# Importaciones críticas
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, BaseMessage, AiMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- CONFIGURACIÓN UI ---
st.set_page_config(page_title="F1 Paddock AI", page_icon="🏎️")
st.title("🏎️ F1 Paddock Intelligence")

# Gestión de claves mediante la barra lateral
with st.sidebar:
    st.header("🔑 API Keys")
    google_key = st.text_input("Google API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password", value="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")

    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

# --- LÓGICA DEL GRAFO ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm

# Lógica de decisión
def should_continue(state: AgentState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return "analyst"

# --- CONSTRUCCIÓN ---
if google_key and tavily_key:
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
    tools = [TavilySearchResults(max_results=3)]
    
    # Agentes especializados
    reporter = create_agent(llm, tools, "Eres un reportero de F1. Busca noticias actuales sobre pilotos, equipos y rumores.")
    analyst = create_agent(llm, [], "Eres un analista técnico de F1. Analiza cómo la noticia afecta al rendimiento o al mundial.")
    
    workflow = StateGraph(AgentState)
    workflow.add_node("reporter", lambda state: {"messages": [reporter.invoke(state)]})
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("analyst", lambda state: {"messages": [analyst.invoke(state)]})

    workflow.set_entry_point("reporter")
    workflow.add_conditional_edges("reporter", should_continue)
    workflow.add_edge("tools", "reporter")
    workflow.add_edge("analyst", END)

    app = workflow.compile()

    # --- INTERACCIÓN ---
    pregunta = st.text_input("Pregunta al Paddock:", placeholder="¿Qué se sabe del fichaje de Newey?")

    if st.button("Lanzar Reporte"):
        with st.spinner("Buscando en el pitlane..."):
            result = app.invoke({"messages": [HumanMessage(content=pregunta)]})
            st.markdown("### 🏁 Resultado de la Investigación")
            st.write(result['messages'][-1].content)
else:
    st.warning("Introduce tus API Keys en la barra lateral para arrancar motores.")
