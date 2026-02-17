import streamlit as st
import os
import functools
from typing import Annotated, Literal, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, BaseMessage, AiMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="F1 AI News Agent", page_icon="🏎️")
st.title("🏎️ F1 Paddock Intelligence")
st.markdown("Agente autónomo de noticias de Fórmula 1")

# --- BARRA LATERAL PARA LLAVES ---
with st.sidebar:
    st.header("Configuración")
    google_key = st.text_input("Google API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password", value="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")
    
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key

# --- LÓGICA DE LANGGRAPH ---
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

# Definición de Nodos
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [result]}

def should_search(state: AgentState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return "strategist"

# --- CONSTRUCCIÓN DEL GRAFO ---
if google_key and tavily_key:
    llm = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')
    tools = [TavilySearchResults(max_results=5)]
    
    # Agentes
    reporter = create_agent(llm, tools, "Eres un reportero de F1 en el pitlane. Busca las últimas noticias, rumores y tiempos. Sé rápido y preciso.")
    strategist = create_agent(llm, [], "Eres un estratega de F1. Analiza las noticias del reportero y explica cómo afectan al campeonato o a la carrera.")
    editor = create_agent(llm, [], "Eres el Editor Jefe de una revista de F1. Escribe un artículo breve, con un titular impactante y estilo deportivo.")

    # Nodos del flujo
    workflow = StateGraph(AgentState)
    workflow.add_node("reporter", functools.partial(agent_node, agent=reporter, name="Reportero"))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("strategist", functools.partial(agent_node, agent=strategist, name="Estratega"))
    workflow.add_node("editor", functools.partial(agent_node, agent=editor, name="Editor"))

    workflow.set_entry_point("reporter")
    workflow.add_conditional_edges("reporter", should_search)
    workflow.add_edge("tools", "reporter")
    workflow.add_edge("strategist", "editor")
    workflow.add_edge("editor", END)

    app = workflow.compile()

    # --- INTERFAZ STREAMLIT ---
    user_input = st.text_input("¿Qué quieres saber de la F1?", placeholder="Ej: Rumores sobre el asiento de Red Bull")

    if st.button("Investigar"):
        if user_input:
            st.session_state.messages = [HumanMessage(content=user_input)]
            
            with st.status("Analizando el Paddock...", expanded=True) as status:
                for event in app.stream({"messages": st.session_state.messages}, stream_mode="values"):
                    last_msg = event['messages'][-1]
                    if isinstance(last_msg, AiMessage):
                        if last_msg.tool_calls:
                            st.write("🔍 Buscando en telemetría y noticias...")
                        else:
                            st.write(f"✅ {last_msg.content[:50]}...")
                status.update(label="¡Investigación completada!", state="complete", expanded=False)

            # Mostrar resultado final
            final_content = event['messages'][-1].content
            st.markdown("---")
            st.markdown(final_content)
        else:
            st.warning("Escribe algo primero.")
else:
    st.info("Por favor, introduce tus API Keys en la barra lateral para empezar.")

# --- PRUEBA ---
inputs = {"messages": [HumanMessage(content="El misterio de las Caras de Bélmez")]}
for event in graph.stream(inputs, stream_mode="values"):
    event['messages'][-1].pretty_print()
