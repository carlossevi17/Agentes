import streamlit as st
import os
import functools
from typing import Annotated, Literal, TypedDict

# Intentar importar con manejo de errores para diagnóstico
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain_core.messages import HumanMessage, BaseMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
except ImportError as e:
    st.error(f"Falta una librería: {e}. Asegúrate de que 'requirements.txt' esté en la raíz de tu repo 'agentes'.")
    st.stop()

st.set_page_config(page_title="F1 Reporter", page_icon="🏎️")
st.title("🏎️ F1 Paddock Agent")

# Configuración de API Keys en el sidebar
with st.sidebar:
    google_key = st.text_input("Google API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password", value="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")
    if google_key: os.environ["GOOGLE_API_KEY"] = google_key
    if tavily_key: os.environ["TAVILY_API_KEY"] = tavily_key

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

if google_key and tavily_key:
    llm = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')
    tools = [TavilySearchResults(max_results=3)]
    
    # Grafo simple de F1
    workflow = StateGraph(AgentState)
    
    # Nodo Reportero
    def reporter_node(state):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres un reportero de F1. Usa herramientas para buscar noticias."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        chain = prompt | llm.bind_tools(tools)
        return {"messages": [chain.invoke(state)]}

    workflow.add_node("reporter", reporter_node)
    workflow.add_node("tools", ToolNode(tools))
    
    # Nodo Editor
    def editor_node(state):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Eres el editor. Resume la noticia de F1 de forma épica."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        return {"messages": [prompt.invoke(state | {"messages": state["messages"][-2:]})]}

    workflow.add_node("editor", editor_node)

    # Flujo
    workflow.set_entry_point("reporter")
    workflow.add_conditional_edges("reporter", lambda x: "tools" if x["messages"][-1].tool_calls else "editor")
    workflow.add_edge("tools", "reporter")
    workflow.add_edge("editor", END)

    app = workflow.compile()

    # UI
    user_query = st.text_input("Pregunta sobre F1:")
    if st.button("Buscar"):
        with st.spinner("Consultando fuentes..."):
            final = app.invoke({"messages": [HumanMessage(content=user_query)]})
            st.markdown(final["messages"][-1].content)
else:
    st.info("Introduce las llaves en el sidebar.")
