import streamlit as st
import os
import functools
from typing import Annotated, TypedDict

# Importaciones de LangChain y LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, BaseMessage
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

# --- FUNCIÓN PARA EXTRAER TEXTO LIMPIO ---
def extraer_texto(mensaje):
    """Evita que la salida se vea como JSON extrayendo solo el contenido de texto."""
    content = mensaje.content
    if isinstance(content, list):
        # Si Gemini devuelve una lista de bloques, unimos los de tipo 'text'
        return "".join([block['text'] for block in content if block.get('type') == 'text'])
    return content

# --- LÓGICA DE NODOS ---
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
    # Usamos gemini-1.5-flash por estabilidad
    llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.1)
    search_tool = TavilySearchResults(max_results=3)
    
    workflow = StateGraph(AgentState)
    
    # Agentes
    reporter_msg = "Eres un reportero de F1. BUSCA siempre noticias actuales antes de responder."
    editor_msg = "Eres el editor jefe. Resume la info en un artículo épico con formato Markdown (negritas, listas, etc.)."

    workflow.add_node("reporter", functools.partial(call_model, llm=llm, tools=[search_tool], system_prompt=reporter_msg))
    workflow.add_node("tools", ToolNode([search_tool]))
    workflow.add_node("editor", functools.partial(call_model, llm=llm, system_prompt=editor_msg))

    # Flujo
    workflow.set_entry_point("reporter")
    workflow.add_conditional_edges("reporter", lambda x: "tools" if x["messages"][-1].tool_calls else "editor")
    workflow.add_edge("tools", "reporter")
    workflow.add_edge("editor", END)

    app = workflow.compile()

    # --- UI ---
    pregunta = st.text_input("¿Qué quieres saber de la F1?")

    if st.button("🏁 Iniciar Reporte"):
        if pregunta:
            with st.status("🛠️ Consultando el Paddock...", expanded=True):
                resultado = app.invoke({"messages": [HumanMessage(content=pregunta)]})
                
            # Extraer y mostrar el texto limpio
            texto_final = extraer_texto(resultado["messages"][-1])
            st.markdown("---")
            st.subheader("📰 Reporte Oficial")
            st.markdown(texto_final)
else:
    st.info("⚠️ Introduce las API Keys en el sidebar.")
