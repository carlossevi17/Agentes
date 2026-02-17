import streamlit as st
import os
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# 1. Configuración de la Página
st.set_page_config(page_title="F1 PARA NOVATOS", page_icon="🏎️", layout="wide")
st.title("🏎️ F1 PARA NOVATOS")
st.markdown("### Entiende la última hora de la Fórmula 1 sin ser ingeniero")

# 2. Sidebar: Configuración de API Keys
with st.sidebar:
    st.header("🔑 Configuración")
    google_key = st.text_input("Google API Key:", type="password")
    tavily_key = st.text_input("Tavily API Key:", type="password")
    
    if google_key and tavily_key:
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["TAVILY_API_KEY"] = tavily_key
        st.success("✅ Boxes listos: APIs configuradas")

# 3. Definición del Estado y el Grafo
class F1AgentState(TypedDict):
    question: str
    news_context: str
    explanation: str

def tool_search_f1_news(state: F1AgentState):
    """Busca noticias de F1 en tiempo real"""
    search = TavilySearchResults(max_results=4)
    # Refinamos la búsqueda añadiendo "F1 news" a la pregunta
    query = f"Fórmula 1 latest news: {state['question']}"
    results = search.invoke(query)
    return {"news_context": str(results)}

def generator_f1_expert(state: F1AgentState):
    """Traduce noticias complejas a lenguaje sencillo de F1"""
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash') # He actualizado a la versión flash 2.0
    
    prompt = f"""
    Eres un comentarista experto de Fórmula 1, amable y muy didáctico.
    Tu objetivo es explicarle a un nuevo fan qué está pasando.
    
    CONTEXTO DE NOTICIAS:
    {state['news_context']}
    
    PREGUNTA DEL FAN:
    {state['question']}
    
    INSTRUCCIÓN: 
    1. Usa analogías de coches de calle para que se entienda.
    2. Explica brevemente términos técnicos si aparecen (como DRS, degradación, undercut).
    3. Mantén un tono emocionante, ¡como si estuviéramos en la parrilla de salida!
    """
    
    response = llm.invoke(prompt)
    return {"explanation": response.content}

# Construcción del flujo (El Grafo)
workflow = StateGraph(F1AgentState)
workflow.add_node("analista_noticias", tool_search_f1_news)
workflow.add_node("comentarista", generator_f1_expert)

workflow.set_entry_point("analista_noticias")
workflow.add_edge("analista_noticias", "comentarista")
workflow.add_edge("comentarista", END)

app_graph = workflow.compile()

# 4. Interfaz de Usuario
if google_key and tavily_key:
    pregunta = st.text_input("¿Qué está pasando en el Paddock?", 
                             placeholder="Ej: ¿Por qué Ferrari es tan rápido hoy? o ¿Qué es el porpoising?")

    if pregunta:
        with st.spinner("🏁 Analizando la telemetría y noticias..."):
            try:
                inputs = {"question": pregunta}
                resultado = app_graph.invoke(inputs)
                
                st.markdown("---")
                st.subheader("🎙️ Análisis del Experto:")
                st.write(resultado["explanation"])
                
                with st.expander("📑 Fuentes consultadas (Pit Wall Data)"):
                    st.code(resultado["news_context"], language="text")
            
            except Exception as e:
                st.error(f"¡Bandera Roja! Error: {str(e)}")
else:
    st.warning("👈 Introduce las claves en el sidebar para arrancar el motor.")
