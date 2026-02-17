import streamlit as st
import os
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# 1. Configuración de la Página
st.set_page_config(page_title="IA 4 DUMMIES", page_icon="🤖", layout="wide")
st.title("🤖 IA 4 DUMMIES")
st.markdown("### Las noticias de IA contadas como cuentos para jóvenes")

# 2. Sidebar: Configuración de API Keys
with st.sidebar:
    st.header("🔑 Configuración")
    google_key = st.text_input("Google API Key:", type="password")
    tavily_key = st.text_input("Tavily API Key:", type="password")
    
    if google_key and tavily_key:
        # Seteo inmediato en el entorno para evitar errores de validación del LLM
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["TAVILY_API_KEY"] = tavily_key
        st.success("✅ APIs configuradas correctamente")

# 3. Definición del Estado y el Grafo
class AgentState(TypedDict):
    question: str
    search_results: str
    final_story: str

def tool_search_news(state: AgentState):
    """Busca en tiempo real usando Tavily"""
    # Se inicializa dentro del nodo para asegurar que use la API Key del sidebar
    search = TavilySearchResults(max_results=3)
    results = search.invoke(state["question"])
    return {"search_results": str(results)}

def generator_story(state: AgentState):
    """Transforma las noticias en un cuento simple"""
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
    
    prompt = f"""
    Eres un narrador experto que explica tecnología a jovencitos de 10 años.
    Usa términos muy simples, metáforas y cuenta una historia emocionante.
    
    CONTEXTO DE NOTICIAS:
    {state['search_results']}
    
    TEMA A EXPLICAR:
    {state['question']}
    
    INSTRUCCIÓN: Explica qué ha pasado como si fuera un cuento corto.
    """
    
    response = llm.invoke(prompt)
    return {"final_story": response.content}

# Construcción del flujo
workflow = StateGraph(AgentState)
workflow.add_node("buscador", tool_search_news)
workflow.add_node("escritor", generator_story)

workflow.set_entry_point("buscador")
workflow.add_edge("buscador", "escritor")
workflow.add_edge("escritor", END)

app_graph = workflow.compile()

# 4. Interfaz de Usuario (Input y Ejecución)
if google_key and tavily_key:
    pregunta = st.text_input("¿Qué quieres entender hoy?", 
                             placeholder="Ej: ¿Qué es Sora de OpenAI?")

    if pregunta:
        with st.spinner("🕵️‍♀️ Buscando noticias y escribiendo tu historia..."):
            try:
                # Ejecución del grafo
                inputs = {"question": pregunta}
                resultado = app_graph.invoke(inputs)
                
                # Resultado principal
                st.markdown("---")
                st.subheader("📖 Tu cuento de IA:")
                st.write(resultado["final_story"])
                
                # Trazabilidad técnica
                with st.expander("🛠️ Ver datos técnicos (Fuentes de Tavily)"):
                    st.code(resultado["search_results"], language="text")
            
            except Exception as e:
                st.error(f"Hubo un error al generar la historia: {str(e)}")
                st.info("Revisa que tus API Keys sean correctas y tengan créditos.")

else:
    st.warning("👈 Introduce tus claves de Google y Tavily en el menú de la izquierda para empezar.")
