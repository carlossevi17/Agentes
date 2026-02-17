import os
import getpass
import functools
from typing import Annotated, Literal, TypedDict

# Importaciones de LangChain y LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display

# --- CONFIGURACIÓN DE LLAVES ---
if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = getpass.getpass('Introduce tu Google API Key: ')

# Usando la API Key de Tavily proporcionada en el notebook original
os.environ["TAVILY_API_KEY"] = "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"

# --- DEFINICIÓN DEL ESTADO ---
class AgentState(TypedDict):
    # Annotated permite que los mensajes se acumulen en lugar de sobrescribirse
    messages: Annotated[list, add_messages]

# --- HERRAMIENTAS ---
# Configuración de búsqueda web con máximo 5 resultados
tools = [TavilySearchResults(max_results=5)]

# --- LÓGICA DE AGENTES ---
def create_agent(llm, tools, system_message: str):
    """Función para instanciar agentes con prompts específicos."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_message}"),
        MessagesPlaceholder(variable_name="messages"),
    ])
    prompt = prompt.partial(system_message=system_message)
    if tools:
        return prompt | llm.bind_tools(tools)
    else:
        return prompt | llm

# --- PERSONALIDADES (TEMÁTICA MISTERIO) ---
investigator_template = """Eres un Investigador de Campo de lo Paranormal. 
Tu misión es buscar evidencias y datos históricos sobre el mito que el usuario mencione.
NOTA: No redactes el informe final, solo recopila hechos y pásalos al Analista."""

analyst_template = """Eres un Analista de Fenómenos Inexplicables. 
Crea un esquema basado en los datos del Investigador: origen, testigos clave y teorías."""

archivist_template = """Eres el Archivista de Casos No Resueltos. 
Escribe el reporte final con tono oscuro y profesional (estilo SCP o Expediente X).
Formato:
ID CASO: <número>
ESTADO: <Clasificado>
INFORME: <body>
VERDICTO: <tu conclusión final>"""

# --- INICIALIZACIÓN ---
# Usando el modelo Gemini
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

investigator_agent = create_agent(llm, tools, investigator_template)
analyst_agent = create_agent(llm, [], analyst_template)
archivist_agent = create_agent(llm, [], archivist_template)

# --- NODOS DEL GRAFO ---
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {'messages': [result]}

investigator_node = functools.partial(agent_node, agent=investigator_agent, name="Investigador")
analyst_node = functools.partial(agent_node, agent=analyst_agent, name="Analista")
archivist_node = functools.partial(agent_node, agent=archivist_agent, name="Archivista")
tool_node = ToolNode(tools)

def should_search(state) -> Literal["tools", "analyst"]:
    """Determina si se debe ir a herramientas o pasar al análisis."""
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return "analyst"

# --- CONSTRUCCIÓN DEL FLUJO (LANGGRAPH) ---
workflow = StateGraph(AgentState)

workflow.add_node("investigator", investigator_node)
workflow.add_node("tools", tool_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("archivist", archivist_node)

workflow.set_entry_point("investigator")
workflow.add_conditional_edges("investigator", should_search)
workflow.add_edge("tools", "investigator")
workflow.add_edge("analyst", "archivist")
workflow.add_edge("archivist", END)

graph = workflow.compile()

# --- EJECUCIÓN ---
input_message = HumanMessage(content="El misterio del vuelo MH370")
for event in graph.stream({"messages": [input_message]}, stream_mode="values"):
    event['messages'][-1].pretty_print()
