import os
import getpass
import functools
from typing import Annotated, Literal, TypedDict

# Importaciones actualizadas
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display

# --- CONFIGURACIÓN DE LLAVES ---
if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = getpass.getpass('Introduce tu Google API Key: ')

# Tavily API Key
os.environ["TAVILY_API_KEY"] = "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"

# --- DEFINICIÓN DEL ESTADO ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# --- HERRAMIENTAS ---
tools = [TavilySearchResults(max_results=5)]

# --- LÓGICA DE AGENTES ---
def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="messages"),
    ])
    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm

# --- PERSONALIDADES ---
investigator_template = "Eres un Investigador de Campo de lo Paranormal. Busca evidencias sobre el misterio del usuario. No redactes informes, solo recopila datos y usa herramientas si es necesario."
analyst_template = "Eres un Analista de lo Oculto. Toma los datos y crea un esquema con: Origen, Testigos y Teorías."
archivist_template = "Eres el Archivista de Casos No Resueltos. Escribe un informe final estilo SCP (ID CASO, ESTADO, INFORME, VERDICTO) con tono oscuro."

# Inicialización (Usamos gemini-1.5-flash para mayor estabilidad)
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

investigator_agent = create_agent(llm, tools, investigator_template)
analyst_agent = create_agent(llm, [], analyst_template)
archivist_agent = create_agent(llm, [], archivist_template)

# --- NODOS ---
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [result]}

investigator_node = functools.partial(agent_node, agent=investigator_agent, name="Investigador")
analyst_node = functools.partial(agent_node, agent=analyst_agent, name="Analista")
archivist_node = functools.partial(agent_node, agent=archivist_agent, name="Archivista")
tool_node = ToolNode(tools)

def should_search(state: AgentState):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return "analyst"

# --- CONSTRUCCIÓN DEL GRAFO ---
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

# --- PRUEBA ---
inputs = {"messages": [HumanMessage(content="El misterio de las Caras de Bélmez")]}
for event in graph.stream(inputs, stream_mode="values"):
    event['messages'][-1].pretty_print()
