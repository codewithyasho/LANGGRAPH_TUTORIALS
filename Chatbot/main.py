from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from langchain.tools import tool
from dotenv import load_dotenv
from datetime import datetime

import sqlite3

load_dotenv()


# TOOLS
@tool()
def search_web(query: str) -> str:
    """This tool is useful for searching the web for latest and up-to-date information."""
    search = DuckDuckGoSearchRun()
    result = search.run(query)
    return result


@tool()
def get_date_time() -> str:
    """This tool is useful for getting the current date and time."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool()
def calculator_tool(expression: str) -> str:
    """This tool is useful for performing calculations."""
    try:
        result = eval(expression)
        return str(result)

    except Exception as e:
        return f"Error in calculation: {str(e)}"


tools = [search_web, get_date_time, calculator_tool]


llm = ChatGroq(model="openai/gpt-oss-120b",
               temperature=0.8).bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}


# Checkpointer
# create the database file if it doesn't exist
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

tools = ToolNode(tools=tools)
graph.add_node("tools", tools)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")

graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def get_threads_in_db():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
