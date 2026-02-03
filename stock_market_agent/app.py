"""
Stock Market Trading Agent - Simple Streamlit UI
"""

import streamlit as st
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated, List
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv
from datetime import datetime
import yfinance as yf

load_dotenv()

st.set_page_config(page_title="Stock Market Trading Agent",
                   page_icon="📈", layout="wide")


# ----------------- State and Tools Setup ----------------- #

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


# Tool 1 - get the current stock price for a given company
@tool
def get_stock_price(ticker_symbol: str, period: str = "1d") -> float:
    """
    This tool fetches the current stock price for a given ticker symbol. Returns the current stock price for the given ticker symbol.

    Args:
        ticker_symbol (str): The ticker symbol to look up.
        period (str): The period for which to fetch the stock price. Default is '1d' (1 day). for example '1d' : 1 Day, '5d': 5 Days, '1mo': 1 Month, '3mo': 3 Months, '6mo': 6 Months, '1y': 1 Year...

    returns:
        The current stock price of the given ticker symbol.

    Fetch live stock price using Yahoo Finance Ticker.
    Example inputs: 'TSLA' (Tesla), 'AAPL' (Apple), 'RELIANCE.NS' (Reliance), 'GOOGL' (Google), 'AMZN' (Amazon)
    """
    symbol = ticker_symbol.upper().strip()

    try:
        stock = yf.Ticker(symbol)
        history = stock.history(period=period)

        if history.empty:
            return f"Error: No data found for symbol '{symbol}'. Check if the ticker is correct."

        current_price = history["Close"].iloc[-1]
        return round(float(current_price), 2)

    except Exception as e:
        return f"An error occurred: {e}"


# Tool 2 - buy stocks for a given company
@tool
def buy_stocks(ticker_symbol: str, quantity: int, total_price: float) -> str:
    """
    This tool is used to buy a specified quantity of stocks for the given company.
    Args:
        ticker_symbol (str): The ticker symbol to buy stocks from.
        quantity (int): The number of stocks to buy.
        total_price (float): The total price of the stocks to buy.

    returns:
        Confirmation message of the purchase.
    """
    if quantity <= 0:
        return "❌ Error: Quantity must be positive"
    if total_price <= 0:
        return "❌ Error: Total price must be positive"

    decision = interrupt(
        f"Do you want to buy {quantity} shares of {ticker_symbol} for ${total_price}? (yes/no)")

    if decision.lower() == "yes":
        return f"✅ You bought {quantity} shares of {ticker_symbol} for ${total_price}."
    else:
        return "❌ Transaction cancelled."


# Tool 3 - sell stocks for a given company
@tool
def sell_stocks(ticker_symbol: str, quantity: int, total_price: float) -> str:
    """
    This tool is used to sell a specified quantity of stocks for the given ticker symbol.
    Args:
        ticker_symbol (str): The ticker symbol to sell stocks from.
        quantity (int): The number of stocks to sell.
        total_price (float): The total price of the stocks to sell.

    returns:
        Confirmation message of the sold stocks.
    """
    if quantity <= 0:
        return "❌ Error: Quantity must be positive"
    if total_price <= 0:
        return "❌ Error: Total price must be positive"

    decision = interrupt(
        f"Do you want to sell {quantity} shares of {ticker_symbol} for ${total_price}? (yes/no)")

    if decision.lower() == "yes":
        return f"✅ You sold {quantity} shares of {ticker_symbol} for ${total_price}."
    else:
        return "❌ Transaction cancelled."


# tool 4 - get current date and time
@tool
def get_current_datetime() -> str:
    """
    This tool returns the current date and time as a string.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


# Toolkit of the agent
tools = [get_stock_price, buy_stocks, sell_stocks, get_current_datetime]


# ----------------- Initialize Session State ----------------- #

if 'memory' not in st.session_state:
    st.session_state.memory = MemorySaver()

if 'app' not in st.session_state:
    # LLM initialization
    llm = ChatGroq(model="openai/gpt-oss-120b",
                   temperature=0.3).bind_tools(tools=tools)

    # Agent node
    def agent(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(content="""
        You are a stock market trading agent. Your goal is to help users make informed decisions about buying and selling stocks.
        Answer the user queries using the available tools to get stock prices, buy stocks, sell stocks, and get the current date and time.
        Be accurate and concise in your responses.
        """)
        response = llm.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    # Graph Building
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)

    tool_node = ToolNode(tools=tools)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    st.session_state.app = graph.compile(checkpointer=st.session_state.memory)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'awaiting_decision' not in st.session_state:
    st.session_state.awaiting_decision = False

if 'pending_response' not in st.session_state:
    st.session_state.pending_response = None

if 'thread_id' not in st.session_state:
    st.session_state.thread_id = "thread_1"

# ----------------- Simple UI ----------------- #

st.title("📈 Stock Market Agent")
st.caption("Ask me about stock prices, buy/sell stocks, or get current time")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.awaiting_decision = False
    st.session_state.pending_response = None
    st.rerun()

st.divider()

# Display chat history
for message in st.session_state.chat_history:
    if message["type"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])

# Handle pending decision (buy/sell confirmation)
if st.session_state.awaiting_decision and st.session_state.pending_response:
    st.warning("⚠️ Confirm your action")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes", use_container_width=True):
            config = {"configurable": {
                "thread_id": st.session_state.thread_id}}
            response = st.session_state.app.invoke(
                Command(resume="yes"), config=config)
            st.session_state.chat_history.append({
                "type": "agent",
                "content": response['messages'][-1].content
            })
            st.session_state.awaiting_decision = False
            st.session_state.pending_response = None
            st.rerun()

    with col2:
        if st.button("❌ No", use_container_width=True):
            config = {"configurable": {
                "thread_id": st.session_state.thread_id}}
            response = st.session_state.app.invoke(
                Command(resume="no"), config=config)
            st.session_state.chat_history.append({
                "type": "agent",
                "content": response['messages'][-1].content
            })
            st.session_state.awaiting_decision = False
            st.session_state.pending_response = None
            st.rerun()

# User input
user_input = st.chat_input("What's the price of Tesla?",
                           disabled=st.session_state.awaiting_decision)

if user_input:
    # Show user message
    st.session_state.chat_history.append(
        {"type": "user", "content": user_input})

    # Get agent response
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.spinner("Thinking..."):
        response = st.session_state.app.invoke({
            "messages": [{"role": "user", "content": user_input}]
        }, config=config)

    # Check if needs confirmation
    if response.get("__interrupt__"):
        agent_msg = response['messages'][-1].content
        st.session_state.chat_history.append(
            {"type": "agent", "content": agent_msg})
        st.session_state.awaiting_decision = True
        st.session_state.pending_response = response
    else:
        agent_msg = response['messages'][-1].content
        st.session_state.chat_history.append(
            {"type": "agent", "content": agent_msg})

    st.rerun()
