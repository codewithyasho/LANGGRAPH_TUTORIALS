"""
Stock Market Trading Agent with Human-in-the-Loop Decision Making
"""

from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from typing import TypedDict, Annotated, List
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from dotenv import load_dotenv
from datetime import datetime
import yfinance as yf
load_dotenv()

# memory for the agent
memory = MemorySaver()


# state initialization
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


# ----------------- Tools for the agent ----------------- #

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

    1. Create a Ticker object using yfinance library.
    2. Fetch the stock history according to the specified period.
    3. Extract the closing price from the history data.
    """
    # 1. Handle common user input errors (remove spaces, make uppercase)
    symbol = ticker_symbol.upper().strip()

    try:
        # 2. Create the Ticker object
        stock = yf.Ticker(symbol)

        # 3. Fetch history (1 day)
        history = stock.history(period=period)

        # 4. Check if data exists (if ticker is wrong, history will be empty)
        if history.empty:
            return f"Error: No data found for symbol '{symbol}'. Check if the ticker is correct."

        # 5. Get the last closing price
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

    decision = interrupt(
        f"Do you want to buy {quantity} shares of {ticker_symbol} for ${total_price}? (yes/no)")

    if decision.lower() == "yes":
        return f"✅You bought {quantity} shares of {ticker_symbol} for ${total_price}."
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
    decision = interrupt(
        f"Do you want to sell {quantity} shares of {ticker_symbol} for ${total_price}? (yes/no)")

    if decision.lower() == "yes":
        return f"✅You sold {quantity} shares of {ticker_symbol} for ${total_price}."

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


# llm initialization
llm = ChatOllama(model="deepseek-v3.1:671b-cloud",
                 temperature=0.3).bind_tools(tools=tools)


# ----------------- agent node ----------------- #

def agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
    You are a stock market trading agent. Your goal is to help users make informed decisions about buying and selling stocks.
    Answer the user queries using the available tools to get stock prices, buy stocks, sell stocks, and get the current date and time.
    Be accurate and concise in your responses.
    """)

    response = llm.invoke([system_prompt] + state["messages"])

    return {"messages": [response]}


# ----------------- Graph Building ----------------- #

graph = StateGraph(AgentState)

graph.add_node("agent", agent)

tools = ToolNode(tools=tools)
graph.add_node("tools", tools)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)

graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=memory)


# configuring threads for memory
config = {
    "configurable": {"thread_id": "thread_1"},
}


if __name__ == "__main__":
    # ----------------- Invoking the agent ----------------- #
    while True:

        # normal user query
        user_input = input("\nEnter your query: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break

        response1 = app.invoke({
            "messages": [{"role": "user", "content": user_input}]
        }, config=config)

        isinterrupt = response1.get("__interrupt__")

        # human in the loop decision
        if isinterrupt:
            print(f"\nAGENT: {response1['messages'][-1].content}")

            decision = input("\nDo you want to Proceed (yes/no): ")

            response2 = app.invoke(Command(resume=decision), config=config)

            print(f"\nAGENT: {response2['messages'][-1].content}")

        else:
            print(f"\nAGENT: {response1['messages'][-1].content}")

    print("\nSession ended.")
