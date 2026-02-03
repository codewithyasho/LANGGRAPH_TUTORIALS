# Stock Market Trading Agent

A production-ready AI agent built with LangGraph that helps users make informed stock trading decisions with Human-in-the-Loop (HITL) confirmations for all transactions.

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0.7+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Overview

This Stock Market Trading Agent is an intelligent conversational AI system that integrates real-time stock market data with safe transaction handling. Built using LangGraph's state management and workflow capabilities, it ensures all buy/sell decisions require explicit human confirmation before execution.

**Key Highlights:**

- ✅ Real-time stock price fetching via Yahoo Finance
- ✅ Human-in-the-Loop (HITL) for transaction safety
- ✅ Memory-enabled conversations
- ✅ Two interfaces: CLI and Streamlit web app
- ✅ Production-grade error handling
- ✅ Input validation for all transactions

## ✨ Features

### Core Capabilities

1. **Stock Price Queries**: Get current prices for any publicly traded stock
2. **Historical Data**: Fetch stock prices for various time periods (1d, 5d, 1mo, 3mo, 6mo, 1y)
3. **Buy Stocks**: Purchase stocks with confirmation workflow
4. **Sell Stocks**: Sell stocks with confirmation workflow
5. **Date/Time**: Get current timestamp for trading records

### User Interfaces

- **CLI Application** (`Stock_Market_Agent.py`): Command-line interface for quick interactions
- **Web Application** (`stock_market_agent_app.py`): Modern Streamlit UI with chat interface

## 🏗️ Architecture

![Graph Architecture](Graph1.png)

### State Management

- Uses `MemorySaver` for conversation persistence
- Thread-based memory for multi-user support
- State contains annotated message list

### Tools Pipeline

1. **get_stock_price**: Fetches real-time stock data
2. **buy_stocks**: Initiates buy transaction with HITL
3. **sell_stocks**: Initiates sell transaction with HITL
4. **get_current_datetime**: Returns current timestamp

## 📦 Installation

### Prerequisites

- Python 3.12 or higher
- Ollama installed with DeepSeek model
- Internet connection for stock data

### Step 1: Clone or Download

```bash
cd LANGGRAPH_TUT
```

### Step 2: Install Dependencies

Using pip:

```bash
pip install langgraph langchain langchain-ollama langchain-community yfinance python-dotenv streamlit
```

Or using the project file:

```bash
pip install -e .
```

### Step 3: Set Up Ollama

Install Ollama and pull the DeepSeek model:

```bash
ollama pull deepseek-v3.1:671b-cloud
```

### Step 4: Environment Variables (Optional)

Create a `.env` file if you need custom configurations:

```env
# Optional: Add any API keys or custom settings
OLLAMA_HOST=http://localhost:11434
```

## 🚀 Usage

### CLI Application

Run the command-line interface:

```bash
python Stock_Market_Agent.py
```

**Example Session:**

```
Enter your query: What's the current price of TSLA?
AGENT: The current stock price of Tesla (TSLA) is $245.32

Enter your query: Buy 10 shares of TSLA
AGENT: Based on the current price of $245.32, 10 shares would cost $2,453.20

Do you want to Proceed (yes/no): yes
AGENT: ✅You bought 10 shares of TSLA for $2453.20.

Enter your query: exit
Session ended.
```

### Streamlit Web Application

Launch the web interface:

```bash
streamlit run stock_market_agent_app.py
```

The app will open in your browser at `http://localhost:8501`

**Features:**

- Chat-style interface
- Clear chat history button
- Visual confirmation dialogs for transactions
- Real-time responses
- Mobile-responsive design

## 🛠️ Tools & Capabilities

### 1. Get Stock Price

```python
get_stock_price(ticker_symbol: str, period: str = "1d") -> float
```

**Description**: Fetches current or historical stock prices

**Parameters:**

- `ticker_symbol`: Stock ticker (e.g., "TSLA", "AAPL", "RELIANCE.NS")
- `period`: Time period - "1d", "5d", "1mo", "3mo", "6mo", "1y"

**Example Queries:**

- "What's the price of TSLA?"
- "Get Apple stock price for the last month"
- "Show me GOOGL price for the past 6 months"

### 2. Buy Stocks

```python
buy_stocks(ticker_symbol: str, quantity: int, total_price: float) -> str
```

**Description**: Initiates stock purchase with confirmation

**Validation:**

- Quantity must be positive
- Total price must be positive

**Example Queries:**

- "Buy 10 shares of AAPL"
- "I want to purchase 5 MSFT stocks"

### 3. Sell Stocks

```python
sell_stocks(ticker_symbol: str, quantity: int, total_price: float) -> str
```

**Description**: Initiates stock sale with confirmation

**Validation:**

- Quantity must be positive
- Total price must be positive

**Example Queries:**

- "Sell 10 shares of TSLA"
- "I want to sell 3 GOOGL stocks"

### 4. Get Current DateTime

```python
get_current_datetime() -> str
```

**Description**: Returns current date and time

**Example Query:**

- "What time is it?"
- "Show me today's date"

## 📁 Project Structure

```
LANGGRAPH_TUT/
├── Stock_Market_Agent.py          # CLI application
├── stock_market_agent_app.py      # Streamlit web app
├── stock_market_agent.md          # This README file
├── pyproject.toml                 # Project dependencies
├── .env                           # Environment variables (optional)
├── Agents/                        # Learning materials
│   ├── Agent1_simple_agent.ipynb
│   ├── Agent2_dual_agents.ipynb
│   ├── ...
│   └── Agent9_HITL.ipynb
└── Graphs/                        # Graph tutorials
    ├── graph1_hello_world.ipynb
    ├── graph2_drive_car.ipynb
    └── ...
```

## ⚙️ Configuration

### Changing the LLM Model

Edit the model in both applications:

```python
# In Stock_Market_Agent.py or stock_market_agent_app.py
llm = ChatOllama(
    model="deepseek-v3.1:671b-cloud",  # Change this
    temperature=0.3
).bind_tools(tools=tools)
```

Available Ollama models:

- `llama2`
- `mistral`
- `gemma`
- `deepseek-v3.1:671b-cloud` (recommended for this project)

### Adjusting Temperature

Higher temperature = more creative responses (0.0 to 1.0):

```python
llm = ChatOllama(model="...", temperature=0.7)  # More creative
llm = ChatOllama(model="...", temperature=0.1)  # More focused
```

### Multi-User Support

For multiple users, use different thread IDs:

```python
config = {
    "configurable": {"thread_id": f"user_{user_id}"}
}
```

## 💡 Examples

### Example 1: Check Stock Price

```
User: What's the current price of Tesla?
Agent: The current stock price of Tesla (TSLA) is $245.32
```

### Example 2: Buy Stocks with Confirmation

```
User: I want to buy 10 shares of Apple
Agent: The current price of AAPL is $187.50. 10 shares would cost $1,875.00

Do you want to buy 10 shares of AAPL for $1875.0? (yes/no)
User: yes
Agent: ✅You bought 10 shares of AAPL for $1875.0.
```

### Example 3: Historical Price Data

```
User: Show me Google's stock price for the last 3 months
Agent: The stock price of GOOGL for the last 3 months is $142.85
```

### Example 4: Cancelled Transaction

```
User: Sell 5 shares of Microsoft
Agent: The current price of MSFT is $420.15. 5 shares would be $2,100.75

Do you want to sell 5 shares of MSFT for $2100.75? (yes/no)
User: no
Agent: ❌ Transaction cancelled.
```

## 🔧 Technologies Used

| Technology                   | Purpose                                     |
| ---------------------------- | ------------------------------------------- |
| **LangGraph**                | State management and workflow orchestration |
| **LangChain**                | LLM integration and tool binding            |
| **Ollama**                   | Local LLM inference (DeepSeek model)        |
| **Yahoo Finance (yfinance)** | Real-time stock market data                 |
| **Streamlit**                | Web application framework                   |
| **Python 3.12+**             | Core programming language                   |

## 🐛 Troubleshooting

### Issue: "No data found for symbol"

**Solution**: Verify the ticker symbol is correct. For international stocks, use proper suffix (e.g., `.NS` for Indian stocks)

### Issue: Ollama connection error

**Solution**: Ensure Ollama is running:

```bash
ollama serve
```

### Issue: Model not found

**Solution**: Pull the required model:

```bash
ollama pull deepseek-v3.1:671b-cloud
```

### Issue: Streamlit port already in use

**Solution**: Run on a different port:

```bash
streamlit run stock_market_agent_app.py --server.port 8502
```

### Issue: Import errors

**Solution**: Reinstall dependencies:

```bash
pip install --upgrade langgraph langchain langchain-ollama yfinance streamlit
```

## 🚧 Future Enhancements

### Planned Features

- [ ] Portfolio tracking and management
- [ ] Transaction history logging to database
- [ ] Stock analysis tools (RSI, MACD, Moving Averages)
- [ ] News sentiment analysis integration
- [ ] Multi-currency support
- [ ] Paper trading simulation mode
- [ ] Email/SMS notifications for price alerts
- [ ] Chart visualization for historical prices
- [ ] Backtesting capabilities
- [ ] Integration with real brokerage APIs

### Technical Improvements

- [ ] Add unit tests with pytest
- [ ] Implement logging framework
- [ ] Database integration (SQLite/PostgreSQL)
- [ ] Docker containerization
- [ ] CI/CD pipeline setup
- [ ] API rate limiting and caching
- [ ] User authentication for web app
- [ ] Real-time WebSocket updates

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2026 LangGraph Tutorial Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

## ⚠️ Disclaimer

**This application is for educational purposes only.** It is not financial advice and should not be used for actual trading without proper research and consultation with financial advisors. The creators are not responsible for any financial losses incurred while using this software.

## 📞 Support

For questions or issues:

- Open an issue on GitHub
- Check existing documentation
- Review the tutorial notebooks in `Agents/` and `Graphs/` folders

## 🙏 Acknowledgments

- **LangChain** team for the amazing framework
- **Ollama** for local LLM inference
- **Yahoo Finance** for stock market data
- **Streamlit** for the web framework
- **DeepSeek** for the powerful language model

---

**Built with ❤️ using LangGraph | Last Updated: February 2026**
