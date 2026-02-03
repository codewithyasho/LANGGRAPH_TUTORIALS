import yfinance as yf


def get_stock_price(ticker_symbol: str):
    """
    Fetch live stock price using Yahoo Finance Ticker.
    Example inputs: 'TSLA' (Tesla), 'AAPL' (Apple), 'RELIANCE.NS' (Reliance)
    """
    # 1. Handle common user input errors (remove spaces, make uppercase)
    symbol = ticker_symbol.upper().strip()

    try:
        # 2. Create the Ticker object
        stock = yf.Ticker(symbol)

        # 3. Fetch history (1 day)
        history = stock.history(period="1d")

        # 4. Check if data exists (if ticker is wrong, history will be empty)
        if history.empty:
            return f"Error: No data found for symbol '{symbol}'. Check if the ticker is correct."

        # 5. Get the last closing price
        current_price = history["Close"].iloc[-1]

        return round(float(current_price), 2)

    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    # Test with Tesla (Ticker: TSLA)
    company_ticker = "TSLA"
    price = get_stock_price(company_ticker)

    print(f"The price for {company_ticker} is: {price}")
