import yfinance as yf
import pandas as pd
import os

def download_data(ticker, start_date, end_date, output_file):
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {ticker}. Please check the symbol.")
            return

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        data.to_csv(output_file)
        print(f"Data saved to {output_file}")
        print(data.tail())
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    # Configuration
    TICKER = "AAPL" # Default to Apple, can be changed to "^N225" for Nikkei 225
    START_DATE = "2020-01-01"
    END_DATE = "2023-12-31"
    OUTPUT_FILE = "data/stock_data.csv"

    download_data(TICKER, START_DATE, END_DATE, OUTPUT_FILE)
