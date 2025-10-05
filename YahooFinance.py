#!/usr/bin/env python3
"""
Stock Technical Analysis App
Downloads stock data and calculates technical indicators using yfinance
"""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yfinance as yf

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class YahooFinance:
    def __init__(self, period, interval, data_dir: Path):
        """
        Initialize the Technical Analyzer

        Args:
            period (str): Time period for data download ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): 1d or 1wk
        """
        self.interval = interval
        self.period = period
        self.data_dir = data_dir

    def download_stock_data(self, symbols):
        """Download stock data for given symbols"""
        print(f"Downloading data for {len(symbols)} symbols...")

        stock_data = {}
        failed_symbols = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=self.period, interval=self.interval)

                if data.empty:
                    print(f"No data found for {symbol}")
                    failed_symbols.append(symbol)
                    continue

                stock_data[symbol] = data
                print(f"✓ Downloaded {symbol}: {len(data)} rows")

            except Exception as e:
                print(f"✗ Failed to download {symbol}: {str(e)}")
                failed_symbols.append(symbol)

        if failed_symbols:
            print(f"\nFailed to download: {', '.join(failed_symbols)}")

        return stock_data


    def save_to_csv(self, data, symbol):
        """Save processed data to CSV file"""
        filename = f"{symbol}_technical_analysis.csv"
        filepath = os.path.join(self.data_dir, filename)

        # Ensure output directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Round numerical columns to 4 decimal places
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].round(4)

        data.to_csv(filepath)
        print(f"✓ Saved {symbol} data to {filepath}")

        return filepath

    def generate_summary_stats(self, data, symbol):
        """Generate summary statistics for the stock"""
        latest_data = data.iloc[-1]

        summary = {
            'Symbol': symbol,
            'Date': latest_data.name.strftime('%Y-%m-%d'),
            'Close': latest_data['Close'],
            'Volume': int(latest_data['Volume']),
            'RSI': latest_data['RSI'],
            'MACD': latest_data['MACD'],
            'BB_Position': ((latest_data['Close'] - latest_data['BB_Lower']) /
                            (latest_data['BB_Upper'] - latest_data['BB_Lower'])) * 100,
            'ATR': latest_data['ATR'],
            '20_Day_Change_%': ((latest_data['Close'] / data['Close'].iloc[-21]) - 1) * 100 if len(
                data) > 20 else np.nan
        }

        return summary



