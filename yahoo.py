#!/usr/bin/env python3
"""
Stock Technical Analysis App
Downloads stock data and calculates technical indicators using yfinance
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TechnicalAnalyzer:
    def __init__(self, period, data_dir:Path):
        """
        Initialize the Technical Analyzer

        Args:
            period (str): Time period for data download ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        """
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
                data = ticker.history(period=self.period)

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

    def calculate_sma(self, data, window):
        """Calculate Simple Moving Average"""
        return data.rolling(window=window).mean()

    def calculate_ema(self, data, window):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=window).mean()

    def calculate_rsi(self, data, window=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1 / window, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / window, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, data, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return upper_band, sma, lower_band

    def calculate_stochastic(self, high, low, close, k_window=14, d_window=3, k_smoothing_period=3):
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))

        # Smooth %K
        k_percent_smoothed = k_percent.rolling(window=k_smoothing_period).mean()
        d_percent = k_percent_smoothed.rolling(window=d_window).mean()

        return k_percent_smoothed, d_percent

    def calculate_atr(self, high, low, close, window=14):
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()

        return atr

    def calculate_cci(self, high, low, close, window=20):
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)

        return cci

    def calculate_obv(self, data):
        """Calculate On-Balance Volume (OBV)"""
        price_diff = data['Close'].diff()
        flow = data['Volume'] * np.sign(price_diff)
        obv = flow.cumsum().fillna(0)
        return obv

    def add_technical_indicators(self, data, symbol):
        """Add all technical indicators to the dataframe"""
        df = data.copy()

        print(f"Calculating indicators for {symbol}...")

        # Moving Averages
        df['SMA_5'] = self.calculate_sma(df['Close'], 5)
        df['SMA_20'] = self.calculate_sma(df['Close'], 20)
        df['EMA_12'] = self.calculate_ema(df['Close'], 12)
        df['EMA_26'] = self.calculate_ema(df['Close'], 26)

        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])

        # MACD
        macd_line, signal_line, histogram = self.calculate_macd(df['Close'])
        df['MACD'] = macd_line
        df['MACD_Signal'] = signal_line
        df['MACD_Histogram'] = histogram

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower

        # Stochastic
        stoch_k, stoch_d = self.calculate_stochastic(df['High'], df['Low'], df['Close'], k_window=10, d_window=3,
                                                     k_smoothing_period=3)
        df['Stoch_K'] = stoch_k
        df['Stoch_D'] = stoch_d

        # ATR
        df['ATR'] = self.calculate_atr(df['High'], df['Low'], df['Close'])

        # CCI
        df['CCI'] = self.calculate_cci(df['High'], df['Low'], df['Close'])

        # OBV
        df['OBV'] = self.calculate_obv(df)

        # Volume indicators
        df['Volume_SMA_20'] = self.calculate_sma(df['Volume'], 20)

        # Price-based indicators
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Range'] = df['High'] - df['Low']
        df['Gap'] = df['Open'] - df['Close'].shift(1)

        return df

    def create_technical_chart(self, data, symbol):
        """Create comprehensive technical analysis chart"""
        print(f"Creating chart for {symbol}...")

        # Set up the figure with subplots
        fig = plt.figure(figsize=(16, 22))  # Increased height for the new chart
        gs = GridSpec(7, 1, height_ratios=[3, 1, 1, 1, 1, 1, 1], hspace=0)

        # Convert index to datetime if it isn't already
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        x_indices = np.arange(len(data.index))

        # 1. Price Chart
        ax1 = fig.add_subplot(gs[0])
        for i in range(len(data)):
            open_price, close_price = data['Open'].iloc[i], data['Close'].iloc[i]
            high_price, low_price = data['High'].iloc[i], data['Low'].iloc[i]
            color = 'green' if close_price >= open_price else 'red'
            ax1.plot([x_indices[i], x_indices[i]], [low_price, high_price], color='black', linewidth=0.5, alpha=0.7)
            ax1.bar(x_indices[i], abs(close_price - open_price), bottom=min(open_price, close_price), width=0.8,
                    color=color, alpha=0.7)
        ax1.plot(x_indices, data['SMA_5'], label='SMA 5', color='orange', linewidth=2)
        ax1.plot(x_indices, data['SMA_20'], label='SMA 20', color='blue', linewidth=2)
        ax1.fill_between(x_indices, data['BB_Upper'], data['BB_Lower'], alpha=0.1, color='gray',
                         label='Bollinger Bands')
        ax1.plot(x_indices, data['BB_Upper'], color='gray', linewidth=1, alpha=0.7)
        ax1.plot(x_indices, data['BB_Middle'], color='gray', linewidth=1, alpha=0.7)
        ax1.plot(x_indices, data['BB_Lower'], color='gray', linewidth=1, alpha=0.7)
        ax1.set_title(f'{symbol} - Technical Analysis Chart', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # 2. Volume Chart
        ax2 = fig.add_subplot(gs[1])
        volume_colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red' for i in range(len(data))]
        ax2.bar(x_indices, data['Volume'], color=volume_colors, alpha=0.7, width=0.8)
        ax2.plot(x_indices, data['Volume_SMA_20'], color='blue', linewidth=2, label='Volume SMA 20')
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 3. MACD Chart
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(x_indices, data['MACD'], color='blue', linewidth=2, label='MACD')
        ax3.plot(x_indices, data['MACD_Signal'], color='red', linewidth=2, label='Signal')
        macd_hist_colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
        ax3.bar(x_indices, data['MACD_Histogram'], color=macd_hist_colors, alpha=0.6, width=0.8, label='Histogram')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_ylabel('MACD', fontsize=12)
        ax3.legend(loc='upper left', fontsize=10)
        ax3.grid(True, alpha=0.3)

        # 4. Stochastic Oscillator
        ax4 = fig.add_subplot(gs[3])
        ax4.plot(x_indices, data['Stoch_K'], color='blue', linewidth=2, label='%K')
        ax4.plot(x_indices, data['Stoch_D'], color='red', linewidth=2, label='%D')
        ax4.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Overbought (80)')
        ax4.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Oversold (20)')
        ax4.fill_between(x_indices, 80, 100, alpha=0.1, color='red')
        ax4.fill_between(x_indices, 0, 20, alpha=0.1, color='green')
        ax4.set_ylabel('Stochastic', fontsize=12)
        ax4.set_ylim(0, 100)
        ax4.legend(loc='upper left', fontsize=10)
        ax4.grid(True, alpha=0.3)

        # 5. ATR and CCI Chart
        ax5 = fig.add_subplot(gs[4])
        ax5.plot(x_indices, data['ATR'], color='orange', linewidth=2, label='ATR (14)')
        ax5.set_ylabel('ATR', fontsize=12, color='orange')
        ax5.tick_params(axis='y', labelcolor='orange')
        ax5.legend(loc='upper left', fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5_twin = ax5.twinx()
        ax5_twin.plot(x_indices, data['CCI'], color='purple', linewidth=2, label='CCI (20)')
        ax5_twin.set_ylabel('CCI', fontsize=12, color='purple')
        ax5_twin.tick_params(axis='y', labelcolor='purple')
        ax5_twin.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Overbought (100)')
        ax5_twin.axhline(y=-100, color='green', linestyle='--', alpha=0.7, label='Oversold (-100)')
        ax5_twin.legend(loc='upper right', fontsize=10)

        # 6. RSI Chart
        ax6 = fig.add_subplot(gs[5])
        ax6.plot(x_indices, data['RSI'], color='purple', linewidth=2, label='RSI (14)')
        ax6.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax6.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax6.fill_between(x_indices, 70, 100, alpha=0.1, color='red')
        ax6.fill_between(x_indices, 0, 30, alpha=0.1, color='green')
        ax6.set_ylabel('RSI', fontsize=12)
        ax6.set_ylim(0, 100)
        ax6.legend(loc='upper left', fontsize=10)
        ax6.grid(True, alpha=0.3)

        # 7. OBV Chart
        ax7 = fig.add_subplot(gs[6])
        ax7.plot(x_indices, data['OBV'], color='purple', linewidth=2, label='OBV')
        ax7.set_ylabel('OBV', fontsize=12)
        ax7.legend(loc='upper left', fontsize=10)
        ax7.grid(True, alpha=0.3)

        # Format x-axis for all subplots
        all_axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
        for ax in all_axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            def format_fn(value, tick_number):
                if int(value) in x_indices:
                    return data.index[int(value)].strftime('%Y-%m-%d')
                return ''
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_fn))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Only show x-axis labels on the bottom chart
        for ax in all_axes[:-1]:
            ax.set_xticklabels([])

        all_axes[-1].set_xlabel('Date', fontsize=12)

        # Adjust layout and save
        plt.tight_layout()

        # Save the chart
        chart_filename = f"{symbol}_technical_chart.png"
        chart_filepath = os.path.join(self.data_dir, chart_filename)

        plt.savefig(chart_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # Close the figure to free memory

        print(f"✓ Chart saved to {chart_filepath}")

        """Save processed data to CSV file"""
        filename = f"{symbol}_technical_analysis.csv"
        filepath = os.path.join(self.data_dir, filename)

        # Ensure output directory exists
        os.makedirs(self.data_dir, exist_ok=True)

        # Round numerical columns to 4 decimal places
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data[numeric_columns] = data[numeric_columns].round(4)

        data.to_csv(filepath)
        return chart_filepath

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

    def run_analysis(self, symbols):
        """Run the complete technical analysis"""
        print(f"Starting Technical Analysis for {len(symbols)} symbols")
        print(f"Period: {self.period}")
        print(f"Output directory: {self.data_dir}")
        print("-" * 50)

        # Download data
        stock_data = self.download_stock_data(symbols)

        if not stock_data:
            print("No data downloaded. Exiting.")
            return

        print(f"\nProcessing {len(stock_data)} stocks...")
        print("-" * 50)

        processed_files = []
        chart_files = []
        summary_data = []

        # Process each stock
        for symbol, data in stock_data.items():
            try:
                # Calculate technical indicators
                processed_data = self.add_technical_indicators(data, symbol)

                # Create technical chart
                chart_path = self.create_technical_chart(processed_data, symbol)
                chart_files.append(chart_path)

                # Save to CSV
                filepath = self.save_to_csv(processed_data, symbol)
                processed_files.append(filepath)

                # Generate summary
                summary = self.generate_summary_stats(processed_data, symbol)
                summary_data.append(summary)

            except Exception as e:
                print(f"✗ Error processing {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()

        # Save summary file
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(self.data_dir, f"stock_summary_{datetime.now().strftime('%Y%m%d')}.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"\n✓ Summary saved to {summary_file}")

        print(f"\nAnalysis complete! Processed {len(processed_files)} stocks.")
        print(f"Generated {len(chart_files)} charts.")
        print(f"Files saved to: {self.data_dir}")

        return processed_files, chart_files, summary_data


