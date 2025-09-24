#!/usr/bin/env python3
"""
Gemini Stock Analysis Advisor
Sends technical analysis data to Google Gemini for buy/sell/hold recommendations
"""
import glob
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
from google.genai import Client
from google.genai.types import GenerateContentConfig

from get_recommended_action import get_recommendation_action

warnings.filterwarnings('ignore')

GEMINI_FLASH = "gemini-2.5-flash"
GEMINI_PRO = "gemini-2.5-pro"

class AnalyzedResult:
    def __init__(self, symbol, file, current_price, recommendation, timestamp):
        self.symbol = symbol
        self.file = file
        self.current_price = current_price
        self.timestamp = timestamp

        self.recommendation = recommendation.removeprefix('```html').removesuffix('```')
        self.action = get_recommendation_action(self.recommendation)

        match = re.search(r'RECOMMENDATION:\s*(.+?)(?=\n|$)', self.recommendation, re.DOTALL | re.IGNORECASE)

        if match:
            self.action = match.group(1)
        else:
            pattern = r'<div class="recommendation hold">\s*([A-Z]+)\s*</div>'
            match = re.search(pattern, self.recommendation, re.DOTALL | re.IGNORECASE)

            if match:
                self.action = match.group(1)
            else:
                pattern = r'<td>Recommendation:</td>\s*<td><strong>([^<]+)</strong></td>'
                match = re.search(pattern, self.recommendation, re.DOTALL | re.IGNORECASE)

                if match:
                    result = match.group(1)
                    print(result)  # "BUY
                else:
                    self.action = 'Unknown'

class GeminiStockAdvisor:
    def __init__(self, working_dir:Path):
        """
        Initialize the Gemini Stock Advisor
        """
        self.data_dir = working_dir
        self.system_prompt = ('You are a professional stock analyst with expertise in technical analysis. '
                              'Based on the technical indicators and data provided by the user, provide a clear BUY/SELL/HOLD recommendation.')

    def find_latest_csv_files(self, symbol):
        """Find the latest technical analysis CSV files"""
        pattern = os.path.join(self.data_dir, f"{symbol}_technical_analysis.csv")

        files = glob.glob(pattern)

        if not files:
            if symbol:
                print(f"No technical analysis files found for {symbol} in {self.data_dir}")
            else:
                print(f"No technical analysis files found in {self.data_dir}")
            return []

        # Sort by modification time to get the latest files
        files.sort(key=os.path.getmtime, reverse=True)

        if symbol:
            return files[:1]  # Return only the latest file for the specific symbol
        else:
            # For multiple symbols, get the latest file for each symbol
            symbols_processed = set()
            latest_files = []

            for file in files:
                filename = os.path.basename(file)
                symbol_name = filename.split('_')[0]

                if symbol_name not in symbols_processed:
                    latest_files.append(file)
                    symbols_processed.add(symbol_name)

            return latest_files

    def load_technical_data(self, csv_file):
        """Load and prepare technical analysis data"""
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

            # Get the symbol from filename
            filename = os.path.basename(csv_file)
            symbol = filename.split('_')[0]

            # Get the latest data point (most recent)
            latest = df.iloc[-1].copy()

            # Get some historical context (last 5 days)
            recent_data = df.tail(5)

            return symbol, latest, recent_data, df

        except Exception as e:
            print(f"Error loading {csv_file}: {str(e)}")
            return None, None, None, None

    def format_data_for_gemini(self, symbol, latest_data, recent_data, full_data):
        """Format the technical data for Gemini analysis"""

        # Calculate some additional metrics
        price_change_1d = ((latest_data['Close'] - recent_data['Close'].iloc[-2]) / recent_data['Close'].iloc[-2]) * 100
        price_change_5d = ((latest_data['Close'] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]) * 100

        # Volume analysis
        avg_volume_20d = recent_data['Volume_SMA_20'].iloc[-1] if 'Volume_SMA_20' in recent_data.columns else \
            recent_data['Volume'].mean()
        current_volume = latest_data['Volume']
        volume_ratio = current_volume / avg_volume_20d if avg_volume_20d > 0 else 1

        # Trend analysis
        sma_5_trend = "UP" if latest_data['Close'] > latest_data['SMA_5'] else "DOWN"
        sma_20_trend = "UP" if latest_data['Close'] > latest_data['SMA_20'] else "DOWN"

        # Bollinger Bands position
        bb_position = ((latest_data['Close'] - latest_data['BB_Lower']) /
                       (latest_data['BB_Upper'] - latest_data['BB_Lower'])) * 100

        formatted_data = f"""
# TECHNICAL ANALYSIS DATA FOR {symbol}

## === CURRENT PRICE DATA ===
* Current Price: ${latest_data['Close']:.2f}
* 1-Day Change: {price_change_1d:.2f}%
* 5-Day Change: {price_change_5d:.2f}%
* Daily High: ${latest_data['High']:.2f}
* Daily Low: ${latest_data['Low']:.2f}
* Volume: {int(current_volume):,} (Ratio to 20d avg: {volume_ratio:.2f}x)

## === MOVING AVERAGES ===
* SMA 5: ${latest_data['SMA_5']:.2f} (Price is {sma_5_trend})
* SMA 20: ${latest_data['SMA_20']:.2f} (Price is {sma_20_trend})
* EMA 12: ${latest_data['EMA_12']:.2f}
* EMA 26: ${latest_data['EMA_26']:.2f}

## === VOLATILITY & BANDS ===
* Bollinger Bands - Upper: ${latest_data['BB_Upper']:.2f}, Lower: ${latest_data['BB_Lower']:.2f}

## === RECENT PRICE TREND/TECH INDICATORS (Last 90 Days) ===
"""

        # Add recent price trend in a markdown table
        last_90_data = full_data.tail(90).copy()
        # Calculate BB Position for the last 90 days
        last_90_data['BB_Position'] = ((last_90_data['Close'] - last_90_data['BB_Lower']) /
                                       (last_90_data['BB_Upper'] - last_90_data['BB_Lower'])) * 100

        formatted_data += "| Day | Date       | Close   | RSI  | MACD Hist         | BB Pos  | ATR      | Stoch K  | Stoch D  | CCI      | Volume      |\n"
        formatted_data += "|-----|------------|---------|------|-------------------|---------|----------|----------|----------|----------|-------------|\n"


        for i, (date, row) in enumerate(last_90_data.iterrows()):
            trend = 'BULLISH' if row['MACD_Histogram'] > 0 else 'BEARISH'
            formatted_data += (f"| {i + 1:<3} | {date.strftime('%Y-%m-%d')} |"
                               f" ${row['Close']:<6.2f} | {row['RSI']:<4.1f} |"
                               f" {row['MACD_Histogram']:<9.4f} {trend:<7} | "
                               f" {row['BB_Position']:<6.1f}% | {row['ATR']:<6.4f} |"
                               f" {row['Stoch_K']:<7.1f} | {row['Stoch_D']:<7.1f} |"
                               f" {row['CCI']:<6.1f} | {row['Volume']:<11,} |\n")
        # print(formatted_data)
        return formatted_data

    async def get_gemini_recommendation(self, formatted_data):
        """Get buy/sell/hold recommendation from Gemini"""

        prompt = f"""
Given the data below: 

{formatted_data}

Please analyze this data and provide:

1. **RECOMMENDATION**: Clearly state BUY, SELL, or HOLD
2. **CONFIDENCE**: Rate your confidence (1-10 scale)
3. **KEY REASONS**: List 3-4 main technical factors supporting your decision
4. **RISK LEVEL**: Assess the current risk (LOW/MEDIUM/HIGH)
5. **TIME HORIZON**: Suggest if this is for short-term (days), medium-term (weeks), or long-term (months)
6. **KEY LEVELS**: Mention important support/resistance levels to watch

Keep your analysis concise but thorough. Focus on the technical indicators provided and their current signals.

Make your answer in html format so that I can email it
"""
        try:

            # Configure Gemini
            gemini_client = Client()
            config = GenerateContentConfig(
                seed=197,  # give it an odd number to cut down varieties of responses
                system_instruction=self.system_prompt,
                temperature=0)  # ignore
            response = await gemini_client.aio.models.generate_content(
                model=GEMINI_PRO, contents=prompt, config=config)

            return response.text

        except Exception as e:
            print(f"Error getting Gemini response: {str(e)}")
            return f"Error: Could not get analysis from Gemini. {str(e)}"

    async def analyze_single_stock(self, csv_file) -> Optional[AnalyzedResult]:
        """Analyze a single stock and get recommendation"""
        print(f"\nAnalyzing: {os.path.basename(csv_file)}")
        print("-" * 60)

        symbol, latest_data, recent_data, full_data = self.load_technical_data(csv_file)

        if symbol is None:
            return None

        # Format data for Gemini
        formatted_data = self.format_data_for_gemini(symbol, latest_data, recent_data, full_data)

        # Get Gemini recommendation
        recommendation = await self.get_gemini_recommendation(formatted_data)

        result = AnalyzedResult(
            symbol= symbol,
            file= csv_file,
            current_price= latest_data['Close'],
            recommendation= recommendation,
            timestamp=datetime.now().isoformat())

        return result

    async def analyze_multiple_stocks(self, symbol_list: List[str]) -> Dict[str, AnalyzedResult]:
        """Analyze multiple stocks and get recommendations"""

        csv_files = []
        for symbol in symbol_list:
            files = self.find_latest_csv_files(symbol)
            csv_files.extend(files)

        if not csv_files:
            print("No CSV files found for analysis.")
            return {}

        print(f"Found {len(csv_files)} files to analyze...")

        results: Dict[str, AnalyzedResult] = {}

        for csv_file in csv_files:
            try:
                result = await self.analyze_single_stock(csv_file)
                if result:
                    results[result.symbol] = result

                    # Print the recommendation
                    print(f"\n{result.symbol}: ${result.current_price:.2f}")
                    print("=" * 60)
                    print(result.recommendation)
                    print("\n" + "=" * 60)

            except Exception as e:
                print(f"Error analyzing {csv_file}: {str(e)}")

        return results

    def save_recommendations(self, results: Dict[str, AnalyzedResult]):
        """Save recommendations to a file"""
        if not results:
            return

        for k, v in results.items():
            output_file = os.path.join(self.data_dir,
                                       f"{k}_gemini_recommendations.html")

            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(v.recommendation)

                print(f"\nRecommendations saved to: {output_file}")

            except Exception as e:
                print(f"Error saving recommendations: {str(e)}")
