import glob
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv

from DeepseekStockAdvisor import DeepseekStockAdvisor
from GeminiStockAdvisor import GeminiStockAdvisor
from GmailSender import GmailSender
from TechnicalAnalyzer import TechnicalAnalyzer
from YahooFinance import YahooFinance
from config import get_llm_vendor, GEMINI, DEEPSEEK
from get_recommended_action import print_summary
from llm_base import AnalyzedResult

hidden_frank_ta = Path.home() / '.frank_ta'


class Orchestrator:
    def __init__(self, period, interval, directory: Path):
        self.interval = interval
        self.period = period
        self.directory = directory

    def delete_all_files(self):
        pattern = os.path.join(self.directory, "*")
        files = glob.glob(pattern)

        for file_path in files:
            if os.path.isfile(file_path):  # Only delete files, not directories
                os.remove(file_path)
                print(f"Deleted: {file_path}")

    async def run(self, symbols: List[str]):
        self.delete_all_files()

        load_dotenv(hidden_frank_ta / '.env')
        symbols.sort()
        await self.yahoo_fetch(symbols)

        if get_llm_vendor() == GEMINI:
            results = await self.gemini_advise(symbols)
        elif get_llm_vendor() == DEEPSEEK:
            await self.deepseek_advise(symbols)

        print_summary(self.directory)

        if False:
            for k, v in results.items():  # ignore
                await send_emails(k, v, data_dir)  # ignore
            pass

    async def send_emails(self, symbol, result: AnalyzedResult):
        # Initialize Gmail sender
        secret_file = hidden_frank_ta / 'google_client_secret.json'
        token_file = hidden_frank_ta / 'google_client_token.json'
        gmail = GmailSender(str(secret_file), str(token_file))

        # Email details
        sender_email = "none@gmail.com"  # Replace with your Gmail
        recipient_email = "none@gmail.com"  # Replace with recipient
        cur_time = datetime.now()
        subject = f'{result.action} {symbol} {cur_time.month}/{cur_time.day}'

        # File paths and HTML content
        png_file_path = f'{self.directory}/{symbol}_technical_chart.png'  # Replace with your PNG path

        # Send the email
        gmail.send_email(
            sender=sender_email,
            to=recipient_email,
            subject=subject,
            png_file_path=png_file_path,
            html_string=result.recommendation
        )

    async def gemini_advise(self, symbols: List[str]) -> Dict[str, AnalyzedResult]:
        """Main function"""
        try:
            # Initialize advisor
            advisor = GeminiStockAdvisor(self.directory)

            print("=" * 60)
            print("GEMINI STOCK ANALYSIS ADVISOR")
            print("=" * 60)
            print(f"Data directory: {advisor.data_dir}")

            results = await advisor.analyze_multiple_stocks(symbols)

            if not results:
                print("No analysis results generated.")
            else:
                print(f"\nCompleted analysis for {len(results)} stocks.")
            return results

        except Exception as e:
            print(f"Error: {str(e)}")
            raise e

    async def deepseek_advise(self, symbols: List[str]) -> Dict[str, AnalyzedResult]:
        """Main function"""
        try:
            # Initialize advisor
            deepseek_api_key = os.environ['DEEPSEEK_API_KEY']
            advisor = DeepseekStockAdvisor(deepseek_api_key, self.directory)

            print("=" * 60)
            print("GEMINI STOCK ANALYSIS ADVISOR")
            print("=" * 60)
            print(f"Data directory: {advisor.data_dir}")

            results = await advisor.analyze_multiple_stocks(symbols)

            if not results:
                print("No analysis results generated.")
            else:
                print(f"\nCompleted analysis for {len(results)} stocks.")
            return results

        except Exception as e:
            print(f"Error: {str(e)}")
            raise e

    def run_analysis(self, yahoo: YahooFinance, analyzer: TechnicalAnalyzer, symbols):
        """Run the complete technical analysis"""
        print(f"Starting Technical Analysis for {len(symbols)} symbols")
        print(f"Period: {yahoo.period}")
        print(f"Output directory: {yahoo.data_dir}")
        print("-" * 50)

        # Download data
        stock_data = yahoo.download_stock_data(symbols)

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
                processed_data = analyzer.add_technical_indicators(data, symbol)

                # Create technical chart
                chart_path = analyzer.create_technical_chart(processed_data, symbol)
                chart_files.append(chart_path)

                # Save to CSV
                filepath = yahoo.save_to_csv(processed_data, symbol)
                processed_files.append(filepath)

                # Generate summary
                summary = yahoo.generate_summary_stats(processed_data, symbol)
                summary_data.append(summary)

            except Exception as e:
                print(f"✗ Error processing {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()

        # Save summary file
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(yahoo.data_dir, f"stock_summary_{datetime.now().strftime('%Y%m%d')}.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"\n✓ Summary saved to {summary_file}")

        print(f"\nAnalysis complete! Processed {len(processed_files)} stocks.")
        print(f"Generated {len(chart_files)} charts.")
        print(f"Files saved to: {yahoo.data_dir}")

        return processed_files, chart_files, summary_data

    async def yahoo_fetch(self, symbols: list[str]):

        # Create analyzer
        yahoo = YahooFinance(period=self.period, interval=self.interval, data_dir=self.directory)
        analyzer = TechnicalAnalyzer(self.directory)

        # Run analysis
        try:
            files, charts, summaries = self.run_analysis(yahoo, analyzer, symbols)

            if summaries:
                print("\n" + "=" * 60)
                print("QUICK SUMMARY (Latest Values)")
                print("=" * 60)
                for summary in summaries:
                    print(f"{summary['Symbol']}: Close=${summary['Close']:.2f}, "
                          f"RSI={summary['RSI']:.1f}, Volume={summary['Volume']:,}")

            if charts:
                print("\n" + "=" * 60)
                print("CHARTS GENERATED")
                print("=" * 60)
                for chart in charts:
                    print(f"✓ {os.path.basename(chart)}")

        except Exception as e:
            print(f"Error running analysis: {str(e)}")
            raise e
