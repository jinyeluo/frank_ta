import asyncio
import glob
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from DeepseekStockAdvisor import DeepseekStockAdvisor
from GeminiStockAdvisor import GeminiStockAdvisor
from GmailSender import GmailSender
from config import get_symbols, get_llm_vendor, GEMINI, DEEPSEEK
from get_recommended_action import print_summary
from llm_base import AnalyzedResult
from yahoo import TechnicalAnalyzer

hidden_frank_ta = Path.home() / '.frank_ta'


def delete_all_files(directory):
    pattern = os.path.join(directory, "*")
    files = glob.glob(pattern)

    for file_path in files:
        if os.path.isfile(file_path):  # Only delete files, not directories
            os.remove(file_path)
            print(f"Deleted: {file_path}")


async def main(symbols: List[str], data_dir:Path):
    delete_all_files(data_dir)

    load_dotenv(hidden_frank_ta / '.env')
    await yahoo_fetch(symbols, data_dir)

    if get_llm_vendor() == GEMINI:
        results = await gemini_advise(symbols, data_dir)
    elif get_llm_vendor() == DEEPSEEK:
        await deepseek_advise(symbols, data_dir)

    print_summary(data_dir)

    if False:
        for k, v in results.items():  # ignore
            await send_emails(k, v, data_dir)  # ignore

async def send_emails(symbol, result:AnalyzedResult, data_dir):
    # Initialize Gmail sender
    secret_file = hidden_frank_ta / 'google_client_secret.json'
    token_file =  hidden_frank_ta /  'google_client_token.json'
    gmail = GmailSender(str(secret_file), str(token_file))

    # Email details
    sender_email = "none@gmail.com"  # Replace with your Gmail
    recipient_email = "none@gmail.com"  # Replace with recipient
    cur_time = datetime.now()
    subject = f'{result.action} {symbol} {cur_time.month}/{cur_time.day}'


    # File paths and HTML content
    png_file_path = f'{data_dir}/{symbol}_technical_chart.png'  # Replace with your PNG path


    # Send the email
    gmail.send_email(
        sender=sender_email,
        to=recipient_email,
        subject=subject,
        png_file_path=png_file_path,
        html_string=result.recommendation
    )

async def gemini_advise(symbols: List[str], working_dir)-> Dict[str, AnalyzedResult]:
    """Main function"""
    try:
        # Initialize advisor
        advisor = GeminiStockAdvisor(working_dir)

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


async def deepseek_advise(symbols: List[str], working_dir) -> Dict[str, AnalyzedResult]:
    """Main function"""
    try:
        # Initialize advisor
        deepseek_api_key = os.environ['DEEPSEEK_API_KEY']
        advisor = DeepseekStockAdvisor(deepseek_api_key, working_dir)

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

async def yahoo_fetch(symbols: list[str],  working_dir:Path):
    # You can customize these parameters
    period = '6mo'  # Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'

    # Create analyzer
    analyzer = TechnicalAnalyzer(period=period, data_dir=working_dir)

    # Run analysis
    try:
        files, charts, summaries = analyzer.run_analysis(symbols)

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


if __name__ == "__main__":
    working_dir = Path('/tmp/frank_ta')

    # stock_symbols = ['CRWD']
    asyncio.run(main(get_symbols(), working_dir))
