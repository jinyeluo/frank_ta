import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from GeminiStockAdvisor import GeminiStockAdvisor, AnalyzedResult
from GmailSender import GmailSender
from yahoo import TechnicalAnalyzer

hidden_frank_ta = Path.home() / '.frank_ta'


async def main(symbols: List[str], data_dir:Path):
    load_dotenv(hidden_frank_ta / '.env')

    await yahoo_fetch(symbols, data_dir)
    results = await gemini_advise(symbols, data_dir)
    for k, v in results.items():
        await send_emails(k, v, data_dir)

async def send_emails(symbol, result:AnalyzedResult, data_dir):
    # Initialize Gmail sender
    secret_file = hidden_frank_ta / 'google_client_secret.json'
    token_file =  hidden_frank_ta /  'google_client_token.json'
    gmail = GmailSender(str(secret_file), str(token_file))

    # Email details
    sender_email = "jinyeluo@gmail.com"  # Replace with your Gmail
    recipient_email = "jinyeluo@gmail.com"  # Replace with recipient
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

        # Save results if requested
        if results:
            advisor.save_recommendations(results)

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
    stock_symbols_account_1 = ['ARKK', 'CRWD', 'DOCS', 'EMQQ', 'LIT', 'PGJ', 'NIO', 'XYZ']
    stock_symbols_account_2 = ['ARKQ', 'BABA', 'TSLA', 'QQQ', 'U', 'OPFI']

    stock_symbols = stock_symbols_account_1
    stock_symbols.extend(stock_symbols_account_2)
    asyncio.run(main(stock_symbols, working_dir))

