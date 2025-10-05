import asyncio
from pathlib import Path

from Orchestrator import Orchestrator
from config import get_weekly_symbols

if __name__ == "__main__":
    # You can customize these parameters
    period = '6mo'  # Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    # working_dir = Path('/tmp/frank_ta')

    # orchestrator = Orchestrator(period, '1d', working_dir)
    # asyncio.run(orchestrator.run(get_symbols()))

    working_dir = Path('/tmp/frank_ta2')
    orchestrator = Orchestrator('2y', '1wk', working_dir)
    asyncio.run(orchestrator.run(get_weekly_symbols()))
