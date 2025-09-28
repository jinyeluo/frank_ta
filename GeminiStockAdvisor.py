#!/usr/bin/env python3
"""
Gemini Stock Analysis Advisor
Sends technical analysis data to Google Gemini for buy/sell/hold recommendations
"""
import warnings
from pathlib import Path

from google.genai import Client
from google.genai.types import GenerateContentConfig

from llm_base import LLMBase

warnings.filterwarnings('ignore')

GEMINI_FLASH = "gemini-2.5-flash"
GEMINI_PRO = "gemini-2.5-pro"


class GeminiStockAdvisor(LLMBase):
    def __init__(self, working_dir: Path):

        super().__init__(working_dir, 'gemini')

    async def llm_chat(self, prompt: str) -> str:
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