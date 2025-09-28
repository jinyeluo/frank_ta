#!/usr/bin/env python3
from pathlib import Path

from openai import OpenAI

"""
Gemini Stock Analysis Advisor
Sends technical analysis data to Google Gemini for buy/sell/hold recommendations
"""
import warnings

from llm_base import LLMBase

warnings.filterwarnings('ignore')


class DeepseekStockAdvisor(LLMBase):
    def __init__(self, api_key, working_dir: Path):
        super().__init__(working_dir, 'deepseek')
        self.api_key = api_key

    async def llm_chat(self, prompt: str) -> str:
        client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")

        response = client.chat.completions.create(
            model='deepseek-reasoner',
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )

        return response.choices[0].message.content
