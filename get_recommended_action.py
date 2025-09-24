import glob
import os
import re
from pathlib import Path


def get_recommendation_action(html_string):
    """
    Most robust extraction that handles various formats:
    - With or without inner span
    - Different class combinations
    - Case insensitive matching
    """
    if 'recommendation buy' in html_string:
        return 'BUY'
    elif 'recommendation hold' in html_string:
        return 'HOLD'
    elif 'recommendation sell' in html_string:
        return 'SELL'

    pattern = r'RECOMMENDATION: (.*?)>(HOLD|BUY|SELL)<'
    match = re.match(pattern, html_string)
    if match:
        return match[2]

    # Step 1: Find div with recommendation class
    div_pattern = r'<div[^>]*class="[^"]*recommendation[^"]*"[^>]*>(.*?)(</div>|$|</h3>)'
    div_match = re.search(div_pattern, html_string, re.DOTALL | re.IGNORECASE)

    if not div_match:
        return None

    # Step 2: Get the content inside the div
    div_content = div_match.group(1)

    # Step 3: Strip HTML tags to get clean text
    clean_text = re.sub(r'<[^>]+>', ' ', div_content)

    # Step 4: Look for our target values
    value_pattern = r'\b(HOLD|BUY|SELL)\b'
    value_match = re.search(value_pattern, clean_text, re.IGNORECASE)

    if value_match:
        return value_match.group(1).upper()

    return 'Unknown'


def print_summary(file_dir: Path):
    pattern = os.path.join(file_dir, "*_gemini_recommendations.html")
    matching_files = glob.glob(pattern)

    with open(file_dir / 'summary.txt', 'w', encoding='utf-8') as summary:
        for filename in matching_files:
            with open(filename, 'r', encoding='utf-8') as f:
                html = f.read()
                content = f'{get_recommendation_action(html)} from {filename}'
                print(content)
                summary.write(content)
                summary.write("\n")
