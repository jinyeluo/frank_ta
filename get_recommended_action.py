import glob
import os
import re
from pathlib import Path


def get_recommendation_action_direct(html_string):
    html_string = html_string.lower()
    if 'recommendation buy' in html_string or 'recommendation: buy' in html_string:
        return 'BUY'
    elif 'recommendation hold' in html_string or 'recommendation: hold' in html_string:
        return 'HOLD'
    elif 'recommendation sell' in html_string or 'recommendation: sell' in html_string:
        return 'SELL'
    return None


def get_recommendation_action_direct2(html_string):
    pattern = r'RECOMMENDATION: (.*?)\>(HOLD|BUY|SELL)\<'
    match = re.match(pattern, html_string)
    if match:
        return match[2]
    return None


def get_recommendation_action_direct3(html_string):
    # Step 5: Look for our target values >SELL<
    value_pattern = r'\>(HOLD|BUY|SELL)\<'
    value_match = re.search(value_pattern, html_string, re.IGNORECASE)

    if value_match:
        return value_match.group(1).upper()

def get_recommendation_action(html_string):
    """
    Most robust extraction that handles various formats:
    - With or without inner span
    - Different class combinations
    - Case insensitive matching
    """
    action = get_recommendation_action_direct(html_string)
    if action:
        return action

    action = get_recommendation_action_direct2(html_string)
    if action:
        return action

    action = get_recommendation_action_direct3(html_string)
    if action:
        return action

    # Step 1: Find div with recommendation class
    div_pattern = r'<div[^>]*class="[^"]*recommendation[^"]*"[^>]*>(.*?)(</div>|$|</h3>)'
    div_match = re.search(div_pattern, html_string, re.DOTALL | re.IGNORECASE)

    if not div_match:
        return 'Unknown'

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
    pattern = os.path.join(file_dir, "*_recommendations.html")
    matching_files = glob.glob(pattern)

    with open(file_dir / 'summary.txt', 'w', encoding='utf-8') as summary:
        for filename in matching_files:
            pattern = r'.*\\([A-Za-z0-9-\.]+)_(([A-Za-z0-9]+))_recommendations.html'
            match = re.match(pattern, filename)
            if match:
                symbol = match[1]
                llm = match[2]
            else:
                symbol = 'unknown'
                llm = 'unknown'
            with open(filename, 'r', encoding='utf-8') as f:
                html = f.read()
                content = f'{get_recommendation_action(html)}\t{symbol}\t{llm}'
                print(content)
                summary.write(content)
                summary.write("\n")

# print_summary(Path('/tmp/frank_ta'))
