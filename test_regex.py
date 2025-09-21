import re


def get_recommendation_value(html_string):
    """
    Most robust extraction that handles various formats:
    - With or without inner span
    - Different class combinations
    - Case insensitive matching
    """

    # Step 1: Find div with recommendation class
    div_pattern = r'<div[^>]*class="[^"]*recommendation[^"]*"[^>]*>(.*?)</div>'
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


def regex_test():
    file_dir = '/tmp/frank_ta/'
    filenames = ['ARKK_gemini_recommendations.html', 'ARKQ_gemini_recommendations.html',
                 'BABA_gemini_recommendations.html',
                 'CRWD_gemini_recommendations.html', 'U_gemini_recommendations.html', 'XYZ_gemini_recommendations.html',
                 'DOCS_gemini_recommendations.html', 'EMQQ_gemini_recommendations.html',
                 'LIT_gemini_recommendations.html',
                 'NIO_gemini_recommendations.html', 'PGJ_gemini_recommendations.html',
                 'QQQ_gemini_recommendations.html',
                 'TSLA_gemini_recommendations.html', 'U_gemini_recommendations.html', 'XYZ_gemini_recommendations.html']
    for filename in filenames:
        with open(file_dir + filename, 'r') as f:
            html = f.read()
            print(f'{get_recommendation_value(html)} from {filename}')

# regex_test()
