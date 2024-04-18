import datetime


def enhance_query(question):
    # Define keywords that trigger adding today's date to the query
    date_keywords = ["today", "now", "event", "news"]
    new_ques = question
    # Check if any of the date_keywords are in the question
    if any(keyword in question.lower() for keyword in date_keywords):
        # Get today's date in ISO format
        today_date = datetime.datetime.now().date().isoformat()
        # Append today's date to the question
        new_ques += " " + today_date

    # Check if the term 'cs' is in the question and append 'computer science'
    if "cs" in question.lower():
        new_ques += " computer science"

    return new_ques
