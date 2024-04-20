from dotenv import load_dotenv
import os
from history_storage import History
from db_client import QdrantDatabaseClient
from openai import OpenAI
from utils import enhance_query


def context_retrieval(client, query):
    response = client.query(query)
    return client.parse(response, query)

def rag(history, question, openai_client, db_client, query, stream=False):
    # generate question embedding
    enhanced_question = enhance_query(question)
    question_embedding = openai_client.embeddings.create(
        input = [enhanced_question.replace("\n", " ")],
        model = 'text-embedding-3-small'
    ).data[0].embedding

    query['question_embedding'] = question_embedding

    retrieved_texts, links = context_retrieval(db_client, query)
    combined_context = " ".join(retrieved_texts)
    content = f"Given the following information {combined_context}\n\nAnswer the question: {enhanced_question}"
    
    # generate answer with context
    response = openai_client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages= history.get_history() + [{"role": "user", "content": content}],
        stream = stream,
    )

    history.add_message("user", question)

    if not stream:
        history.add_message("assistant", response.choices[0].message.content, links)
        return response.choices[0].message.content, links
    else:
        # defer recording assistant's history to upstream
        return response, links

if __name__ == "__main__":
    load_dotenv()
    db_url = os.getenv("QDRANT_URL")
    db_key = os.getenv("QDRANT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # connect to db & openai
    db_client = QdrantDatabaseClient(db_url, db_key)
    openai_client = OpenAI(api_key = openai_api_key)

    # prepare db query
    query = {}
    query['collection_name'] = 'CSWebsiteContent'
    query['property'] = ["text_content", "url"]
    query['certainty'] = 0.6
    query['limit'] = 5
    
    history = History()
    history.add_message("system", "You are a helpful assistant to answer any question related to Brown University's Computer Science department.")
    print("---------------------------------------")

    while True:
        question = input()
        if question.lower() == 'quit' or question.lower() == 'q':
            break
        answer, links = rag(history, question, openai_client, db_client, query)
        print(answer)
        print('* Reference link: ' + ', '.join(links))