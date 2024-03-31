from dotenv import load_dotenv
import os
from history_storage import Message, History

import weaviate
from openai import OpenAI


def context_retrieval(client, collection_name, embedding):
    response = client.query.aggregate(collection_name).with_near_vector({
        'vector': embedding,
        'certainty': 0.7
    }).with_limit(5).with_meta().do()
    
    retrieved_texts = [result['path']['text_content'] for result in response['data']['Aggregate'][collection_name]]
    return retrieved_texts

def rag(history, question, openai_client, weaviate_client, collection_name):
    # generate question embedding
    question_embedding = openai_client.embeddings.create(
        input = [question.replace("\n", " ")],
        model = 'text-embedding-3-small'
    ).data[0].embedding

    # retrieved_texts = context_retrieval(weaviate_client, collection_name, question_embedding)
    retrieved_texts = ""
    combined_context = " ".join(retrieved_texts)
    content = f"Given the following information {combined_context}\n\nAnswer the question: {question}"
    
    # generate answer with context
    response = openai_client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages= history.get_history() + [{"role": "user", "content": content}]
    )

    history.add_message("user", content)
    history.add_message("assistant", response.choices[0].message.content)
    
    return response.choices[0].message.content

if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
    weaviate_endpoint = os.getenv("WEAVIATE_WCS_URL")

    # connect to a WCS instance & openai
    # weaviate_client = weaviate.connect_to_wcs(
    #     cluster_url = weaviate_endpoint,
    #     auth_credentials = weaviate.auth.AuthApiKey(weaviate_api_key),
    # )
    weaviate_client = None
    openai_client = OpenAI(api_key = openai_api_key)
    
    history = History()
    history.add_message("system", "You are a helpful assistant to answer any question related to Brown University's Computer Science department.")

    while True:
        question = input()
        if question.lower() == 'quit' or question.lower() == 'q':
            break
        answer = rag(history, question, openai_client, weaviate_client, "CSWebsiteContent")
        print(answer)