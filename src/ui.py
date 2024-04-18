from openai_pipeline import rag
from history_storage import History
from openai import OpenAI
from db_client import QdrantDatabaseClient
from dotenv import load_dotenv
import os
import streamlit as st

st.title("Brown CS Assistant")

if "history" not in st.session_state:
    st.session_state.history = History()
    st.session_state.history.add_message("system", "I am an AI assistant to answer any question related to Brown University's Computer Science department.")

if "openai_client" not in st.session_state:
    load_dotenv()
    st.session_state.openai_client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

if "db_client" not in st.session_state:
    load_dotenv()
    st.session_state.db_client = QdrantDatabaseClient(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY"))

if "query" not in st.session_state:
    query = {}
    query['collection_name'] = 'CSWebsiteContent'
    query['property'] = ["text_content", "url"]
    query['certainty'] = 0.6
    query['limit'] = 3
    st.session_state.query = query

for message in st.session_state.history.get_history():
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about Brown CS"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response, links = rag(st.session_state.history, prompt, st.session_state.openai_client, st.session_state.db_client, st.session_state.query, stream=True)
        full_response = st.write_stream(response)
        st.session_state.history.add_message("assistant", full_response)
        st.markdown("> Reference Links:  \n" + "  \n".join(links))