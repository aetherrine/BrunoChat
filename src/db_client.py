import abc
import weaviate
from qdrant_client import QdrantClient

class DatabaseClient(abc.ABC):
    @abc.abstractmethod
    def query(self, query):
        pass

    @abc.abstractmethod
    def parse(self, response, query):
        pass

class WeaviateDatabaseClient(DatabaseClient):
    def __init__(self, url, key):
        self.client = weaviate.Client(url=url, auth_client_secret=weaviate.AuthApiKey(api_key=key))

    def query(self, query):
        return (self.client.query.get(query['collection_name'], query['property'])
                .with_near_vector({"vector": query['question_embedding'], "certainty": query['certainty']})
                .with_limit(query['limit'])
                .with_additional(["distance"])
                .do())
    
    def parse(self, response, query):
        retrieved_texts = [result['text_content'] for result in response['data']['Get'][query['collection_name']]]
        links = [result['url'] for result in response['data']['Get'][query['collection_name']]]
        return retrieved_texts, links

class QdrantDatabaseClient(DatabaseClient):
    def __init__(self, url, key):
        self.client = QdrantClient(url=url, api_key=key)

    def query(self, query):
        return self.client.search(
            collection_name = query['collection_name'],
            query_vector = query['question_embedding'],
            with_payload = True,
            limit = query['limit'],
        )
    
    def parse(self, response, query):
        retrieved_texts = [result.payload['text_content'] for result in response]
        links = [result.payload['url'] for result in response]
        return retrieved_texts, links