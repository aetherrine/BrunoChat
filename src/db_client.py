import abc
import weaviate
from qdrant_client import QdrantClient
from datetime import datetime, timezone

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

    def sort_key(self, item):
        timestamp = item[0]
        if timestamp is None:
            return (datetime.min.replace(tzinfo=timezone.utc),)
        return (datetime.fromisoformat(timestamp.replace('Z', '+00:00')),)

    def query(self, query):
        return self.client.search(
            collection_name = query['collection_name'],
            query_vector = query['question_embedding'],
            with_payload = True,
            limit = query['limit'] * 3,
        )
    
    def parse(self, response, query):
        retrieved_texts = [result.payload['text_content'] for result in response]
        links = [result.payload['url'] for result in response]
        timestamps = [result.payload.get('last_modified') for result in response]
        
        zipped_lists = zip(timestamps, retrieved_texts, links)
        sorted_zipped_lists = sorted(zipped_lists, key=self.sort_key, reverse=True)
        timestamps, retrieved_texts, links = zip(*sorted_zipped_lists)

        return retrieved_texts[:query['limit']], links[:query['limit']]