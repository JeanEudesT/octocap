from chromadb import Client, Settings


class EmbeddingRepository:

    _collection: any

    def __init__(self):
        chromadb_client = Client(Settings(persist_directory="./chroma_db", is_persistent=True))
        self._collection = chromadb_client.get_or_create_collection(name="documents")

    def save(self, embeddings_to_save):
        for idx, embedding_to_save in enumerate(embeddings_to_save):
            self._collection.add(
                ids=[str(idx)],
                embeddings=[embedding_to_save["embedding"]],
                documents=[embedding_to_save["chunk"]]
            )

    def query(self, embedded_query):
        return self._collection.query(
            query_embeddings=[embedded_query],
            n_results=3
        )
