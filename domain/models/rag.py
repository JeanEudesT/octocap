import os

from sentence_transformers import SentenceTransformer

class Rag:
    _documents: [] = []
    _chunked_documents: [] = []
    _embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    _embeddings: [] = []

    def load_documents(self, path="documents", ext=".txt") -> list:
        for filename in os.listdir(path):
            if filename.endswith(ext):
                with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
                    content = f.read()
                    self._documents.append(content)
        return self._documents

    def create_chunks(self) -> list:
        def split_text(text, chunk_size=1500, chunk_overlap=20) -> list:
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start = end - chunk_overlap
            return chunks

        for doc in self._documents:
            for chunk in split_text(doc):
                self._chunked_documents.append(chunk)
        return self._chunked_documents

    def create_embeddings(self) -> list:
        for idx, chunk in enumerate(self._chunked_documents):
            embedding = self._embedding_model.encode(chunk).tolist()
            self._embeddings.append({
                "embedding": embedding,
                "chunk": chunk
            })
        return self._embeddings
