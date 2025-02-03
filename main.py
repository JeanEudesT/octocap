import os
from chromadb import Client, Settings
from sentence_transformers import SentenceTransformer
import ollama


def vectorize_text(model_embedding, text: str) -> list:
    return model_embedding.encode(text).tolist()


def split_text(text, chunk_size=1500, chunk_overlap=20) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


def load_documents(path="documents", ext=".txt") -> list:
    documents = []
    for filename in os.listdir(path):
        if filename.endswith(ext):
            with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
                content = f.read()
                documents.append(content)
    return documents


def create_chunks(documents) -> list:
    chunked_documents = []
    for doc in documents:
        for chunk in split_text(doc):
            chunked_documents.append(chunk)
    return chunked_documents


def persist_vectorized_documents(persistance, model_embedding, chunked_documents):
    for idx, chunk in enumerate(chunked_documents):
        embedding = vectorize_text(model_embedding, chunk)
        persistance.add(
            ids=[str(idx)],
            embeddings=[embedding],
            documents=[chunk]
        )


# --------------------------------------------
# Étape 3 : Interface CLI avec recherche RAG
# --------------------------------------------
def search_documents(persistance, embedded_query: str, k=3) -> list:
    results = persistance.query(
        query_embeddings=[embedded_query],
        n_results=k
    )
    # print("results", results,"\n")
    return results['documents'][0]


def build_prompt(query: str, context: list):
    return f"""
        Tu es un assistant intelligent spécialisé dans l'analyse et la synthèse d'informations à partir de documents.
        Tu dois fournir des réponses précises et détaillées basées uniquement sur les documents fournis, en évitant toute invention.
        Fais moi une réponse courte\nQuestion de l'utilisateur :\n{query}\nRéponse :\nContexte :\n{'-'.join(context)}\n\nQuestion : {query}\nRéponse :"""


def generate_response(prompt: str):
    stream = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)


model_embedding = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
chromadb_client = Client(Settings(persist_directory="./chroma_db", is_persistent=True))
# chromadb_client.delete_collection("documents")
collection = chromadb_client.get_or_create_collection(name="documents")

documents = load_documents("documents", ".txt")
chunked_documents = create_chunks(documents)
persist_vectorized_documents(collection, model_embedding, chunked_documents)


while True:
    query = input("\nPrompt : ")

    if query.lower() == "exit":
        break

    embedded_query = vectorize_text(model_embedding, query)
    context = search_documents(collection, embedded_query)
    generate_response(build_prompt(query, context))

    print("Fin")
