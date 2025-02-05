from sentence_transformers import SentenceTransformer
from domain.usecases.prepare_rag_from_path import PrepareRagFromPath
from infrastructure.embedding_repository import EmbeddingRepository
import ollama


def vectorize_text(model_embedding, text: str) -> list:
    return model_embedding.encode(text).tolist()


def search_documents(persistance, embedded_query: str) -> list:
    results = persistance.query(embedded_query)
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

embeddingRepository = EmbeddingRepository()

def main():
    while True:
        option = input("\nChoose your option (1: prompt, 2: prepare the RAGzmoket: ")
        if option.lower() == "exit":
            break

        if option.lower() == "1":
            prompt()

        if option.lower() == "2":
            prepare()


def prepare():
    prepareRagFromPathUsecase = PrepareRagFromPath(embeddingRepository)
    prepareRagFromPathUsecase.execute("documents", "txt")

def prompt():
    while True:
        query = input("\nPrompt : ")

        if query.lower() == "exit":
            break

        embedded_query = vectorize_text(SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'), query)
        context = search_documents(embeddingRepository, embedded_query)
        generate_response(build_prompt(query, context))


if __name__ == '__main__':
    main()


