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


def build_prompt(query: str, context: list) -> str:
    return f"""
        Tu es un assistant intelligent spécialisé dans l'analyse et la synthèse d'informations à partir de documents.
        Tu dois fournir des réponses précises et détaillées basées uniquement sur les documents fournis, en évitant toute invention.
        Fais moi une réponse courte\nQuestion de l'utilisateur :\n{query}\nRéponse :\nContexte :\n{'-'.join(context)}\n\nQuestion : {query}\nRéponse :"""

discussion = []

def generate_response(prompt: str):
    discussion.append({'role': 'user', 'content': prompt})
    full_response = ""
    stream = ollama.chat(
        model='llama3.2',
        messages=discussion,
        stream=True,
    )

    for chunk in stream:
        full_response += chunk['message']['content']
        print(chunk['message']['content'], end='', flush=True)

    return full_response

def chat():
    query = input("\nPrompt : ")
    generate_response(query)

embeddingRepository = EmbeddingRepository()

def main():
    while True:
        option = input("\nChoose your option: \n1: prompt \n2: prepare the RAGzmoket \n3: challenge the answer \nWrite exit to quit\n")
        if option.lower() == "exit":
            print("Bye")
            break

        if option.lower() == "1":
            prompt()

        if option.lower() == "2":
            prepare()

        if option.lower() == "3":
            chat()




def add_line_to_readme(query, response, context, flag):
    with open('README.md', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.strip() == "|---------------------------------------------------------------|:----------------------:|:---------------:|:---------------:|":
            index_end = i + 1

    new_line = f"| {query} | {response} | {context} | {flag} toto |\n"

    lines.insert(index_end, new_line)

    with open('README.md', 'w', encoding='utf-8') as file:
        file.writelines(lines)

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
        prompt = build_prompt(query, context)
        response = generate_response(prompt)
        add_to_readme = input("\n Do you want to add a new line in the readme ? (y/n)")

        if add_to_readme == 'y':
            flag = input("\n OK ou KO ? ")
            if flag == "ok":
                flag = '<div style="background-color: green; width: 50px; text-align: center;  margin:0">OK</div>'
            elif flag == "ko" :
                flag = '<div style="background-color: red; width: 50px; text-align: center;  margin:0">KO</div>'

            add_line_to_readme(query, response, prompt, flag)


if __name__ == '__main__':
    main()


