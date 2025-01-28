import os
from openai import OpenAI
from chromadb import Client, Settings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Configuration OpenAI
api_key = os.getenv("OPENAI_API_KEY")

clientOpenAI = OpenAI(api_key=api_key)
print("GO")

print("setup model")
# --------------------------------------------
# Étape 1 : Vectorisation des documents
# --------------------------------------------
modelPourEmbedding = SentenceTransformer('all-MiniLM-L6-v2')
print("done")
# --------------------------------------------
# Étape 2 : Configuration de ChromaDB
# --------------------------------------------

print("setup db")
dbClient = Client(Settings(persist_directory="./chroma_db", is_persistent=True))
collection = dbClient.get_or_create_collection(name="documents")
print("done")

def vectoriser_texte(texte: str) -> list:
    """Vectorise un texte avec SentenceTransformers."""
    return modelPourEmbedding.encode(texte).tolist()

'''
# Charger les documents
documents = []
for filename in os.listdir("documents"):
    if filename.endswith(".txt"):
        with open(f"documents/{filename}", "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(content)


print("Debut de la vectorisation")
# Vectoriser et stocker dans ChromaDB
for idx, doc in enumerate(documents):
    print(f'{idx}/{len(documents)}')
    embedding = vectoriser_texte(doc)
    collection.add(
        ids=[str(idx)],
        embeddings=[embedding],
        documents=[doc]
    )
print("Fin de la vectorisation")'''


# --------------------------------------------
# Étape 3 : Interface CLI avec recherche RAG
# --------------------------------------------

def rechercher_documents(question: str, k=10) -> list:
    """Recherche les documents pertinents avec ChromaDB."""
    embedding = vectoriser_texte(question)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=k
    )
    print("results", results)
    return results['documents'][0]


def generer_reponse(question: str, contexte: list) -> str:
    """Génère une réponse avec OpenAI."""
    prompt = f"Contexte :\n{'-'.join(contexte)}\n\nQuestion : {question}\nRéponse :"

    print('-'.join(contexte))
'''    response = clientOpenAI.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "Vous êtes un assistant qui répond aux questions basées sur les documents fournis."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content'''




# --------------------------------------------
# Boucle interactive
# --------------------------------------------
print("Assistant RAG - Tapez 'exit' pour quitter\n")

question = ""

while question.lower() != "exit":
    question = input("\nQuestion : ")

    # Recherche des documents
    contexte = rechercher_documents(question)

    # Génération de la réponse
    reponse = generer_reponse(question, contexte)
    print(f"\nRéponse : {reponse}\n")
