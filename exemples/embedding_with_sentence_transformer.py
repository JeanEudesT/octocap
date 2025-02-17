'''from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])
'''

def ajouter_ligne_au_readme(nouvelle_question, nouvelle_reponse, nouveau_contexte, nouveau_flag):
    # Lire le contenu actuel du fichier README.md
    with open('README.md', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Trouver l'endroit où se termine le tableau (dernière ligne du tableau)
    for i, line in enumerate(lines):
        if line.strip() == "|---------------------------------------------------------------|:----------------------:|:---------------:|:---------------:|":
            index_tableau_fin = i + 1  # C'est la ligne après l'en-tête du tableau

    # Créer la nouvelle ligne à ajouter
    nouvelle_ligne = f"| {nouvelle_question} | {nouvelle_reponse} | {nouveau_contexte} | {nouveau_flag} toto |\n"

    # Ajouter la nouvelle ligne dans le tableau
    lines.insert(index_tableau_fin, nouvelle_ligne)

    # Sauvegarder les modifications dans le fichier README.md
    with open('README.md', 'w', encoding='utf-8') as file:
        file.writelines(lines)


ajouter_ligne_au_readme(
    "Quel est le langaazeaeazeezaeazezazeazeaeazeazeaeazeazeazeazeazeazeazeazeazaeazeaze  ge utilisé pour le développement du programme ?Quel est le langage utilisé pour le développement du programme ?Quel est le langage utilisé pour le développement du programme ?",
    "Python",
    "Développement principal",
    "Optimisation"
)
