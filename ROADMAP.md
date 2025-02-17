# Roadmap


## [Etape 1 - Avoir un RAG en local](https://www.notion.so/Etape-1-Avoir-un-RAG-en-local-18945fbd8832800894abca50d8ccbb9a?pvs=21)

### TODO:
- faire des chunks différemment : avec des séparateurs
- faire un comparatif (google sheets) avec des mêmes questions etc (nbr de mensonges) -> In progress
- log le contexte du rag pour voir si les pb viennent du rag ou du llm -> OK
- Pouvoir discuter avec le LLM pour challenger sur ses réponses -> In progress
- Essayer d'enrichir la Query (Passer à l'étape 5 Mouahhaha)

### Methodologie
- On a commencé avec un premier model pris sans trop réfléchir: all-MiniLM-L6-v2
  - On avait des mauvais résultats (les mots clés de la query n'étaient même pas présent dans les résultas remontés par chromadb)
  - On s'est rendu compte que  la langue était importante et qu'il fallait choisir un modèle en fonction de la langue qu'on souhaite vectoriser.
- On choisi un modèle qui est multilingue: models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2
  - On a commencé à avoir de meilleurs résultats (les mots clés de la query étaient présent dans les résultats remontés par chromadb)
  - On a l'impression qu'il faut plus du "word matching" que de la recherche en tenant compte de la sémantique.
  - On a essayé d'utiliser un autre modele(models--intfloat--multilingual-e5-large) pour invalider l'hypothèse que le modele utilisé est mauvais. Le résultat est le même (word matching)
  - On a essayé de changer la taille de nos chunks(On est passé de 1200 de base à 600) et de l'overlap(On est passé de 20 à 100)
- D'après notre veille sur le RAG, nous avons implémenté un RAG naïf qui est connu tendre vers du word matching.
- Nous allons implémenter un RAG Augmenté qui va enrichir la query afin d'avoir une meilleure compréhension de la query (capturer un meilleur contexte)


[Etape 2 - Brancher confluence au rag pour automatiser la vectorization des pages confluences](https://www.notion.so/Etape-2-Brancher-confluence-au-rag-pour-automatiser-la-vectorization-des-pages-confluences-18945fbd8832808390fbd40f01992ba1?pvs=21)

[Etape 3 - Deployer sur AWS](https://www.notion.so/Etape-3-Deployer-sur-AWS-18945fbd88328099a94efbd70da38a36?pvs=21)

## Bonus

[Etape 4 - Brancher avec MM](https://www.notion.so/Etape-4-Brancher-avec-MM-18945fbd8832807abcfdf9be59fed95e?pvs=21)

[Etape 5 - RAG Augmenté (jouer avec les paramètres du RAG)](https://www.notion.so/Etape-5-RAG-Augment-jouer-avec-les-param-tres-du-RAG-18945fbd88328028ba1efca75ff21e21?pvs=21)
