# Octocap

## Description
Octocap est un RAG qui permet de faire de la recherche dans un langage naturel. Il se base sur la connaissance
d'une entreprise (Pages confluences par exemple)

### Equipe
- Zaki
- Jean-Eudes

## Prerequires
- Python 3.12
- Pip
- ollama

## Getting started

### Install and pull ollama model locally

1) First, go to the [ollama](https://ollama.com/) website and download ollama
2) Run the ollama command to pull the expected model  
```ollama pull llama3.2```

### Creating Virtual Environments
```python3.12 -m venv octocap-env```

### Activate the virtual environment
```source octocap-env/bin/activate```

### Install dependencies
```pip install requirements.txt```

### Rapport de performance
<div style="background-color: green; width: 50px; text-align: center;  margin:0">OK</div>
<div style="background-color: red; width: 50px; text-align: center;  margin:0">KO</div>

## ollama3.2
| Question                                                      |        Réponse         |    Contexte     |      Flag       |
|---------------------------------------------------------------|:----------------------:|:---------------:|:---------------:|
| Quel est le dev ops du projet Octo Cloud Optimizer ?          |           |  |             |
| Quels sont les outils utilisés pour la gestion de projet ?    |           |  |             |
| coucou qsdqd |sdsd

