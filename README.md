# Fast reverse image search engine

---
## English 🇬🇧

### Context

- While the Internet made it easier to share information, image theft has become a real plague.
    - Example: real estate market → malicious people can easily reuse your photos
    - Sometimes not just copy → slight changes like watermark
- Can be extended to tackle another problem: counterfeiting → big loss of revenues for brands (especially luxury)

## Solution

We decided to go with a 2-step solution involving fine-tuning ResNet-50 on the BAM! dataset and using a random vector LSH strategy for search.

## Results

- 88% accuracy on the validation split of BAM! for the classification task
- The approximate LSH-based image retrieval takes a few seconds against about an hour using the naive exact search.


---
## French 🇫🇷

- While the Internet made it easier to share information, image theft has become a real plague.
    - Example: real estate market → malicious people can easily reuse your photos
    - Sometimes not just copy → slight changes like watermark
- Can be extended to tackle another problem: counterfeiting → big loss of revenues for brands (especially luxury)

Nous avons implémenté un moteur de recherche d'images inversée, c'est-à-dire que pour une image donnée, l'algorithme renvoie les images les plus similaires qu'il possède dans la base de données. L'objectif a été de fournir le modèle le plus précis possible tout en imposant une grande performance en terme de vitesse.

Il est divisé en trois grandes parties :

- la gestion de la base de données,
- la formation au modèle
- un serveur web pour mettre le tout en place.


## Extrait du rapport inclus dans le repository

![extrait_rapport](ressources/extrait_rapport.png)



## Setup

Le dossier data gère la base de données et le fichier data.py contient les
principaux des fonctions pour travailler avec la base de données en sql.
 Le fichier `DataGen.py` sert à de créer un générateur de données keras pour
pouvoir gérer la base de données plus efficacement.

Dans le dossier feature_extractor, il contient les principaux fichiers pour la
entrainement création de notre modèle de feature extraction.
Dans le carnet Jupyter `Train_Classification_model.ipynb` détaille les étapes
de la création du modèle.

Dans le dossier web se trouve la page HTML créée.

Pour plus d'informations techniques sur les travaux, veuillez lire l'article
écrit dans le dossier 'article'.

Initialisez le serveur web : Le serveur web nommé `index.py` ne peut être que
initialisé si tous les prérequis qui se trouvent dans le fichier
`requirements.txt` sont satisfaits.

Si toutes les conditions sont remplies, l'initialisation du serveur web
il suffit d'exécuter le fichier setup.py pour que les fichiers nécessaires
sont telechargés (comme par exemple les poids des modèles) et il suffit ensuite
de utilisez la commande `python3 index.py` pour que le serveur se démarre
à l'adresse `localhost` à la porte `80`.
