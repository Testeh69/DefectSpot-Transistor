# DefectSpot-Transistor

**Détection de défauts sur transistors par apprentissage non supervisé**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)

---

## **📌 Description**

Ce dépôt contient des **notebooks Jupyter** et des notes techniques pour la détection de défauts sur transistors en utilisant des méthodes d’**apprentissage non supervisé** ( AutoEncodeur.). L’objectif est de développer un modèle capable d’identifier des anomalies dans les données de test des transistors, sans nécessiter de données étiquetées.

**Applications potentielles** :

- Contrôle qualité en production industrielle
- Maintenance prédictive pour systèmes embarqués
- Optimisation des processus de fabrication en robotique

## **📦 Structure du projet**

```
DefectSpot-Transistor/
│
├─ api/ # FastAPI backend pour l'inférence
│ └─ main.py
├─ notebooks/ # Jupyter notebooks pour expérimentation
├─ models/ # Modèles PyTorch entraînés (.pth) et fichiers.py
├─ ui/streamlit_app.py # Interface utilisateur Streamlit
├─ requirements.txt # Dépendances Python
├─ paper/ #papier scientifique 
├─ research #Note
└─ README.md

```

## **⚙️ Installation**

```bash
git clone https://github.com/votre-utilisateur/DefectSpot-Transistor.git
cd DefectSpot-Transistor
Créer un environnement virtuel :

python -m venv venv
Activer l’environnement :

Windows :

.\venv\Scripts\activate
Linux / macOS :

source venv/bin/activate
Installer les dépendances :

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## **🚀 Utilisation**

```
1. Lancer l’API FastAPI
python -m uvicorn api.main:app --reload
L’API sera disponible sur http://127.0.0.1:8000

Endpoint principal : /file pour l’inférence sur une image de transistor

2. Lancer l’interface Streamlit
streamlit run streamlit_app.py
```


## **📌 Conseils**

Toujours utiliser le même pipeline de transformation (Resize, ToTensor) pour l’inférence et l’entraînement.

Les anomalies sont détectées par la différence absolue entre l’image reconstruite et l’image originale.

Ajustez le seuil pour le masque binaire selon les besoins de détection.