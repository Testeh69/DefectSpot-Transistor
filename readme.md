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

## 🚀 Utilisation

### 1️⃣ Lancer le projet localement (sans Docker)

**Démarrer l’API FastAPI :**

```bash
python -m uvicorn api.main:app --reload
```

L’API sera disponible sur : http://127.0.0.1:8000

Endpoint principal pour l’inférence sur une image de transistor : /file

Démarrer l’interface Streamlit :

```
streamlit run streamlit_app.py
```

L’interface sera disponible sur : http://localhost:8501

2️⃣ Lancer le projet avec Docker Compose
Le projet contient deux services :

FastAPI → backend pour l’inférence

Streamlit → interface utilisateur

Pour construire les images et démarrer les conteneurs :

```
docker-compose up --build
```

FastAPI sera accessible à : http://localhost:8000

Streamlit sera accessible à : http://localhost:8501

3️⃣ Utilisation de l’interface Streamlit
Ouvrez votre navigateur à http://localhost:8501

Téléversez une image de transistor (.png, .jpg, .jpeg)

Cliquez sur “Lancer l’inférence”

Visualisez les résultats :

Masque de défaut binaire → zones noires/blanches indiquant les anomalies

Overlay rouge → superposition des défauts sur l’image originale

⚠️ Le masque binaire utilise un seuil automatique (0.3 par défaut). Vous pouvez l’ajuster dans le code Streamlit si nécessaire.

4️⃣ Utilisation de FastAPI directement
Vous pouvez appeler FastAPI depuis Postman, curl, ou un script Python :


```python
import requests

API_URL = "http://localhost:8000/file"

with open("image_transistor.png", "rb") as f:
    response = requests.post(API_URL, files={"file": f})

if response.status_code == 200:
    predicted_mask = response.json()["result"]
    print("Masque reçu:", predicted_mask)
else:
    print(f"Erreur API : {response.status_code}")
```

Endpoint attendu : POST http://localhost:8000/file

Form-data : file=<votre_image.png>

5️⃣ Arrêter les conteneurs Docker

```
docker-compose down
```

Cela arrête et supprime les conteneurs, mais ne supprime pas les images.
Pour reconstruire avec un nouveau modèle ou une nouvelle version :

```
docker-compose up --build
```

## **📌 Conseils**

Toujours utiliser le même pipeline de transformation (Resize, ToTensor) pour l’inférence et l’entraînement.

Les anomalies sont détectées par la différence absolue entre l’image reconstruite et l’image originale.

Ajustez le seuil pour le masque binaire selon les besoins de détection.