import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cells content
cells = []

# Title and Intro
cells.append(nbf.v4.new_markdown_cell("""
# ⏳ Prévision Temporelle des Coûts (Time Series)

## Objectif
Ce notebook a pour but d'entraîner un modèle spécifique capable de prédire les coûts de traitement des déchets pour une **date future donnée**.

Contrairement au modèle classique qui se base sur les caractéristiques du déchet (type, poids, etc.), ce modèle se concentre sur l'**historique temporel** et les **tendances** par hôpital.

### Étapes :
1.  Chargement des données.
2.  **Feature Engineering Temporel** : Création de variables explicatives à partir de la date (année, mois, jour, jour de la semaine...).
3.  **Encodage** : Transformation des noms d'hôpitaux en valeurs numériques.
4.  **Entraînement** : Utilisation d'un RandomForestRegressor.
5.  **Sauvegarde** : Export du modèle pour l'application Web.
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Configuration pour l'affichage
sns.set(style="whitegrid")
%matplotlib inline

# Définition des chemins
BASE_DIR = "../../web_app"
DATA_PATH = "../../notebooks/data/dechets_hospitaliers.csv"
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Création du dossier models s'il n'existe pas
os.makedirs(MODELS_DIR, exist_ok=True)
"""))

# Load Data
cells.append(nbf.v4.new_markdown_cell("## 1. Chargement des Données"))
cells.append(nbf.v4.new_code_cell("""
print("Chargement du dataset...")
df = pd.read_csv(DATA_PATH)

# Conversion de la colonne date
df['date_collecte'] = pd.to_datetime(df['date_collecte'])

display(df.head())
df.info()
"""))

# Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""
## 2. Feature Engineering Temporel

Pour qu'un modèle puisse comprendre la notion de "temps" et de "saisonnalité", nous devons décomposer la date en plusieurs caractéristiques numériques :
*   **Année** : Pour capter la tendance globale (inflation, augmentation du volume...).
*   **Mois** : Pour la saisonnalité annuelle.
*   **Jour du mois** : Pour les cycles mensuels.
*   **Jour de la semaine** : Pour les cycles hebdomadaires (ex: moins de collectes le week-end).
*   **Jour de l'année** : Pour une granularité fine.
"""))
cells.append(nbf.v4.new_code_cell("""
df['annee'] = df['date_collecte'].dt.year
df['mois'] = df['date_collecte'].dt.month
df['jour'] = df['date_collecte'].dt.day
df['jour_semaine'] = df['date_collecte'].dt.dayofweek
df['jour_annee'] = df['date_collecte'].dt.dayofyear

print("Aperçu des nouvelles features temporelles :")
display(df[['date_collecte', 'annee', 'mois', 'jour', 'jour_semaine']].head())
"""))

# Encoding
cells.append(nbf.v4.new_markdown_cell("""
## 3. Encodage des Hôpitaux

Le modèle a besoin de savoir pour quel hôpital il prédit. Comme les algorithmes ne comprennent que les nombres, nous utilisons un **LabelEncoder** pour transformer chaque nom d'hôpital en un nombre unique entier.

Il est CRUCIAL de sauvegarder cet encodeur pour que l'application Web puisse transformer les choix de l'utilisateur de la même manière.
"""))
cells.append(nbf.v4.new_code_cell("""
le_hopital = LabelEncoder()
df['hopital_encoded'] = le_hopital.fit_transform(df['hopital'])

# Vérification
print(f"Nombre d'hôpitaux uniques : {len(le_hopital.classes_)}")
display(df[['hopital', 'hopital_encoded']].head())
"""))

# Visualisation
cells.append(nbf.v4.new_markdown_cell("## 4. Visualisation de l'Évolution Temporelle"))
cells.append(nbf.v4.new_code_cell("""
# Agrégation par mois pour voir la tendance globale
df['mois_annee'] = df['date_collecte'].dt.to_period('M')
monthly_cost = df.groupby('mois_annee')['cout_traitement'].sum()

plt.figure(figsize=(12, 6))
monthly_cost.plot(kind='line', marker='o', color='teal')
plt.title('Évolution du Coût Total Mensuel')
plt.ylabel('Coût ($)')
plt.xlabel('Mois')
plt.grid(True)
plt.show()
"""))

# Training
cells.append(nbf.v4.new_markdown_cell("""
## 5. Entraînement du Modèle

Nous utilisons un **RandomForestRegressor**. C'est un modèle robuste capable de capturer des relations non-linéaires complexes entre la date, l'hôpital et le coût.
"""))
cells.append(nbf.v4.new_code_cell("""
# Définition des features (X) et de la cible (y)
features_cols = ['hopital_encoded', 'annee', 'mois', 'jour', 'jour_semaine', 'jour_annee']
X = df[features_cols]
y = df['cout_traitement']

print("Entraînement du modèle en cours...")
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X, y)

print("✅ Modèle entraîné avec succès.")

# Évaluation rapide (Score R2 sur le train set - pour validation simple)
score = model.score(X, y)
print(f"Score R² (sur données d'entraînement) : {score:.4f}")
"""))

# Saving
cells.append(nbf.v4.new_markdown_cell("""
## 6. Sauvegarde des Artefacts

Nous sauvegardons :
1.  Le modèle entraîné (`model_future_cout.pkl`).
2.  L'encodeur des hôpitaux (`encoder_hopital_future.pkl`).

Ces fichiers seront chargés par l'application Flask via le fichier `future_routes.py`.
"""))
cells.append(nbf.v4.new_code_cell("""
model_path = os.path.join(MODELS_DIR, 'model_future_cout.pkl')
encoder_path = os.path.join(MODELS_DIR, 'encoder_hopital_future.pkl')

joblib.dump(model, model_path)
joblib.dump(le_hopital, encoder_path)

print(f"Modèle sauvegardé sous : {model_path}")
print(f"Encodeur sauvegardé sous : {encoder_path}")
"""))

nb['cells'] = cells

with open('/home/axel-renaud/Images/PROJET_INF_365/notebooks/NoteBook/Time_Series_Prediction.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
