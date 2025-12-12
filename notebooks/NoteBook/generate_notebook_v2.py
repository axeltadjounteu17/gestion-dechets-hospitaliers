import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cells content
cells = []

# Title and Intro
cells.append(nbf.v4.new_markdown_cell("""
# 🏥 Analyse et Prédiction des Déchets Hospitaliers

## Objectifs
- Analyser les données des déchets hospitaliers.
- Implémenter et comparer au moins 4 modèles de machine learning pour prédire le coût de traitement.
- Sauvegarder le meilleur modèle pour l'application web.
- Mettre en place un pipeline de prédiction complet.
"""))

# Imports
cells.append(nbf.v4.new_code_cell("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configuration
sns.set(style="whitegrid")
%matplotlib inline

# Création du dossier pour les modèles
MODELS_DIR = "../../web_app/models"
os.makedirs(MODELS_DIR, exist_ok=True)
"""))

# Load Data
cells.append(nbf.v4.new_markdown_cell("## 1. Chargement et Aperçu des Données"))
cells.append(nbf.v4.new_code_cell("""
DATA_PATH = "../data/dechets_hospitaliers.csv"
df = pd.read_csv(DATA_PATH)

print(f"Dimensions du dataset : {df.shape}")
display(df.head())
"""))

cells.append(nbf.v4.new_code_cell("""
df.info()
"""))

# EDA
cells.append(nbf.v4.new_markdown_cell("## 2. Analyse Exploratoire (EDA)"))
cells.append(nbf.v4.new_code_cell("""
# Distribution de la variable cible (Coût)
plt.figure(figsize=(10, 6))
sns.histplot(df['cout_traitement'], kde=True, color='blue')
plt.title('Distribution des Coûts de Traitement')
plt.show()

# Relation Poids vs Coût
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='poids_kg', y='cout_traitement', hue='type_dechet')
plt.title('Relation Poids vs Coût par Type de Déchet')
plt.show()

# Coût moyen par Pays
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='pays', y='cout_traitement', estimator=np.mean)
plt.title('Coût Moyen par Pays')
plt.xticks(rotation=45)
plt.show()
"""))

# Preprocessing
cells.append(nbf.v4.new_markdown_cell("## 3. Préparation des Données"))
cells.append(nbf.v4.new_code_cell("""
# Séparation Features / Target
# Nous voulons prédire 'cout_traitement'
# Nous utilisons 'poids_kg', 'distance_traitement_km', 'type_dechet', 'niveau_risque', 'pays', 'region' comme features.
# 'hopital' est potentiellement trop spécifique (cardinalité élevée), mais nous pouvons l'inclure si nous pensons qu'il y a un biais par hôpital.
# Pour ce modèle généraliste, nous allons ignorer 'hopital' et 'date_collecte' (sauf si feature engineering sur le mois).

X = df[['poids_kg', 'distance_traitement_km', 'type_dechet', 'niveau_risque', 'pays', 'region', 'type_conteneur']]
y = df['cout_traitement']

# Gestion des valeurs manquantes (si présentes)
# Variables numériques
numeric_features = ['poids_kg', 'distance_traitement_km']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Variables catégorielles
categorical_features = ['type_dechet', 'niveau_risque', 'pays', 'region', 'type_conteneur']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Préprocesseur global
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape X_train:", X_train.shape)
print("Shape y_train:", y_train.shape)
"""))

# Modelling
cells.append(nbf.v4.new_markdown_cell("## 4. Modélisation et Comparaison"))
cells.append(nbf.v4.new_code_cell("""
# Définition des 4 modèles
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

results = {}

print("Entraînement des modèles en cours...")
for name, model in models.items():
    # Création du pipeline complet
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    # Entraînement
    clf.fit(X_train, y_train)
    
    # Prédiction
    y_pred = clf.predict(X_test)
    
    # Évaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae, 'Model': clf}
    
    print(f"✅ {name}: RMSE={rmse:.2f}, R2={r2:.3f}")

"""))

# Visualization of Results
cells.append(nbf.v4.new_markdown_cell("## 5. Visualisation des Performances"))
cells.append(nbf.v4.new_code_cell("""
res_df = pd.DataFrame(results).T.drop('Model', axis=1) # Exclure l'objet modèle pour l'affichage
display(res_df)

# Graphique R2
plt.figure(figsize=(10, 5))
sns.barplot(x=res_df.index, y=res_df['R2'].astype(float))
plt.title('Comparaison R2 Score')
plt.ylabel('R2 Score')
plt.show()
"""))

# Save Best Model
cells.append(nbf.v4.new_markdown_cell("## 6. Sauvegarde du Meilleur Modèle"))
cells.append(nbf.v4.new_code_cell("""
# Identifier le meilleur modèle (basé sur R2)
best_model_name = res_df['R2'].astype(float).idxmax()
best_pipeline = results[best_model_name]['Model']

print(f"🏆 Le meilleur modèle est : {best_model_name}")

# Sauvegarder le modèle
model_path = os.path.join(MODELS_DIR, 'best_model_cout.pkl')
joblib.dump(best_pipeline, model_path)
print(f"Modèle sauvegardé sous : {model_path}")

# Sauvegarder aussi les performances pour l'app web
import json
perf_path = os.path.join(MODELS_DIR, 'performances.json')
# Convertir l'index et les valeurs float pour JSON
perf_dict = res_df.to_dict(orient='index')
with open(perf_path, 'w') as f:
    json.dump(perf_dict, f, indent=4)
print(f"Performances sauvegardées sous : {perf_path}")
"""))

# Add specific predictors for other targets (Mock-up or actual simple models for the app needs)
# The App predicts: cout_estime, niveau_risque, mode_elimination, conformite.
# Currently app uses random. Let's make simple classifiers for them too if requested, 
# but the user asked for "entrainer au moins 4 model differents", which we did for Cost.
# We can just reuse the pipeline for logic or train quick classifiers.
# To keep notebook clean, let's focus on Cost as the primary 'AI' task, 
# but saving metadata for others might be useful.

cells.append(nbf.v4.new_markdown_cell("## 7. Entraînement des Classifieurs Annexes (Risque, Élimination)"))
cells.append(nbf.v4.new_code_cell("""
from sklearn.ensemble import RandomForestClassifier

# Prédire 'niveau_risque'
y_risk = df['niveau_risque']
# Simuler des features
clf_risk = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(n_estimators=50))])
clf_risk.fit(X_train, df.loc[X_train.index, 'niveau_risque'])
joblib.dump(clf_risk, os.path.join(MODELS_DIR, 'model_risk.pkl'))
print("Modèle Risque sauvegardé.")

# Prédire 'mode_elimination'
y_mode = df['mode_elimination']
clf_mode = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(n_estimators=50))])
clf_mode.fit(X_train, df.loc[X_train.index, 'mode_elimination'])
joblib.dump(clf_mode, os.path.join(MODELS_DIR, 'model_mode.pkl'))
print("Modèle Mode Élimination sauvegardé.")
"""))

nb['cells'] = cells

with open('/home/axel-renaud/Images/PROJET_INF_365/notebooks/NoteBook/Travail1.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
