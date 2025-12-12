import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dechets_hospitaliers.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("⏳ Chargement des données...")
df = pd.read_csv(DATA_PATH)

# Feature Engineering Temporel
print("⚙️ Préparation des données temporelles...")
df['date_collecte'] = pd.to_datetime(df['date_collecte'])

# On veut prédire le FUTUR, donc on a besoin de décomposer la date
df['annee'] = df['date_collecte'].dt.year
df['mois'] = df['date_collecte'].dt.month
df['jour'] = df['date_collecte'].dt.day
df['jour_semaine'] = df['date_collecte'].dt.dayofweek
df['jour_annee'] = df['date_collecte'].dt.dayofyear

# Features pour la prédiction
# On utilise l'hôpital + la date pour prédire le coût
X = df[['hopital', 'annee', 'mois', 'jour', 'jour_semaine', 'jour_annee']]
y = df['cout_traitement']

# Encodage de l'hôpital (Label Encoding via Pipeline pas direct car ColumnTransformer mieux)
# Pour simplifier ici (et car hopital est une string), on va utiliser un Preprocessor
# Mais RandomForest gère mal les strings sans encodage.

# On va faire un encodage manuel et sauvegarder les classes pour être sûr de la cohérence
le_hopital = LabelEncoder()
X.loc[:, 'hopital_encoded'] = le_hopital.fit_transform(X['hopital'])

# Features finales
X_final = X[['hopital_encoded', 'annee', 'mois', 'jour', 'jour_semaine', 'jour_annee']]

# Entraînement
print("🚀 Entraînement du modèle temporel...")
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_final, y)

# Sauvegarde
print("💾 Sauvegarde du modèle et de l'encodeur...")
joblib.dump(model, os.path.join(MODELS_DIR, 'model_future_cout.pkl'))
joblib.dump(le_hopital, os.path.join(MODELS_DIR, 'encoder_hopital_future.pkl'))

print("✅ Terminé ! Modèle prêt pour prédire le futur.")
