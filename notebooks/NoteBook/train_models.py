import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configuration
MODELS_DIR = "/home/axel-renaud/Images/PROJET_INF_365/web_app/models"
os.makedirs(MODELS_DIR, exist_ok=True)
DATA_PATH = "/home/axel-renaud/Images/PROJET_INF_365/notebooks/data/dechets_hospitaliers.csv"

# Load Data
print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Preprocessing
X = df[['poids_kg', 'distance_traitement_km', 'type_dechet', 'niveau_risque', 'pays', 'region', 'type_conteneur']]
y = df['cout_traitement']

numeric_features = ['poids_kg', 'distance_traitement_km']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['type_dechet', 'niveau_risque', 'pays', 'region', 'type_conteneur']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf')
}

results = {}

print("Training models...")
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'R2': r2, 'MAE': mae, 'Model': clf}
    print(f"✅ {name}: RMSE={rmse:.2f}, R2={r2:.3f}")

# Save Best Model
best_model_name = max(results, key=lambda k: results[k]['R2'])
best_pipeline = results[best_model_name]['Model']
print(f"🏆 Best model: {best_model_name}")

joblib.dump(best_pipeline, os.path.join(MODELS_DIR, 'best_model_cout.pkl'))

# Save performances
perf_dict = {k: {m: v for m, v in val.items() if m != 'Model'} for k, val in results.items()}
with open(os.path.join(MODELS_DIR, 'performances.json'), 'w') as f:
    json.dump(perf_dict, f, indent=4)

# Train other classifiers
print("Training auxiliary classifiers...")
# Risk
clf_risk = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(n_estimators=50))])
clf_risk.fit(X_train, df.loc[X_train.index, 'niveau_risque'])
joblib.dump(clf_risk, os.path.join(MODELS_DIR, 'model_risk.pkl'))

# Mode Elimination
clf_mode = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(n_estimators=50))])
clf_mode.fit(X_train, df.loc[X_train.index, 'mode_elimination'])
joblib.dump(clf_mode, os.path.join(MODELS_DIR, 'model_mode.pkl'))

print("All models saved.")
