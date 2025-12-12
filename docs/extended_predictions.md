# 📘 Guide détaillé – Prédictions étendues pour le projet **Gestion des déchets hospitaliers**

## 🎯 Objectif du fichier

Ce document décrit **pas à pas** comment ajouter, entraîner et évaluer de nouvelles cibles (prédictions) :

- `poids_kg` (régression)
- `volume_m3` (régression)
- `distance_traitement_km` (régression)
- `conformite` (classification Oui/Non)
- `incident` (classification Oui/Non)
- `type_conteneur` (classification multi‑classe)
- `entreprise_transport` (classification multi‑classe)

Le processus suit exactement le même pipeline que le notebook `01_pipeline.ipynb`, mais il est **isolé** dans un script dédié (`src/extended_predictions.py`) et un notebook supplémentaire (`notebooks/02_extended_predictions.ipynb`).

---

## 📂 Structure du projet (rappel)

```
PROJET_INF_365/
│
├─ data/                     # ← CSV source
│   └─ dechets_hospitaliers.csv
│
├─ notebooks/
│   ├─ 01_pipeline.ipynb    # pipeline de base
│   └─ 02_extended_predictions.ipynb   # **nouveau** – prédictions étendues
│
├─ results/                  # figures & CSV générés
│
├─ src/
│   ├─ pipeline_full.py
│   ├─ extended_predictions.py   # **nouveau** – script complet
│   └─ … (wrappers)
│
└─ docs/
    └─ extended_predictions.md   # **ce fichier** – guide détaillé
```

---

## 🛠️ Étape 1 – Pré‑requis (installations)

```bash
# Créez un environnement virtuel (optionnel mais recommandé)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# pip install les dépendances du projet
pip install -r requirements.txt   # pandas, numpy, scikit‑learn, matplotlib, seaborn, jupyterlab
```

---

## 📓 Étape 2 – Notebook `02_extended_predictions.ipynb`

Le notebook est découpé **cellule par cellule** ; chaque cellule comporte une description (Markdown) suivie du code (Python). Vous pouvez simplement ouvrir le fichier dans JupyterLab et exécuter **Kernel → Restart & Run All**.

### Cell 1 – Titre & contexte (Markdown)

```markdown
# 🧩 Prédictions étendues – Gestion des déchets hospitaliers

Ce notebook reproduit le pipeline complet du projet, mais se concentre sur les nouvelles cibles listées ci‑dessus. Toutes les étapes (import, EDA, pré‑traitement, entraînement, évaluation, visualisation) sont détaillées.
```

### Cell 2 – Imports (Code)

```python
import os, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             r2_score, roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
warnings.filterwarnings('ignore')
%matplotlib inline
sns.set(style='whitegrid')
```

### Cell 3 – Chemins & dossiers (Code)

```python
DATA_PATH = os.path.join('..', 'data', 'dechets_hospitaliers.csv')
RESULTS_DIR = os.path.join('..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
```

### Cell 4 – Chargement du CSV (Code)

```python
print('🔎 Chargement du jeu de données…')
df = pd.read_csv(DATA_PATH)
print('Shape :', df.shape)
display(df.head())
```

### Cell 5 – Vérification des colonnes disponibles (Markdown + Code)

```markdown
## ✅ Vérification des colonnes

Nous listons les colonnes pour nous assurer que les nouvelles cibles existent.
```

```python
print('Colonnes du DataFrame :')
print(df.columns.tolist())
```

### Cell 6 – Dictionnaire `TARGETS` étendu (Code)

```python
TARGETS = {
    # Cibles déjà présentes dans le notebook 01
    'cout'        : 'cout_traitement',
    'type'        : 'type_dechet',
    'risque'      : 'niveau_risque',
    'elimination' : 'mode_elimination',
    # ---------- Nouvelles cibles ----------
    'poids'       : 'poids_kg',                # régression
    'volume'      : 'volume_m3',               # régression
    'distance'    : 'distance_traitement_km',  # régression
    'conformite'  : 'conformite',             # classification (Oui/Non)
    'incident'    : 'incident',               # classification (Oui/Non)
    'conteneur'   : 'type_conteneur',         # classification multi‑classe
    'transport'   : 'entreprise_transport'    # classification multi‑classe
}
```

### Cell 7 – Fonction utilitaire `prepare_data` (Code)

```python
def prepare_data(target_column: str):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    return X, y, numeric_features, categorical_features
```

### Cell 8 – Transformateurs (Code) – identiques à ceux du notebook 01

```python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

### Cell 9 – Fonction `train_and_evaluate` (Code) – **nouvelle version** qui crée un `ColumnTransformer` local à chaque appel (évite les conflits) :

```python
def train_and_evaluate(task: str, target_key: str):
    target_column = TARGETS[target_key]
    X, y, num_cols, cat_cols = prepare_data(target_column)

    # ColumnTransformer local
    preprocess_local = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ])

    # Split (stratify si <20 classes)
    stratify = y if (y.nunique() < 20) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify)

    # Modèle
    if task == 'regression':
        model = RandomForestRegressor(n_estimators=300, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=300, random_state=42)

    pipe = Pipeline(steps=[('preprocess', preprocess_local), ('model', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = {}
    if task == 'regression':
        metrics['R2']  = r2_score(y_test, y_pred)
        metrics['MAE'] = mean_absolute_error(y_test, y_pred)
    else:
        # Encodage du target si texte
        if y.dtype == 'object':
            le = LabelEncoder()
            y_test_enc = le.fit_transform(y_test)
            y_pred_enc = le.transform(y_pred) if isinstance(y_pred[0], str) else y_pred
            target_names = le.classes_
        else:
            y_test_enc = y_test
            y_pred_enc = y_pred
            target_names = None
        metrics['Accuracy'] = accuracy_score(y_test_enc, y_pred_enc)
        metrics['ClassificationReport'] = classification_report(
            y_test_enc, y_pred_enc, target_names=target_names, output_dict=True)
        # ROC uniquement si binaire
        if len(np.unique(y_test_enc)) == 2:
            y_proba = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test_enc, y_proba)
            metrics['ROC'] = {'fpr': fpr, 'tpr': tpr,
                               'AUC': roc_auc_score(y_test_enc, y_proba)}
    return pipe, metrics, (X_test, y_test)
```

### Cell 10 – Entraînement de **toutes** les nouvelles cibles (Code)

```python
new_keys = ['poids', 'volume', 'distance',
            'conformite', 'incident', 'conteneur', 'transport']

model_results = {}
for key in new_keys:
    task = 'regression' if key in ['poids', 'volume', 'distance'] else 'classification'
    pipe, metrics, data = train_and_evaluate(task, key)
    model_results[key] = {'pipe': pipe, 'metrics': metrics, 'data': data}
    if task == 'regression':
        print(f'🔹 {key} (régression) – R² = {metrics["R2"]:.4f}, MAE = {metrics["MAE"]:.2f}')
    else:
        print(f'🔹 {key} (classification) – Accuracy = {metrics["Accuracy"]:.2%}')
```

### Cell 11 – Fonctions de visualisation (Code) – mêmes que dans le notebook 01

```python
def plot_confusion(cm, classes, title, fname):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Vrai')
    plt.xlabel('Prédit')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname))
    plt.close()

def plot_roc(fpr, tpr, auc, title, fname):
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0,1],[0,1],'k--')
    plt.title(title)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname))
    plt.close()
```

### Cell 12 – Visualisations pour les nouvelles classifications (Code)

```python
classification_keys = ['conformite', 'incident', 'conteneur', 'transport']
for key in classification_keys:
    metrics = model_results[key]['metrics']
    X_test, y_test = model_results[key]['data']
    # Décodage si besoin
    if isinstance(y_test.iloc[0], str):
        le = LabelEncoder()
        y_test_enc = le.fit_transform(y_test)
        y_pred_enc = le.transform(
            model_results[key]['pipe'].predict(X_test))
        class_names = le.classes_
    else:
        y_test_enc = y_test
        y_pred_enc = model_results[key]['pipe'].predict(X_test)
        class_names = np.unique(y_test_enc).astype(str)
    # Confusion
    cm = confusion_matrix(y_test_enc, y_pred_enc)
    plot_confusion(cm, class_names,
                   f'Confusion matrix – {key}', f'confusion_{key}.png')
    # ROC (binaire uniquement)
    if 'ROC' in metrics:
        plot_roc(metrics['ROC']['fpr'], metrics['ROC']['tpr'],
                 metrics['ROC']['AUC'], f'ROC – {key}', f'roc_{key}.png')
```

### Cell 13 – Tableau récapitulatif étendu (Code)

```python
rows = []
for key, info in model_results.items():
    target = TARGETS[key]
    m = info['metrics']
    if 'R2' in m:  # régression
        rows.append({'Modèle': f'{target} (Régression)',
                     'R²/Accuracy': m['R2'],
                     'MAE': m['MAE']})
    else:          # classification
        rows.append({'Modèle': f'{target} (Classification)',
                     'R²/Accuracy': m['Accuracy'],
                     'MAE': np.nan})
summary_ext = pd.DataFrame(rows)
display(summary_ext)
# Sauvegarde CSV
summary_path = os.path.join(RESULTS_DIR, 'summary_extended.csv')
summary_ext.to_csv(summary_path, index=False)
print(f'📁 Tableau récapitulatif sauvegardé → {summary_path}')
```

### Cell 14 – Sélection du meilleur modèle (Code)

```python
best_reg = max(
    [(r['Modèle'], r['R²/Accuracy']) for _, r in summary_ext.iterrows()
    if 'Régression' in r['Modèle']],
    key=lambda x: x[1])
best_clf = max(
    [(r['Modèle'], r['R²/Accuracy']) for _, r in summary_ext.iterrows()
    if 'Classification' in r['Modèle']],
    key=lambda x: x[1])
print(f'🔎 Meilleur modèle **régression** → {best_reg[0]} (R² = {best_reg[1]:.4f})')
print(f'🔎 Meilleur modèle **classification** → {best_clf[0]} (Accuracy = {best_clf[1]:.2%})')
```

---

## 🖥️ Étape 3 – Script Python autonome (`src/extended_predictions.py`)

Le script reproduit exactement le notebook ci‑dessus, mais il peut être exécuté depuis le terminal :

```bash
python src/extended_predictions.py
```

Il crée les mêmes figures dans `results/` et le fichier CSV `summary_extended.csv`.

---

## 📦 Étape 4 – Exécution & vérification

1. **Lancez le notebook** `02_extended_predictions.ipynb` et assurez‑vous que toutes les cellules s’exécutent sans erreur.
2. **Vérifiez le dossier `results/`** : vous y trouverez les matrices de confusion, les courbes ROC (pour les variables binaires) et le bar‑plot global.
3. **Ouvrez `results/summary_extended.csv`** pour comparer les scores de chaque modèle.
4. **Utilisez le script** `src/extended_predictions.py` si vous préférez une exécution en ligne de commande.

---

## 🚀 Prochaines améliorations possibles

- **Hyper‑parameter tuning** avec `GridSearchCV` ou `RandomizedSearchCV` pour chaque modèle.
- **Essayer d’autres algorithmes** : XGBoost, LightGBM, CatBoost (surtout pour les variables catégorielles).
- **Feature engineering** : créer des interactions (ex. `poids * distance`) ou des variables dérivées (`jour_semaine` à partir de `date_collecte`).
- **Enregistrement des modèles** (`joblib.dump(pipe, 'model_<cible>.pkl')`) pour les ré‑utiliser dans une API ou un tableau de bord.

---

## 📚 Conclusion

Ce guide vous fournit **tout le nécessaire** pour ajouter, entraîner et évaluer de nouvelles prédictions dans le projet de gestion des déchets hospitaliers, à la fois sous forme de **notebook** détaillé et de **script Python** autonome. Vous pouvez maintenant explorer davantage les relations entre les variables et enrichir votre analyse avec les métriques les plus pertinentes.

_Bon codage !_ 🎉
