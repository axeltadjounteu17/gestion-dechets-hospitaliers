# 🌐 Application Web - Gestion des Déchets Hospitaliers

Application web interactive développée avec Flask pour visualiser, analyser et prédire les données de gestion des déchets hospitaliers.

## 🚀 Fonctionnalités

### 📊 Dashboard Analytique

- Visualisations interactives avec Plotly
- Distribution des types de déchets
- Analyse des coûts par catégorie
- Évolution temporelle des collectes
- Répartition des niveaux de risque

### 🤖 Prédiction par IA

- Estimation du coût de traitement
- Classification du niveau de risque
- Recommandation du mode d'élimination
- Prédiction de conformité

### 📈 Exploration des Données

- Consultation de la base de données complète
- Pagination et filtrage
- Export des données

### 🏆 Performance des Modèles

- Métriques de performance (R², Accuracy, F1-Score)
- Comparaison visuelle des modèles
- Matrices de confusion

## 📦 Installation

### Prérequis

- Python 3.8+
- pip

### Étapes

1. **Installer les dépendances**

```bash
cd web_app
pip install -r requirements.txt
```

2. **Lancer l'application**

```bash
python app.py
```

3. **Accéder à l'application**
   Ouvrez votre navigateur à l'adresse : `http://localhost:5000`

## 📁 Structure du Projet

```
web_app/
│
├── app.py                 # Application Flask principale
├── requirements.txt       # Dépendances Python
│
├── templates/            # Templates HTML
│   ├── base.html         # Template de base
│   ├── index.html        # Page d'accueil
│   ├── dashboard.html    # Dashboard analytique
│   ├── prediction.html   # Interface de prédiction
│   ├── data.html         # Exploration des données
│   └── performance.html  # Performance des modèles
│
├── static/               # Fichiers statiques (CSS, JS, images)
│   └── (à créer si nécessaire)
│
└── models/               # Modèles ML sauvegardés
    └── (sera créé automatiquement)
```

## 🎨 Technologies Utilisées

- **Backend** : Flask 3.0
- **Visualisation** : Plotly 5.18
- **Data Science** : Pandas, NumPy, Scikit-learn
- **Frontend** : Bootstrap 5, Font Awesome, JavaScript
- **Graphiques** : Plotly.js

## 🔧 Configuration

### Variables d'environnement (optionnel)

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
```

### Mode production

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📊 API Endpoints

| Endpoint       | Méthode   | Description               |
| -------------- | --------- | ------------------------- |
| `/`            | GET       | Page d'accueil            |
| `/dashboard`   | GET       | Dashboard analytique      |
| `/prediction`  | GET, POST | Interface de prédiction   |
| `/data`        | GET       | Exploration des données   |
| `/performance` | GET       | Performance des modèles   |
| `/api/stats`   | GET       | API JSON des statistiques |

## 🎯 Utilisation

### Faire une Prédiction

1. Accédez à la page **Prédiction**
2. Remplissez le formulaire avec les informations du déchet
3. Cliquez sur **Lancer la Prédiction**
4. Consultez les résultats :
   - Coût estimé
   - Niveau de risque
   - Mode d'élimination recommandé
   - Conformité prédite

### Visualiser les Données

1. Accédez au **Dashboard**
2. Explorez les graphiques interactifs :
   - Survolez pour voir les détails
   - Zoomez et dézoomez
   - Téléchargez les graphiques

## 🚀 Améliorations Futures

- [ ] Authentification utilisateur
- [ ] Export PDF des rapports
- [ ] API REST complète
- [ ] Notifications en temps réel
- [ ] Mode sombre
- [ ] Support multilingue
- [ ] Intégration avec base de données PostgreSQL
- [ ] Déploiement sur cloud (Heroku, AWS, Azure)

## 🐛 Dépannage

### Erreur de port déjà utilisé

```bash
# Changer le port dans app.py
app.run(debug=True, host='0.0.0.0', port=8000)
```

### Problème d'import

```bash
# Réinstaller les dépendances
pip install -r requirements.txt --force-reinstall
```

## 📝 Licence

Projet académique - INF 365

## 👥 Auteur

Développé dans le cadre du projet de gestion des déchets hospitaliers

---

**🎉 Bon développement !**
