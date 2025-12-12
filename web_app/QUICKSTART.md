# 🚀 Guide de Démarrage Rapide

## Installation en 3 étapes

### 1️⃣ Installer les dépendances

```bash
cd web_app
pip install -r requirements.txt
```

### 2️⃣ Lancer l'application

```bash
python app.py
```

**OU** utiliser le script de lancement :

```bash
./start.sh
```

### 3️⃣ Accéder à l'application

Ouvrez votre navigateur à l'adresse :

```
http://localhost:5000
```

---

## 📱 Navigation

### Page d'Accueil (`/`)

- Vue d'ensemble des statistiques
- Accès rapide aux fonctionnalités

### Dashboard (`/dashboard`)

- Graphiques interactifs
- Distribution des types de déchets
- Analyse des coûts
- Évolution temporelle

### Prédiction (`/prediction`)

1. Remplissez le formulaire
2. Cliquez sur "Lancer la Prédiction"
3. Consultez les résultats :
   - Coût estimé
   - Niveau de risque
   - Mode d'élimination
   - Conformité

### Données (`/data`)

- Exploration de la base de données
- Pagination automatique
- 50 entrées par page

### Performance (`/performance`)

- Métriques des modèles
- Comparaison visuelle
- Détails techniques

---

## 🛠️ Dépannage

### Port déjà utilisé

```bash
# Modifier le port dans app.py (ligne finale)
app.run(debug=True, host='0.0.0.0', port=8000)
```

### Erreur de module

```bash
pip install -r requirements.txt --force-reinstall
```

### Données non trouvées

Vérifiez que le fichier `../data/dechets_hospitaliers.csv` existe.

---

## 🎨 Personnalisation

### Changer les couleurs

Modifiez les variables CSS dans `templates/base.html` :

```css
:root {
  --primary: #2563eb;
  --secondary: #10b981;
  --danger: #ef4444;
  --warning: #f59e0b;
}
```

### Ajouter une nouvelle page

1. Créer une route dans `app.py`
2. Créer un template dans `templates/`
3. Ajouter un lien dans la navbar

---

## 📞 Support

Pour toute question ou problème :

- Consultez le `README.md` complet
- Vérifiez les logs dans le terminal
- Assurez-vous que toutes les dépendances sont installées

---

**Bon développement ! 🎉**
