# SCRIPT DE PRÉSENTATION VIDÉO (5 Minutes)
## Projet : Gestion Intelligente des Déchets Hospitaliers

**Durée totale :** 5:00
**Présentateur :** [Votre Nom]

---

### 1. Introduction & Problématique (0:00 - 0:45)
**Visuel :** Slide d'introduction ou caméra face présentateur.

*   "Bonjour à tous. Aujourd'hui, je vous présente mon projet de fin de module sur la **Gestion et la Prévision des Déchets Hospitaliers en Afrique par l'Intelligence Artificielle**." 
*   "La gestion des déchets médicaux est un enjeu critique : risque de contamination, coûts élevés et logistique complexe, surtout dans des contextes à ressources limitées."
*   "**Mon objectif :** Créer une solution capable non seulement d'analyser l'historique des déchets, mais surtout de **prédire les coûts et les risques** futurs pour aider les hôpitaux à mieux planifier leur logistique."

---

### 2. Données & Analyse (Notebook Jupyter) (0:45 - 2:00)
**Visuel :** Capture d'écran du Notebook `Travail1.ipynb`. Scrollez doucement.

*   "Tout commence par la data. J'ai travaillé avec un jeu de données enrichi comprenant des informations réelles sur des hôpitaux dans plusieurs pays (Sénégal, Cameroun, RDC, etc.)."
*   **Action :** Montrer les premières cellules (Imports, Chargement des données).
*   "J'ai mis en place un pipeline complet de Data Science :"
    1.  "**Nettoyage des données :** Gestion des valeurs manquantes et incohérentes."
    2.  "**Analyse Exploratoire (EDA) :** Visualisation des distributions de coûts et des types déchets." *(Montrer un ou deux graphiques Seaborn générés)*.
    3.  "**Modélisation :** J'ai entraîné et comparé 4 algorithmes différents : Régression Linéaire, Random Forest, Gradient Boosting et SVR."
*   "Le modèle **Gradient Boosting** s'est révélé être le plus performant avec un score R² d'environ 0.85, que j'ai ensuite sauvegardé pour l'intégrer à l'application web."

---

### 3. Démonstration de l'Application Web (2:00 - 3:30)
**Visuel :** Interface de l'Application Web (Navigateur).

*   "Voici le cœur du projet : l'application web développée avec **Flask**."
*   **Action :** Naviguer sur le **Dashboard**.
    *   "Le tableau de bord offre une vue macroscopique : répartition des déchets, évolution temporelle et alertes de risques. C'est un outil de pilotage pour les décideurs."
*   **Action :** Aller sur la page **Prédiction**.
    *   "Mais la fonctionnalité clé est ce moteur de prédiction intelligent."
    *   "Regardez l'ergonomie du formulaire : quand je sélectionne 'Sénégal', seules les régions du Sénégal apparaissent. Si je choisis 'Dakar', je n'ai que les hôpitaux de Dakar. C'est une expérience utilisateur fluide."
    *   **Action :** Remplir le formulaire (ex: Hôpital Principal, Déchets Infectieux, 150kg).
    *   "En un clic, notre modèle IA interroge le backend et nous renvoie une estimation précise du coût de traitement, le niveau de risque associé et le mode d'élimination recommandé."

---

### 4. Performance & Technique (3:30 - 4:15)
**Visuel :** Page "Performance" de l'application Web.

*   **Action :** Montrer la page Performance.
*   "La transparence est essentielle. Cette page affiche en temps réel les performances des modèles embarqués."
*   "On peut voir ici que le modèle Gradient Boosting surpasse les autres sur ce jeu de données spécifique, ce qui justifie son utilisation en production."
*   "Techniquement, l'architecture repose sur :"
    *   **Backend :** Python/Flask
    *   **Frontend :** HTML5/Bootstrap/JS pour le dynamisme.
    *   **IA :** Scikit-learn avec des pipelines de transformation robustes (OneHotEncoding, Scaling)."

---

### 5. Conclusion (4:15 - 5:00)
**Visuel :** Retour caméra ou Slide de fin.

*   "Pour conclure, ce projet démontre comment l'IA peut apporter des solutions concrètes à des problèmes de santé publique."
*   "Les prochaines étapes seraient de connecter l'application aux API des transporteurs pour une planification logistique en temps réel."
*   "Merci de votre attention."

---
**Conseils pour la vidéo :**
- Parlez clairement et posément.
- Utilisez un logiciel comme OBS Studio ou Loom pour enregistrer votre écran.
- Assurez-vous que le texte des pages web est lisible (zoomez si nécessaire).
