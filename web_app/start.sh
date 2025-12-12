#!/bin/bash

# Script de lancement de l'application web
# Gestion des Déchets Hospitaliers

echo "🚀 Lancement de l'application web..."
echo ""

# Vérifier que nous sommes dans le bon répertoire
cd "$(dirname "$0")"

# Vérifier l'installation des dépendances
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3 -m venv venv
fi

# Activer l'environnement virtuel
source venv/bin/activate

# Installer/mettre à jour les dépendances
echo "📥 Installation des dépendances..."
pip install -q -r requirements.txt

# Lancer l'application
echo ""
echo "✅ Démarrage du serveur Flask..."
echo "🌐 Accédez à l'application sur : http://localhost:5000"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter le serveur"
echo ""

python app.py
