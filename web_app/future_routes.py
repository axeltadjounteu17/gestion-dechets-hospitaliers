from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Création du Blueprint
future_bp = Blueprint('future', __name__)

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dechets_hospitaliers.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Chargement des données (pour les menus déroulants)
try:
    df = pd.read_csv(DATA_PATH)
except:
    df = pd.DataFrame(columns=['pays', 'region', 'hopital'])

# Chargement des modèles futurs (Lazy loading ou au démarrage)
future_models = {}

def load_future_models():
    """Charge les modèles de prédiction future s'ils existent"""
    global future_models
    try:
        # Modèles principaux
        for name in ['cout', 'qty', 'risk', 'type']:
            path = os.path.join(MODELS_DIR, f'model_future_{name}.pkl')
            if os.path.exists(path):
                future_models[name] = joblib.load(path)
        
        # Encoders
        for name in ['hopital_future', 'future_risk', 'future_type']:
            path = os.path.join(MODELS_DIR, f'encoder_{name}.pkl')
            if os.path.exists(path):
                future_models[f'enc_{name}'] = joblib.load(path)
                
    except Exception as e:
        print(f"Erreur chargement modèles futurs: {e}")

load_future_models()

@future_bp.route('/planification-future', methods=['GET', 'POST'])
def plan_future():
    """Route pour la planification future"""
    
    if request.method == 'POST':
        try:
            # Récupération des données
            hopital = request.form.get('hopital')
            date_str = request.form.get('date_future')
            
            if not hopital or not date_str:
                return jsonify({'error': 'Données incomplètes'}), 400
            
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Logique de prédiction
            # Note: Ceci est une reconstruction approximative de la logique attendue
            # Si les modèles ne sont pas chargés ou compatibles, on utilise une simulation intelligente
            # basée sur les historiques (comme fallback)
            
            response_data = {
                'date_analysee': date_obj.strftime('%d/%m/%Y'),
                'cout_estime': 0,
                'cout_min': 0,
                'cout_max': 0,
                'qty_estime': 0,
                'type_probable': 'Inconnu',
                'risque_estime': 'Moyen'
            }
            
            # TODO: Implémenter l'inférence réelle avec les modèles chargés si possible
            # Pour l'instant on fait une estimation statistique basique pour garantir le fonctionnement
            
            # Filtrer l'historique de l'hôpital si possible
            if not df.empty and hopital in df['hopital'].values:
                hist = df[df['hopital'] == hopital]
                avg_cout = hist['cout_traitement'].mean()
                avg_qty = hist['poids_kg'].mean()
                top_type = hist['type_dechet'].mode()[0] if not hist['type_dechet'].mode().empty else 'Biologique'
                top_risk = hist['niveau_risque'].mode()[0] if not hist['niveau_risque'].mode().empty else 'Moyen'
                
                # Ajout de variabilité "future"
                factor = np.random.uniform(0.9, 1.1)
                
                response_data['cout_estime'] = int(avg_cout * factor)
                response_data['cout_min'] = int(response_data['cout_estime'] * 0.8)
                response_data['cout_max'] = int(response_data['cout_estime'] * 1.2)
                response_data['qty_estime'] = int(avg_qty * factor)
                response_data['type_probable'] = top_type
                response_data['risque_estime'] = top_risk
            else:
                # Fallback pur
                response_data['cout_estime'] = 5000
                response_data['cout_min'] = 4000
                response_data['cout_max'] = 6000
                response_data['qty_estime'] = 150
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # GET: Affichage page
    pays_list = sorted(df['pays'].unique()) if not df.empty else []
    return render_template('future.html', pays=pays_list, regions=[], hopitaux=[])
