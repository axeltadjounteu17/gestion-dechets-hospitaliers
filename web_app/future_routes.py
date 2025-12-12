from flask import Blueprint, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Définition du Blueprint
future_bp = Blueprint('future', __name__)

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(BASE_DIR, 'dechets_hospitaliers.csv')

# Chargement unique des ressources au démarrage du blueprint (lazy loading possible aussi)
model_future = None
encoder_hopital = None
hopital_list = []

def load_resources():
    global model_future, encoder_hopital, hopital_list
    try:
        if os.path.exists(os.path.join(MODELS_DIR, 'model_future_cout.pkl')):
            model_future = joblib.load(os.path.join(MODELS_DIR, 'model_future_cout.pkl'))
            encoder_hopital = joblib.load(os.path.join(MODELS_DIR, 'encoder_hopital_future.pkl'))
            print("✅ Blueprint Futur : Modèles chargés.")
        
        # Charger la liste des hôpitaux pour le menu déroulant
        df = pd.read_csv(DATA_PATH)
        hopital_list = sorted(df['hopital'].unique().tolist())
        
    except Exception as e:
        print(f"⚠️ Erreur Blueprint Futur : {e}")

load_resources()

@future_bp.route('/planification-future', methods=['GET', 'POST'])
def index():
    """Page de prédiction temporelle"""
    if request.method == 'POST':
        try:
            date_str = request.form.get('date_future')
            hopital = request.form.get('hopital')
            
            if not date_str or not hopital:
                return jsonify({'error': 'Date et Hôpital requis'}), 400
                
            # Parsing de la date
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Préparation des features
            annee = date_obj.year
            mois = date_obj.month
            jour = date_obj.day
            jour_semaine = date_obj.weekday()
            jour_annee = date_obj.timetuple().tm_yday
            
            # Encodage Hôpital
            try:
                hopital_enc = encoder_hopital.transform([hopital])[0]
            except ValueError:
                # Hôpital inconnu (fallback: on prend une moyenne ou erreur)
                # Ici on simule pour ne pas casser l'app si données changent
                hopital_enc = 0 
            
            # Création vecteur
            features = pd.DataFrame([[hopital_enc, annee, mois, jour, jour_semaine, jour_annee]], 
                                  columns=['hopital_encoded', 'annee', 'mois', 'jour', 'jour_semaine', 'jour_annee'])
            
            # Prédiction
            cout_pred = model_future.predict(features)[0]
            
            # Simulation variation (intervalle de confiance simple)
            variation_min = cout_pred * 0.9
            variation_max = cout_pred * 1.1
            
            return jsonify({
                'cout_estime': round(cout_pred, 2),
                'cout_min': round(variation_min, 2),
                'cout_max': round(variation_max, 2),
                'date_analysee': date_obj.strftime('%d/%m/%Y')
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return render_template('future.html', hopitaux=hopital_list)
