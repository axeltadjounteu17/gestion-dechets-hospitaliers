#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
app.py
======

Application web Flask pour le système de gestion des déchets hospitaliers.
Permet de visualiser les données, faire des prédictions et consulter les performances des modèles.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

# ============================================================================
# CONFIGURATION
# ============================================================================

from config import config

app = Flask(__name__)
app.config.from_object(config['development'])

# --- NOUVEAU MODULE (SÉPARÉ) ---
from future_routes import future_bp
app.register_blueprint(future_bp)
# ------------------------------

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'dechets_hospitaliers.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')

# Charger les données globalement
df = pd.read_csv(DATA_PATH)

# Charger les modèles (si disponibles)
models = {}
best_model = None

def load_models():
    global models, best_model
    try:
        # Charger le modèle de coût principal
        cout_model_path = os.path.join(MODELS_DIR, 'best_model_cout.pkl')
        if os.path.exists(cout_model_path):
            models['cout'] = joblib.load(cout_model_path)
            best_model = models['cout']
            print("✅ Modèle de coût chargé.")

        # Charger les modèles secondaires
        risk_model_path = os.path.join(MODELS_DIR, 'model_risk.pkl')
        if os.path.exists(risk_model_path):
            models['risk'] = joblib.load(risk_model_path)
            print("✅ Modèle de risque chargé.")
            
        mode_model_path = os.path.join(MODELS_DIR, 'model_mode.pkl')
        if os.path.exists(mode_model_path):
            models['mode'] = joblib.load(mode_model_path)
            print("✅ Modèle de mode d'élimination chargé.")
            
    except Exception as e:
        print(f"⚠️ Erreur lors du chargement des modèles : {e}")

# Appel initial
load_models()

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Page d'accueil avec vue d'ensemble"""
    stats = {
        'total_dechets': len(df),
        'total_hopitaux': df['hopital'].nunique(),
        'total_pays': df['pays'].nunique(),
        'cout_moyen': f"{df['cout_traitement'].mean():.2f}",
        'poids_total': f"{df['poids_kg'].sum():.2f}",
        'incidents': df[df['incident'] == 'Oui'].shape[0]
    }
    return render_template('index.html', stats=stats)


@app.route('/dashboard')
def dashboard():
    """Tableau de bord avec visualisations interactives"""
    
    # Graphique 1 : Distribution des types de déchets
    type_counts = df['type_dechet'].value_counts()
    fig_types = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title='Distribution des Types de Déchets',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    # Graphique 2 : Coût par type de déchet
    fig_cout = px.box(
        df,
        x='type_dechet',
        y='cout_traitement',
        title='Coût de Traitement par Type de Déchet ($)',
        color='type_dechet'
    )
    
    # Graphique 3 : Évolution temporelle
    df_temp = df.copy()
    try:
        df_temp['date_collecte'] = pd.to_datetime(df_temp['date_collecte'])
        df_temp['mois'] = df_temp['date_collecte'].dt.to_period('M').astype(str)
        monthly = df_temp.groupby('mois').size().reset_index(name='count')
        fig_temps = px.line(
            monthly,
            x='mois',
            y='count',
            title='Évolution Mensuelle des Collectes',
            markers=True
        )
    except Exception as e:
        print(f"Erreur graphe temporel: {e}")
        fig_temps = px.line(title="Données temporelles non disponibles")
    
    # Graphique 4 : Niveau de risque
    risque_counts = df['niveau_risque'].value_counts()
    fig_risque = px.bar(
        x=risque_counts.index,
        y=risque_counts.values,
        title='Répartition des Niveaux de Risque',
        labels={'x': 'Niveau de Risque', 'y': 'Nombre'},
        color=risque_counts.index,
        color_discrete_map={'Faible': 'green', 'Moyen': 'orange', 'Élevé': 'red'}
    )
    
    # Convertir en JSON pour Plotly.js
    graphs = {
        'types': json.dumps(fig_types, cls=PlotlyJSONEncoder),
        'cout': json.dumps(fig_cout, cls=PlotlyJSONEncoder),
        'temps': json.dumps(fig_temps, cls=PlotlyJSONEncoder),
        'risque': json.dumps(fig_risque, cls=PlotlyJSONEncoder)
    }
    
    return render_template('dashboard.html', graphs=graphs)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    """Page de prédiction interactive"""
    
    if request.method == 'POST':
        # Récupérer les données du formulaire
        input_data = {
            'hopital': request.form.get('hopital'),
            'pays': request.form.get('pays'),
            'region': request.form.get('region'),
            'type_dechet': request.form.get('type_dechet'),
            'poids_kg': float(request.form.get('poids_kg', 0)),
            'distance_traitement_km': float(request.form.get('distance_km', 0)),
            'type_conteneur': request.form.get('type_conteneur'),
            'service_origine': request.form.get('service_origine'),
            # Valeur par défaut pour niveau_risque si non fourni (pour le modèle de coût)
            'niveau_risque': 'Moyen', 
        }

        # Création DataFrame pour prédiction
        X_pred = pd.DataFrame([input_data])
        
        predictions = {}

        try:
            # Prédiction du Risque (si modèle disponible)
            if 'risk' in models:
                pred_risk = models['risk'].predict(X_pred)[0]
                input_data['niveau_risque'] = pred_risk # Mise à jour pour le coût
                predictions['niveau_risque'] = pred_risk
            else:
                predictions['niveau_risque'] = np.random.choice(['Faible', 'Moyen', 'Élevé'])

            # Prédiction du Mode d'Élimination (si modèle disponible)
            if 'mode' in models:
                predictions['mode_elimination'] = models['mode'].predict(X_pred)[0]
            else:
                predictions['mode_elimination'] = "Incinération"

            # Prédiction du Coût (Modèle principal)
            if 'cout' in models:
                # S'assurer que le DataFrame a les mêmes colonnes que lors de l'entraînement
                # Le pipeline se charge des transformations, mais il faut les colonnes brutes.
                # Le modèle de coût utilise risque comme feature, donc on l'a mis à jour.
                cost_pred = models['cout'].predict(X_pred)[0]
                predictions['cout_estime'] = max(0, float(cost_pred)) # Pas de coût négatif
            else:
                predictions['cout_estime'] = np.random.uniform(5000, 80000)

            # Conformité (simulée pour l'instant ou modèle simple si ajouté)
            predictions['conformite'] = "Oui" if predictions['niveau_risque'] != 'Élevé' else "Partielle"

        except Exception as e:
            print(f"Erreur prédiction: {e}")
            return jsonify({'error': str(e)}), 500
        
        return jsonify(predictions)
    
    # GET : afficher le formulaire
    options = {
        'hopitaux': sorted(df['hopital'].unique()),
        'pays': sorted(df['pays'].unique()),
        'regions': sorted(df['region'].unique()),
        'types_dechet': sorted(df['type_dechet'].unique()),
        'conteneurs': sorted(df['type_conteneur'].unique()),
        'services': sorted(df['service_origine'].dropna().unique())
    }
    
    return render_template('prediction.html', options=options)


@app.route('/data')
def data():
    """Page d'exploration des données"""
    
    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 50
    start = (page - 1) * per_page
    end = start + per_page
    
    data_page = df.iloc[start:end].to_dict('records')
    total_pages = (len(df) + per_page - 1) // per_page
    
    return render_template('data.html', 
                         data=data_page, 
                         page=page, 
                         total_pages=total_pages,
                         columns=df.columns.tolist())


@app.route('/performance')
def performance():
    """Page des performances des modèles"""
    
    # Charger les résultats réels depuis JSON
    perf_path = os.path.join(MODELS_DIR, 'performances.json')
    models_perf = {}
    
    if os.path.exists(perf_path):
        with open(perf_path, 'r') as f:
            models_perf = json.load(f)
    else:
        # Fallback si pas de fichier
        models_perf = {
            'Erreur': {'Message': 'Veuillez entraîner les modèles via le notebook.'}
        }
    
    # Graphique de comparaison R2
    models_names = list(models_perf.keys())
    scores = [models_perf[m].get('R2', 0) for m in models_names]
    
    fig = px.bar(
        x=models_names,
        y=scores,
        title='Performance des Modèles (R² Score)',
        labels={'x': 'Modèle', 'y': 'R² Score'},
        color=scores,
        color_continuous_scale='Viridis'
    )
    
    graph = json.dumps(fig, cls=PlotlyJSONEncoder)
    
    return render_template('performance.html', 
                         models=models_perf,
                         graph=graph)


@app.route('/api/stats')
def api_stats():
    """API pour obtenir les statistiques"""
    stats = {
        'total_dechets': int(len(df)),
        'total_hopitaux': int(df['hopital'].nunique()),
        'cout_moyen': float(df['cout_traitement'].mean()),
        'poids_total': float(df['poids_kg'].sum()),
        'par_type': df['type_dechet'].value_counts().to_dict(),
        'par_risque': df['niveau_risque'].value_counts().to_dict()
    }
    return jsonify(stats)


# --- NOUVELLES API POUR LE FORMULAIRE DYNAMIQUE ---

@app.route('/api/regions_by_pays')
def api_regions_by_pays():
    """API: Récupérer les régions pour un pays donné"""
    pays = request.args.get('pays', '')
    if not pays:
        # Si pas de pays, renvoyer vide ou toutes
        return jsonify({'regions': []})
    
    regions = sorted(df[df['pays'] == pays]['region'].unique().tolist())
    return jsonify({'regions': regions})


@app.route('/api/hopitaux_by_region')
def api_hopitaux_by_region():
    """API: Récupérer les hôpitaux pour une région donnée"""
    region = request.args.get('region', '')
    pays = request.args.get('pays', '')
    
    data_filt = df
    if pays:
        data_filt = data_filt[data_filt['pays'] == pays]
    if region:
        data_filt = data_filt[data_filt['region'] == region]
        
    hopitaux = sorted(data_filt['hopital'].unique().tolist())
    return jsonify({'hopitaux': hopitaux})


@app.route('/api/details_by_hopital')
def api_details_by_hopital():
    """API pour obtenir les détails spécifiques à un hôpital"""
    hopital = request.args.get('hopital', '')
    if not hopital:
        return jsonify({'error': 'Paramètre hopital requis'}), 400
    
    df_filtered = df[df['hopital'] == hopital]
    
    # Pour 'poids_kg' et 'distance', on pourrait renvoyer des moyennes, ou rien.
    # On renvoie les options disponibles dans l'historique de cet hôpital
    details = {
        'types_dechet': sorted(df_filtered['type_dechet'].unique().tolist()),
        'conteneurs': sorted(df_filtered['type_conteneur'].unique().tolist()),
        'services': sorted(df_filtered['service_origine'].dropna().unique().tolist())
    }
    
    return jsonify(details)


# ============================================================================
# LANCEMENT
# ============================================================================

if __name__ == '__main__':
    # Créer les dossiers nécessaires
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Lancer l'application
    app.run(debug=True, host='0.0.0.0', port=5000)
