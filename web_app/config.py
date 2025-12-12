# Configuration de l'application Flask

import os

class Config:
    """Configuration de base"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = False
    TESTING = False
    
    # Chemins
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'dechets_hospitaliers.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
    
    # Pagination
    ITEMS_PER_PAGE = 50
    
    # Flask
    JSON_AS_ASCII = False
    JSON_SORT_KEYS = False


class DevelopmentConfig(Config):
    """Configuration de développement"""
    DEBUG = True
    ENV = 'development'


class ProductionConfig(Config):
    """Configuration de production"""
    DEBUG = False
    ENV = 'production'
    # Ajoutez ici les configurations spécifiques à la production
    # comme la connexion à une vraie base de données


class TestingConfig(Config):
    """Configuration de test"""
    TESTING = True
    DEBUG = True


# Dictionnaire de configuration
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
