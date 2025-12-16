import os

class Config:
    """Configuration de base"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev_key_secret_12345'
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    """Configuration de développement"""
    DEBUG = True

class ProductionConfig(Config):
    """Configuration de production"""
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
