"""
Point d'entrée principal de l'application Flask
Initialise Flask, enregistre les Blueprints, configure CORS et la gestion d'erreurs globale
"""
from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv

from backend.api.analytics.routes import analytics_bp
from backend.utils.response import error_response

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """
    Factory function pour créer et configurer l'application Flask
    
    Returns:
        Instance configurée de Flask
    """
    app = Flask(__name__)
    
    # Configuration depuis les variables d'environnement
    app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'development')
    app.config['FLASK_PORT'] = int(os.getenv('FLASK_PORT', 5050))
    
    # Configuration CORS
    cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:5173').split(',')
    CORS(app, origins=cors_origins, supports_credentials=True)
    
    # Enregistrement des Blueprints
    app.register_blueprint(analytics_bp)
    
    # Route racine
    @app.route('/', methods=['GET'])
    def root() -> tuple:
        """Route racine de l'API"""
        return jsonify({
            'status': 'success',
            'message': 'PSID Medical Analytics API',
            'version': '2.0.0',
            'endpoints': {
                'health': '/api/analytics/health',
                'stats': '/api/analytics/stats',
                'demographics': '/api/analytics/demographics',
                'pathologies': '/api/analytics/pathologies',
                'correlation': '/api/analytics/correlation',
                # AXE 1 : Démographie
                'age_pyramid': '/api/analytics/age-pyramid',
                'age_boxplot': '/api/analytics/age-boxplot',
                'patient_frequency': '/api/analytics/patient-frequency',
                # AXE 2 : Clinique
                'prevalence': '/api/analytics/prevalence',
                'cooccurrence': '/api/analytics/cooccurrence',
                # AXE 3 : Incertitude
                'uncertainty_stacked': '/api/analytics/uncertainty-stacked',
                'uncertainty_treemap': '/api/analytics/uncertainty-treemap',
                'uncertainty_by_view': '/api/analytics/uncertainty-by-view',
                # AXE 4 : Métadonnées
                'view_distribution': '/api/analytics/view-distribution',
                'pathology_devices': '/api/analytics/pathology-devices'
            }
        }), 200
    
    # Global Error Handler
    @app.errorhandler(404)
    def not_found(error) -> tuple:
        """Gestion des erreurs 404"""
        return error_response('Endpoint non trouvé', 404)
    
    @app.errorhandler(500)
    def internal_error(error) -> tuple:
        """Gestion des erreurs 500"""
        return error_response('Erreur interne du serveur', 500)
    
    @app.errorhandler(Exception)
    def handle_exception(e: Exception) -> tuple:
        """
        Gestion globale des exceptions pour éviter les crashes du serveur
        
        Args:
            e: Exception levée
        
        Returns:
            Réponse d'erreur formatée
        """
        # En développement, retourner le message d'erreur complet
        if app.config['FLASK_ENV'] == 'development':
            return error_response(f"Erreur : {str(e)}", 500)
        else:
            # En production, message générique
            return error_response("Une erreur est survenue", 500)
    
    return app


if __name__ == '__main__':
    app = create_app()
    port = app.config['FLASK_PORT']
    debug = app.config['FLASK_ENV'] == 'development'
    
    logger.info(f"Démarrage du serveur Flask sur le port {port}")
    logger.info(f"Mode: {app.config['FLASK_ENV']}")
    logger.info(f"CORS autorisé pour: {os.getenv('CORS_ORIGINS', 'http://localhost:5173')}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
