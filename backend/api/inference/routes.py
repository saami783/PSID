"""Routes d'inférence pour le modèle CheXpert"""
from flask import Blueprint, request
import logging

from backend.services.inference_service import inference_service
from backend.utils.response import success_response, error_response

logger = logging.getLogger(__name__)

inference_bp = Blueprint('inference', __name__, url_prefix='/api/inference')


@inference_bp.route('/predict', methods=['POST'])
def predict():
    """Reçoit une image (multipart/form-data, champ 'image') et retourne les prédictions."""
    try:
        if 'image' not in request.files:
            return error_response("Aucun fichier 'image' dans la requête", 400)

        file = request.files['image']
        result = inference_service.predict(file)

        return success_response(result), 200

    except FileNotFoundError as e:
        logger.error("Modèle introuvable: %s", e)
        return error_response(str(e), 500)
    except ValueError as e:
        logger.warning("Image invalide: %s", e)
        return error_response(str(e), 400)
    except Exception as e:  # pragma: no cover - pour sécurité runtime
        logger.exception("Erreur pendant l'inférence")
        return error_response(f"Erreur pendant l'inférence: {e}", 500)


@inference_bp.route('/health', methods=['GET'])
def inference_health():
    """Vérifie que le service d'inférence est prêt (poids accessibles)."""
    try:
        inference_service._load_model()
        return success_response({
            "status": "ready",
            "device": inference_service.device.type,
            "model_path": inference_service.model_path
        }), 200
    except Exception as e:  # pragma: no cover
        logger.exception("Inference health check failed")
        return error_response(str(e), 500)

