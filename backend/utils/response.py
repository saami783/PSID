"""
Utilitaires pour formater les réponses API de manière standardisée
"""
from typing import Any, Dict, Optional, Tuple


def success_response(data: Any, meta: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Formate une réponse de succès standardisée
    
    Args:
        data: Les données à retourner
        meta: Métadonnées optionnelles (pagination, timestamps, etc.)
    
    Returns:
        Dictionnaire formaté avec status, data et meta
    """
    return {
        "status": "success",
        "data": data,
        "meta": meta or {}
    }


def error_response(message: str, code: int = 400, details: Optional[Dict] = None) -> Tuple[Dict[str, Any], int]:
    """
    Formate une réponse d'erreur standardisée
    
    Args:
        message: Message d'erreur descriptif
        code: Code HTTP d'erreur (défaut: 400)
        details: Détails supplémentaires optionnels
    
    Returns:
        Tuple contenant le dictionnaire d'erreur et le code HTTP
    """
    response = {
        "status": "error",
        "message": message
    }
    
    if details:
        response["details"] = details
    
    return response, code
