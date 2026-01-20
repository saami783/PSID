"""
Modèles de données pour le dataset CheXpert
Utilise TypedDict pour la validation de type des réponses API
"""
from typing import TypedDict, List, Optional, Dict


class PathologyStats(TypedDict):
    """Statistiques pour une pathologie"""
    pathology: str
    positive_count: int
    negative_count: int
    uncertain_count: int
    percentage: float


class DemographicsStats(TypedDict):
    """Statistiques démographiques"""
    total_patients: int
    total_images: int
    average_age: float
    median_age: float
    age_std: float
    age_min: int
    age_max: int
    sex_distribution: Dict[str, int]
    images_per_patient: float


class DatasetInfo(TypedDict):
    """Informations générales sur le dataset"""
    total_rows: int
    total_columns: int
    pathology_columns: List[str]
    missing_values_summary: Dict[str, int]


class CorrelationData(TypedDict):
    """Données de corrélation entre pathologies"""
    correlation_matrix: List[List[float]]
    pathologies: List[str]
