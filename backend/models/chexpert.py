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


# =========================================================================
# Nouveaux modèles pour le Dashboard Analytics 10 Visualisations
# =========================================================================

class AgePyramidData(TypedDict):
    """Données pour la pyramide des âges"""
    age_bins: List[str]
    male_counts: List[int]
    female_counts: List[int]


class BoxplotData(TypedDict):
    """Données pour le boxplot d'une pathologie"""
    pathology: str
    min: float
    q1: float
    median: float
    q3: float
    max: float
    outliers: List[float]
    count: int


class PatientFrequencyData(TypedDict):
    """Données pour l'histogramme de fréquence patient"""
    bins: List[str]
    counts: List[int]
    total_patients: int
    mean_images: float
    median_images: float
    max_images: int


class PrevalenceData(TypedDict):
    """Données pour le bar chart de prévalence"""
    pathology: str
    positive_count: int
    negative_count: int
    prevalence_percent: float


class CooccurrenceData(TypedDict):
    """Données pour la heatmap de co-occurrence"""
    matrix: List[List[int]]
    pathologies: List[str]
    max_value: int


class UncertaintyStackedData(TypedDict):
    """Données pour le stacked bar chart d'incertitude"""
    pathology: str
    positive_count: int
    uncertain_count: int
    negative_count: int
    positive_percent: float
    uncertain_percent: float


class UncertaintyTreemapData(TypedDict):
    """Données pour le treemap d'incertitude"""
    name: str
    value: int
    percent: float


class UncertaintyByViewData(TypedDict):
    """Données d'incertitude par type de vue"""
    view_type: str
    uncertain_count: int
    positive_count: int
    negative_count: int
    uncertain_percent: float
    total_labels: int


class ViewDistributionItem(TypedDict):
    """Item de distribution des types de vues"""
    name: str
    value: int
    percent: float


class ViewDistributionData(TypedDict):
    """Données complètes de distribution des vues"""
    frontal_lateral: List[ViewDistributionItem]
    ap_pa: List[ViewDistributionItem]


class PathologyDevicesData(TypedDict):
    """Données pour le grouped bar chart pathologies vs devices"""
    pathology: str
    with_devices: int
    without_devices: int
    with_devices_percent: float
    without_devices_percent: float
