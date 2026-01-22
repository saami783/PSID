"""
Routes API pour les analytics du dataset CheXpert
Blueprint Flask modulaire pour séparer les préoccupations
"""
from flask import Blueprint, request
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

from backend.services.data_loader import CheXpertDataLoader
from backend.services.analytics_service import analytics_service
from backend.utils.response import success_response, error_response
from backend.models.chexpert import (
    PathologyStats,
    DemographicsStats,
    DatasetInfo,
    CorrelationData
)
from backend.constants import PATHOLOGY_COLUMNS

logger = logging.getLogger(__name__)

# Création du Blueprint
analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')

# Instance du data loader (Singleton)
data_loader = CheXpertDataLoader()


def get_filter_params() -> tuple:
    """
    Extrait les paramètres de filtre depuis la requête
    
    Returns:
        Tuple (sex, view_type) avec None si non spécifié
    """
    sex = request.args.get('sex', None)
    view_type = request.args.get('view_type', None)
    
    # Valider les valeurs
    if sex and sex not in ['Male', 'Female']:
        sex = None
    if view_type and view_type not in ['Frontal', 'Lateral', 'AP', 'PA']:
        view_type = None
    
    return sex, view_type


@analytics_bp.route('/stats', methods=['GET'])
def get_stats() -> tuple:
    """
    Retourne les statistiques générales du dataset
    
    Returns:
        Réponse JSON avec les statistiques du dataset
    """
    try:
        df = data_loader.load_data()
        
        # Calcul des valeurs manquantes
        missing_values = {}
        for col in PATHOLOGY_COLUMNS:
            if col in df.columns:
                missing_values[col] = int(df[col].isna().sum())
        
        dataset_info: DatasetInfo = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'pathology_columns': PATHOLOGY_COLUMNS,
            'missing_values_summary': missing_values
        }
        
        return success_response(dataset_info), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des statistiques : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du chargement des statistiques : {str(e)}", 500)


@analytics_bp.route('/demographics', methods=['GET'])
def get_demographics() -> tuple:
    """
    Retourne les statistiques démographiques
    
    Returns:
        Réponse JSON avec les statistiques démographiques
    """
    try:
        df = data_loader.load_data()
        
        # Statistiques sur l'âge
        age_stats = {
            'average_age': float(df['Age'].mean()),
            'median_age': float(df['Age'].median()),
            'age_std': float(df['Age'].std()),
            'age_min': int(df['Age'].min()),
            'age_max': int(df['Age'].max())
        }
        
        # Répartition par sexe
        sex_distribution = df['Sex'].value_counts().to_dict()
        
        # Nombre de patients uniques (extraction depuis Path)
        df['Patient_ID'] = df['Path'].str.extract(r'(patient\d+)')
        unique_patients = df['Patient_ID'].nunique()
        total_images = len(df)
        images_per_patient = total_images / unique_patients if unique_patients > 0 else 0
        
        demographics: DemographicsStats = {
            'total_patients': int(unique_patients),
            'total_images': total_images,
            'average_age': age_stats['average_age'],
            'median_age': age_stats['median_age'],
            'age_std': age_stats['age_std'],
            'age_min': age_stats['age_min'],
            'age_max': age_stats['age_max'],
            'sex_distribution': {str(k): int(v) for k, v in sex_distribution.items()},
            'images_per_patient': round(images_per_patient, 2)
        }
        
        return success_response(demographics), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des statistiques démographiques : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du chargement des statistiques démographiques : {str(e)}", 500)


@analytics_bp.route('/pathologies', methods=['GET'])
def get_pathologies() -> tuple:
    """
    Retourne les statistiques pour chaque pathologie
    
    Returns:
        Réponse JSON avec les statistiques par pathologie
    """
    try:
        df = data_loader.load_data()
        
        pathology_stats_list: List[PathologyStats] = []
        
        for col in PATHOLOGY_COLUMNS:
            if col not in df.columns:
                continue
            
            # Compter les valeurs positives (1.0), négatives (0.0) et incertaines (NaN après traitement)
            positive_count = int((df[col] == 1.0).sum())
            negative_count = int((df[col] == 0.0).sum())
            uncertain_count = int(df[col].isna().sum())
            
            # Calculer le pourcentage de cas positifs parmi les valeurs non-manquantes
            total_non_missing = positive_count + negative_count
            percentage = (positive_count / total_non_missing * 100) if total_non_missing > 0 else 0.0
            
            pathology_stats: PathologyStats = {
                'pathology': col,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'uncertain_count': uncertain_count,
                'percentage': round(percentage, 2)
            }
            
            pathology_stats_list.append(pathology_stats)
        
        # Trier par nombre de cas positifs (décroissant)
        pathology_stats_list.sort(key=lambda x: x['positive_count'], reverse=True)
        
        return success_response(pathology_stats_list), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des statistiques de pathologies : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du chargement des statistiques de pathologies : {str(e)}", 500)


@analytics_bp.route('/correlation', methods=['GET'])
def get_correlation() -> tuple:
    """
    Retourne la matrice de corrélation entre les pathologies
    
    Returns:
        Réponse JSON avec la matrice de corrélation
    """
    try:
        df = data_loader.load_data()
        
        # Préparer les données pour la corrélation
        pathology_data = df[PATHOLOGY_COLUMNS].copy()
        
        # Calculer la matrice de corrélation
        correlation_matrix = pathology_data.corr()
        
        # Remplacer les NaN par 0 avant conversion
        correlation_matrix = correlation_matrix.fillna(0.0)
        
        # Convertir en numpy array avec dtype explicite pour éviter les problèmes de types pandas
        correlation_array = correlation_matrix.to_numpy(dtype=np.float64)
        
        # Remplacer tous les NaN restants par 0 (au cas où)
        correlation_array = np.nan_to_num(correlation_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Convertir en liste de listes Python native pour JSON
        correlation_list = correlation_array.tolist()
        
        # S'assurer que toutes les valeurs sont des floats Python natifs (pas numpy)
        correlation_list = [[float(x) for x in row] for row in correlation_list]
        
        correlation_data: CorrelationData = {
            'correlation_matrix': correlation_list,
            'pathologies': PATHOLOGY_COLUMNS
        }
        
        return success_response(correlation_data), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la corrélation : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du calcul de la corrélation : {str(e)}", 500)


@analytics_bp.route('/health', methods=['GET'])
def health_check() -> tuple:
    """
    Endpoint de santé pour vérifier que l'API fonctionne
    
    Returns:
        Réponse JSON simple indiquant que l'API est opérationnelle
    """
    try:
        df = data_loader.get_data()
        is_loaded = df is not None
        
        return success_response({
            'status': 'healthy',
            'data_loaded': is_loaded,
            'rows_loaded': len(df) if is_loaded else 0
        }), 200
    
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de santé : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors de la vérification de santé : {str(e)}", 500)


# =========================================================================
# AXE 1 : Démographie & Biais
# =========================================================================

@analytics_bp.route('/age-pyramid', methods=['GET'])
def get_age_pyramid() -> tuple:
    """
    Retourne les données pour la pyramide des âges
    
    Query params:
        sex: Filtre par sexe ('Male' ou 'Female')
        view_type: Filtre par type de vue
    
    Returns:
        Réponse JSON avec les comptages par tranche d'âge et genre
    """
    try:
        sex, view_type = get_filter_params()
        data = analytics_service.get_age_pyramid(sex=sex, view_type=view_type)
        return success_response(data), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la pyramide des âges : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du calcul de la pyramide des âges : {str(e)}", 500)


@analytics_bp.route('/age-boxplot', methods=['GET'])
def get_age_boxplot() -> tuple:
    """
    Retourne les statistiques de boxplot pour l'âge par pathologie
    
    Query params:
        sex: Filtre par sexe
        view_type: Filtre par type de vue
    
    Returns:
        Réponse JSON avec min, Q1, médiane, Q3, max par pathologie
    """
    try:
        sex, view_type = get_filter_params()
        data = analytics_service.get_age_boxplot_by_pathology(sex=sex, view_type=view_type)
        return success_response(data), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul des boxplots : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du calcul des boxplots : {str(e)}", 500)


@analytics_bp.route('/patient-frequency', methods=['GET'])
def get_patient_frequency() -> tuple:
    """
    Retourne l'histogramme de fréquence des images par patient
    
    Query params:
        sex: Filtre par sexe
        view_type: Filtre par type de vue
    
    Returns:
        Réponse JSON avec la distribution du nombre d'images par patient
    """
    try:
        sex, view_type = get_filter_params()
        data = analytics_service.get_patient_frequency_histogram(sex=sex, view_type=view_type)
        return success_response(data), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la fréquence patient : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du calcul de la fréquence patient : {str(e)}", 500)


# =========================================================================
# AXE 2 : Panorama Clinique
# =========================================================================

@analytics_bp.route('/prevalence', methods=['GET'])
def get_prevalence() -> tuple:
    """
    Retourne le bar chart de prévalence des pathologies
    
    Query params:
        sex: Filtre par sexe
        view_type: Filtre par type de vue
    
    Returns:
        Réponse JSON avec la prévalence de chaque pathologie
    """
    try:
        sex, view_type = get_filter_params()
        data = analytics_service.get_prevalence_bar_chart(sex=sex, view_type=view_type)
        return success_response(data), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la prévalence : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du calcul de la prévalence : {str(e)}", 500)


@analytics_bp.route('/cooccurrence', methods=['GET'])
def get_cooccurrence() -> tuple:
    """
    Retourne la matrice de co-occurrence entre pathologies (avec cache)
    
    Query params:
        sex: Filtre par sexe
        view_type: Filtre par type de vue
    
    Returns:
        Réponse JSON avec la matrice de co-occurrence
    """
    try:
        sex, view_type = get_filter_params()
        data = analytics_service.get_cooccurrence_heatmap(sex=sex, view_type=view_type)
        return success_response(data), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la co-occurrence : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du calcul de la co-occurrence : {str(e)}", 500)


# =========================================================================
# AXE 3 : Fiabilité & Bruit (Analyse de l'Incertitude)
# =========================================================================

@analytics_bp.route('/uncertainty-stacked', methods=['GET'])
def get_uncertainty_stacked() -> tuple:
    """
    Retourne les données pour le stacked bar chart (Positive vs Uncertain)
    
    Query params:
        sex: Filtre par sexe
        view_type: Filtre par type de vue
    
    Returns:
        Réponse JSON avec la proportion positive vs incertaine par pathologie
    """
    try:
        sex, view_type = get_filter_params()
        data = analytics_service.get_uncertainty_stacked_bar(sex=sex, view_type=view_type)
        return success_response(data), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de l'incertitude stacked : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du calcul de l'incertitude stacked : {str(e)}", 500)


@analytics_bp.route('/uncertainty-treemap', methods=['GET'])
def get_uncertainty_treemap() -> tuple:
    """
    Retourne les données pour le treemap de l'incertitude
    
    Query params:
        sex: Filtre par sexe
        view_type: Filtre par type de vue
    
    Returns:
        Réponse JSON avec la répartition du volume d'incertitude par pathologie
    """
    try:
        sex, view_type = get_filter_params()
        data = analytics_service.get_uncertainty_treemap(sex=sex, view_type=view_type)
        return success_response(data), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul du treemap d'incertitude : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du calcul du treemap d'incertitude : {str(e)}", 500)


@analytics_bp.route('/uncertainty-by-view', methods=['GET'])
def get_uncertainty_by_view() -> tuple:
    """
    Retourne l'analyse de l'incertitude par type de vue
    
    Query params:
        sex: Filtre par sexe
    
    Returns:
        Réponse JSON avec l'incertitude par Frontal/Lateral et AP/PA
    """
    try:
        sex, _ = get_filter_params()
        data = analytics_service.get_uncertainty_by_view_type(sex=sex)
        return success_response(data), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de l'incertitude par vue : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du calcul de l'incertitude par vue : {str(e)}", 500)


# =========================================================================
# AXE 4 : Métadonnées Techniques
# =========================================================================

@analytics_bp.route('/view-distribution', methods=['GET'])
def get_view_distribution() -> tuple:
    """
    Retourne la distribution des types de vues
    
    Query params:
        sex: Filtre par sexe
    
    Returns:
        Réponse JSON avec la distribution Frontal/Lateral et AP/PA
    """
    try:
        sex, _ = get_filter_params()
        data = analytics_service.get_view_type_distribution(sex=sex)
        return success_response(data), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul de la distribution des vues : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du calcul de la distribution des vues : {str(e)}", 500)


@analytics_bp.route('/pathology-devices', methods=['GET'])
def get_pathology_devices() -> tuple:
    """
    Retourne l'analyse de l'impact des Support Devices sur les pathologies
    
    Query params:
        sex: Filtre par sexe
        view_type: Filtre par type de vue
    
    Returns:
        Réponse JSON avec la comparaison avec/sans devices pour chaque pathologie
    """
    try:
        sex, view_type = get_filter_params()
        data = analytics_service.get_pathology_vs_devices(sex=sex, view_type=view_type)
        return success_response(data), 200
    
    except Exception as e:
        logger.error(f"Erreur lors du calcul pathologies vs devices : {str(e)}", exc_info=True)
        return error_response(f"Erreur lors du calcul pathologies vs devices : {str(e)}", 500)
