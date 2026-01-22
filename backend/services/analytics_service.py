"""
Service de calcul pour les analytics du dataset CheXpert
Centralise toute la logique de calcul avec système de cache
"""
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import hashlib
from functools import wraps
import logging

from backend.services.data_loader import CheXpertDataLoader
from backend.constants import PATHOLOGY_COLUMNS

logger = logging.getLogger(__name__)


def cache_result(func):
    """
    Décorateur pour mettre en cache les résultats des calculs lourds
    Le cache est basé sur les arguments de la fonction
    """
    cache: Dict[str, Any] = {}
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Créer une clé unique basée sur les arguments
        # Exclure 'self' des args pour le hash
        cache_args = args[1:] if args else ()
        key_data = f"{func.__name__}:{str(cache_args)}:{str(sorted(kwargs.items()))}"
        key = hashlib.md5(key_data.encode()).hexdigest()
        
        if key not in cache:
            logger.info(f"Cache miss pour {func.__name__}, calcul en cours...")
            cache[key] = func(*args, **kwargs)
            logger.info(f"Résultat mis en cache pour {func.__name__}")
        else:
            logger.debug(f"Cache hit pour {func.__name__}")
        
        return cache[key]
    
    # Exposer le cache pour permettre l'invalidation si nécessaire
    wrapper.cache = cache
    wrapper.clear_cache = lambda: cache.clear()
    
    return wrapper


class AnalyticsService:
    """
    Service centralisé pour tous les calculs analytics
    Utilise le Singleton CheXpertDataLoader et implémente le cache
    """
    
    _instance: Optional['AnalyticsService'] = None
    
    def __new__(cls) -> 'AnalyticsService':
        """Implémentation du pattern Singleton"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._loader = CheXpertDataLoader()
        self._raw_data: Optional[pd.DataFrame] = None
        self._initialized = True
    
    def _get_data(self) -> pd.DataFrame:
        """Récupère les données du loader"""
        return self._loader.load_data()
    
    def _get_raw_data(self) -> pd.DataFrame:
        """
        Récupère les données brutes SANS transformation des -1
        Nécessaire pour l'analyse de l'incertitude
        """
        if self._raw_data is None:
            import os
            from dotenv import load_dotenv
            from backend.constants import DEFAULT_DATA_PATH
            
            load_dotenv()
            csv_path = os.getenv('DATA_PATH', DEFAULT_DATA_PATH)
            
            if os.path.exists(csv_path):
                logger.info("Chargement des données brutes (avec -1) pour l'analyse d'incertitude...")
                self._raw_data = pd.read_csv(csv_path)
            else:
                # Fallback sur les données traitées
                self._raw_data = self._get_data().copy()
        
        return self._raw_data
    
    def _apply_filters(self, df: pd.DataFrame, sex: Optional[str] = None, 
                       view_type: Optional[str] = None,
                       target_pathology: Optional[str] = None) -> pd.DataFrame:
        """
        Applique les filtres sur le DataFrame
        
        Args:
            df: DataFrame source
            sex: Filtre par sexe ('Male' ou 'Female')
            view_type: Filtre par type de vue ('Frontal', 'Lateral', 'AP', 'PA')
            target_pathology: Filtre par pathologie cible (ex: 'Pneumonia')
        
        Returns:
            DataFrame filtré
        """
        filtered_df = df.copy()
        
        if sex and 'Sex' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Sex'] == sex]
        
        if view_type and 'Frontal/Lateral' in filtered_df.columns:
            if view_type in ['Frontal', 'Lateral']:
                filtered_df = filtered_df[filtered_df['Frontal/Lateral'] == view_type]
            elif view_type in ['AP', 'PA'] and 'AP/PA' in filtered_df.columns:
                # Filtrer aussi par AP/PA si spécifié
                filtered_df = filtered_df[filtered_df['AP/PA'] == view_type]
        
        # Nouveau filtre : pathologie cible
        if target_pathology and target_pathology in filtered_df.columns:
            # Filtrer sur les patients ayant cette pathologie (valeur = 1.0)
            filtered_df = filtered_df[filtered_df[target_pathology] == 1.0]
        
        return filtered_df
    
    # =========================================================================
    # AXE 1 : Démographie & Biais
    # =========================================================================
    
    def get_age_pyramid(self, sex: Optional[str] = None, 
                        view_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Calcule les données pour la pyramide des âges
        Retourne les comptages par tranche d'âge et par genre
        """
        df = self._apply_filters(self._get_data(), sex, view_type)
        
        # Définir les bins d'âge (0-9, 10-19, ..., 90+)
        bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 120]
        labels = ['0-9', '10-19', '20-29', '30-39', '40-49', 
                  '50-59', '60-69', '70-79', '80-89', '90+']
        
        df['age_bin'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
        
        # Grouper par tranche d'âge et sexe
        grouped = df.groupby(['age_bin', 'Sex'], observed=True).size().unstack(fill_value=0)
        
        # S'assurer que les deux colonnes existent
        male_counts = grouped.get('Male', pd.Series([0] * len(labels), index=labels)).tolist()
        female_counts = grouped.get('Female', pd.Series([0] * len(labels), index=labels)).tolist()
        
        return {
            'age_bins': labels,
            'male_counts': [int(x) for x in male_counts],
            'female_counts': [int(x) for x in female_counts]
        }
    
    def get_age_boxplot_by_pathology(self, sex: Optional[str] = None,
                                      view_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Calcule les statistiques de boxplot pour l'âge par pathologie
        Retourne min, Q1, médiane, Q3, max pour chaque pathologie
        """
        df = self._apply_filters(self._get_data(), sex, view_type)
        
        boxplot_data = []
        
        for pathology in PATHOLOGY_COLUMNS:
            if pathology not in df.columns:
                continue
            
            # Filtrer les patients avec cette pathologie (valeur = 1)
            patients_with_pathology = df[df[pathology] == 1.0]['Age'].dropna()
            
            if len(patients_with_pathology) < 5:
                # Pas assez de données pour un boxplot significatif
                continue
            
            q1 = float(patients_with_pathology.quantile(0.25))
            q3 = float(patients_with_pathology.quantile(0.75))
            iqr = q3 - q1
            
            # Calcul des outliers (valeurs au-delà de 1.5 * IQR)
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = patients_with_pathology[
                (patients_with_pathology < lower_bound) | 
                (patients_with_pathology > upper_bound)
            ].tolist()
            
            # Limiter le nombre d'outliers retournés pour éviter une réponse trop volumineuse
            outliers = outliers[:50] if len(outliers) > 50 else outliers
            
            boxplot_data.append({
                'pathology': pathology,
                'min': float(patients_with_pathology.min()),
                'q1': q1,
                'median': float(patients_with_pathology.median()),
                'q3': q3,
                'max': float(patients_with_pathology.max()),
                'outliers': [float(x) for x in outliers],
                'count': int(len(patients_with_pathology))
            })
        
        return boxplot_data
    
    def get_patient_frequency_histogram(self, sex: Optional[str] = None,
                                         view_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Calcule la distribution du nombre d'images par patient
        Permet de détecter les outliers (patients avec beaucoup d'examens)
        """
        df = self._apply_filters(self._get_data(), sex, view_type)
        
        # Extraire le PatientID depuis le chemin
        df['Patient_ID'] = df['Path'].str.extract(r'(patient\d+)')
        
        # Compter le nombre d'images par patient
        images_per_patient = df.groupby('Patient_ID').size()
        
        # Créer l'histogramme avec des bins appropriés
        max_images = int(images_per_patient.max())
        
        # Définir des bins intelligents
        if max_images <= 20:
            bins = list(range(1, max_images + 2))
        else:
            bins = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50, 100, max_images + 1]
            bins = [b for b in bins if b <= max_images + 1]
            if bins[-1] != max_images + 1:
                bins.append(max_images + 1)
        
        hist, bin_edges = np.histogram(images_per_patient, bins=bins)
        
        # Formater les labels des bins
        bin_labels = []
        for i in range(len(hist)):
            if bin_edges[i + 1] - bin_edges[i] == 1:
                bin_labels.append(str(int(bin_edges[i])))
            else:
                bin_labels.append(f"{int(bin_edges[i])}-{int(bin_edges[i + 1] - 1)}")
        
        return {
            'bins': bin_labels,
            'counts': [int(x) for x in hist.tolist()],
            'total_patients': int(len(images_per_patient)),
            'mean_images': float(images_per_patient.mean()),
            'median_images': float(images_per_patient.median()),
            'max_images': int(max_images)
        }
    
    # =========================================================================
    # AXE 2 : Panorama Clinique
    # =========================================================================
    
    def get_prevalence_bar_chart(self, sex: Optional[str] = None,
                                  view_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Calcule la prévalence de chaque pathologie
        Retourne les données triées par fréquence décroissante
        """
        df = self._apply_filters(self._get_data(), sex, view_type)
        
        prevalence_data = []
        
        for pathology in PATHOLOGY_COLUMNS:
            if pathology not in df.columns:
                continue
            
            positive_count = int((df[pathology] == 1.0).sum())
            negative_count = int((df[pathology] == 0.0).sum())
            total = len(df)
            
            prevalence_data.append({
                'pathology': pathology,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'prevalence_percent': round(positive_count / total * 100, 2) if total > 0 else 0
            })
        
        # Trier par prévalence décroissante
        prevalence_data.sort(key=lambda x: x['positive_count'], reverse=True)
        
        return prevalence_data
    
    def get_multi_pathologies_histogram(self, sex: Optional[str] = None,
                                         view_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Calcule l'histogramme de sévérité (complexité des cas)
        Compte combien de patients ont 0, 1, 2, 3+ pathologies simultanées
        
        Objectif: Montrer la complexité des cas et identifier les comorbidités
        """
        df = self._apply_filters(self._get_data(), sex, view_type)
        
        # Calculer la somme row-wise des pathologies positives (=1)
        # Exclure 'No Finding' car c'est l'absence de pathologie
        pathology_cols_to_count = [col for col in PATHOLOGY_COLUMNS 
                                    if col != 'No Finding' and col in df.columns]
        
        # Somme par ligne : nombre de pathologies par image
        df['pathology_count'] = df[pathology_cols_to_count].apply(
            lambda row: int((row == 1.0).sum()), 
            axis=1
        )
        
        # Grouper par nombre de pathologies
        count_distribution = df['pathology_count'].value_counts().sort_index()
        
        # Préparer les données pour le bar chart
        bins = []
        counts = []
        
        for num_pathologies, count in count_distribution.items():
            if num_pathologies >= 5:
                # Grouper 5+ ensemble
                label = "5+"
            else:
                label = str(num_pathologies)
            
            if label in bins:
                # Additionner si déjà présent (pour le groupe 5+)
                idx = bins.index(label)
                counts[idx] += int(count)
            else:
                bins.append(label)
                counts.append(int(count))
        
        total_patients = len(df)
        
        return {
            'bins': bins,
            'counts': counts,
            'total_patients': total_patients,
            'mean_pathologies': float(df['pathology_count'].mean()),
            'median_pathologies': float(df['pathology_count'].median()),
            'max_pathologies': int(df['pathology_count'].max())
        }
    
    def get_conditional_probabilities(self, target_disease: str,
                                       sex: Optional[str] = None,
                                       view_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Calcule les probabilités conditionnelles P(Pathologie Y | Pathologie X)
        
        Question: "Sachant que le patient a [target_disease], 
                   quelle est la probabilité qu'il ait aussi les autres ?"
        
        Args:
            target_disease: La pathologie de référence (ex: 'Pneumonia')
            sex: Filtre par sexe
            view_type: Filtre par vue
        
        Returns:
            Dictionnaire avec les fréquences conditionnelles
        """
        df = self._apply_filters(self._get_data(), sex, view_type)
        
        # Validation de la pathologie cible
        if target_disease not in PATHOLOGY_COLUMNS or target_disease not in df.columns:
            return {
                'error': f'Pathologie "{target_disease}" non trouvée',
                'target_disease': target_disease,
                'comorbidities': []
            }
        
        # Filtrer sur les patients ayant la pathologie cible
        patients_with_target = df[df[target_disease] == 1.0]
        
        if len(patients_with_target) == 0:
            return {
                'target_disease': target_disease,
                'target_count': 0,
                'comorbidities': [],
                'message': f'Aucun patient avec {target_disease} dans le dataset filtré'
            }
        
        # Calculer les fréquences des autres pathologies dans ce sous-ensemble
        comorbidities = []
        
        for pathology in PATHOLOGY_COLUMNS:
            if pathology == target_disease or pathology not in df.columns:
                continue
            
            # Compter les patients ayant AUSSI cette pathologie
            has_both = int((patients_with_target[pathology] == 1.0).sum())
            probability = has_both / len(patients_with_target) * 100
            
            comorbidities.append({
                'pathology': pathology,
                'count': has_both,
                'probability_percent': round(probability, 2)
            })
        
        # Trier par probabilité décroissante
        comorbidities.sort(key=lambda x: x['probability_percent'], reverse=True)
        
        return {
            'target_disease': target_disease,
            'target_count': int(len(patients_with_target)),
            'comorbidities': comorbidities,
            'total_patients_in_dataset': int(len(df))
        }
    
    @cache_result
    def get_cooccurrence_heatmap(self, sex: Optional[str] = None,
                                  view_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Calcule la matrice de co-occurrence entre pathologies
        Cache les résultats car le calcul est coûteux
        """
        df = self._apply_filters(self._get_data(), sex, view_type)
        
        n_pathologies = len(PATHOLOGY_COLUMNS)
        matrix = np.zeros((n_pathologies, n_pathologies), dtype=int)
        
        # Calculer la co-occurrence
        for i, p1 in enumerate(PATHOLOGY_COLUMNS):
            if p1 not in df.columns:
                continue
            for j, p2 in enumerate(PATHOLOGY_COLUMNS):
                if p2 not in df.columns:
                    continue
                # Compter les cas où les deux pathologies sont présentes
                mask = (df[p1] == 1.0) & (df[p2] == 1.0)
                matrix[i][j] = int(mask.sum())
        
        return {
            'matrix': matrix.tolist(),
            'pathologies': PATHOLOGY_COLUMNS,
            'max_value': int(matrix.max())
        }
    
    # =========================================================================
    # AXE 3 : Fiabilité & Bruit (Analyse de l'Incertitude)
    # =========================================================================
    
    def get_uncertainty_stacked_bar(self, sex: Optional[str] = None,
                                     view_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Calcule la proportion de labels positifs (1) vs incertains (-1)
        pour chaque pathologie
        """
        df = self._apply_filters(self._get_raw_data(), sex, view_type)
        
        uncertainty_data = []
        
        for pathology in PATHOLOGY_COLUMNS:
            if pathology not in df.columns:
                continue
            
            positive_count = int((df[pathology] == 1.0).sum())
            uncertain_count = int((df[pathology] == -1.0).sum())
            negative_count = int((df[pathology] == 0.0).sum())
            
            total = positive_count + uncertain_count + negative_count
            
            uncertainty_data.append({
                'pathology': pathology,
                'positive_count': positive_count,
                'uncertain_count': uncertain_count,
                'negative_count': negative_count,
                'positive_percent': round(positive_count / total * 100, 2) if total > 0 else 0,
                'uncertain_percent': round(uncertain_count / total * 100, 2) if total > 0 else 0
            })
        
        # Trier par nombre d'incertains décroissant
        uncertainty_data.sort(key=lambda x: x['uncertain_count'], reverse=True)
        
        return uncertainty_data
    
    def get_uncertainty_treemap(self, sex: Optional[str] = None,
                                 view_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Calcule les données pour le treemap de l'incertitude
        Montre la répartition du volume de "bruit" par pathologie
        """
        df = self._apply_filters(self._get_raw_data(), sex, view_type)
        
        treemap_data = []
        total_uncertain = 0
        
        for pathology in PATHOLOGY_COLUMNS:
            if pathology not in df.columns:
                continue
            
            uncertain_count = int((df[pathology] == -1.0).sum())
            total_uncertain += uncertain_count
            
            treemap_data.append({
                'name': pathology,
                'value': uncertain_count
            })
        
        # Ajouter le pourcentage
        for item in treemap_data:
            item['percent'] = round(item['value'] / total_uncertain * 100, 2) if total_uncertain > 0 else 0
        
        # Trier par valeur décroissante
        treemap_data.sort(key=lambda x: x['value'], reverse=True)
        
        return treemap_data
    
    def get_uncertainty_by_view_type(self, sex: Optional[str] = None,
                                      target_pathology: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyse si certains types de vues (AP, PA, etc.) génèrent plus d'incertitude
        
        Args:
            sex: Filtre par sexe
            target_pathology: Filtre par pathologie cible (ex: 'Pneumonia')
        
        Returns:
            Dictionnaire avec l'incertitude par Frontal/Lateral et AP/PA
        """
        # Utiliser _apply_filters pour appliquer tous les filtres
        # Note: on utilise _get_raw_data() pour conserver les valeurs -1 (incertitude)
        df = self._get_raw_data()
        
        # Appliquer les filtres manuellement car _apply_filters utilise _get_data() qui transforme -1
        if sex and 'Sex' in df.columns:
            df = df[df['Sex'] == sex]
        
        if target_pathology and target_pathology in df.columns:
            # Filtrer sur les patients ayant cette pathologie (valeur = 1.0)
            df = df[df[target_pathology] == 1.0]
        
        result = {
            'by_frontal_lateral': [],
            'by_ap_pa': []
        }
        
        # Analyse par Frontal/Lateral
        if 'Frontal/Lateral' in df.columns:
            for view in ['Frontal', 'Lateral']:
                view_df = df[df['Frontal/Lateral'] == view]
                
                total_uncertain = 0
                total_positive = 0
                total_negative = 0
                
                for pathology in PATHOLOGY_COLUMNS:
                    if pathology in view_df.columns:
                        total_uncertain += int((view_df[pathology] == -1.0).sum())
                        total_positive += int((view_df[pathology] == 1.0).sum())
                        total_negative += int((view_df[pathology] == 0.0).sum())
                
                total = total_uncertain + total_positive + total_negative
                
                result['by_frontal_lateral'].append({
                    'view_type': view,
                    'uncertain_count': total_uncertain,
                    'positive_count': total_positive,
                    'negative_count': total_negative,
                    'uncertain_percent': round(total_uncertain / total * 100, 2) if total > 0 else 0,
                    'total_labels': total
                })
        
        # Analyse par AP/PA (si la colonne existe)
        if 'AP/PA' in df.columns:
            for view in df['AP/PA'].dropna().unique():
                view_df = df[df['AP/PA'] == view]
                
                total_uncertain = 0
                total_positive = 0
                total_negative = 0
                
                for pathology in PATHOLOGY_COLUMNS:
                    if pathology in view_df.columns:
                        total_uncertain += int((view_df[pathology] == -1.0).sum())
                        total_positive += int((view_df[pathology] == 1.0).sum())
                        total_negative += int((view_df[pathology] == 0.0).sum())
                
                total = total_uncertain + total_positive + total_negative
                
                result['by_ap_pa'].append({
                    'view_type': str(view),
                    'uncertain_count': total_uncertain,
                    'positive_count': total_positive,
                    'negative_count': total_negative,
                    'uncertain_percent': round(total_uncertain / total * 100, 2) if total > 0 else 0,
                    'total_labels': total
                })
        
        return result
    
    # =========================================================================
    # AXE 4 : Métadonnées Techniques
    # =========================================================================
    
    def get_view_type_distribution(self, sex: Optional[str] = None,
                                    target_pathology: Optional[str] = None) -> Dict[str, Any]:
        """
        Calcule la distribution des types de vues
        Pour le donut chart
        
        Args:
            sex: Filtre par sexe
            target_pathology: Filtre par pathologie cible (ex: 'Pneumonia')
        
        Returns:
            Dictionnaire avec la distribution Frontal/Lateral et AP/PA
        """
        # Utiliser _apply_filters pour appliquer tous les filtres (sex + target_pathology)
        df = self._apply_filters(self._get_data(), sex=sex, target_pathology=target_pathology)
        
        result = {
            'frontal_lateral': [],
            'ap_pa': []
        }
        
        # Distribution Frontal/Lateral
        if 'Frontal/Lateral' in df.columns:
            distribution = df['Frontal/Lateral'].value_counts()
            total = distribution.sum()
            
            for view_type, count in distribution.items():
                result['frontal_lateral'].append({
                    'name': str(view_type),
                    'value': int(count),
                    'percent': round(count / total * 100, 2) if total > 0 else 0
                })
        
        # Distribution AP/PA
        if 'AP/PA' in df.columns:
            distribution = df['AP/PA'].value_counts()
            total = distribution.sum()
            
            for view_type, count in distribution.items():
                result['ap_pa'].append({
                    'name': str(view_type),
                    'value': int(count),
                    'percent': round(count / total * 100, 2) if total > 0 else 0
                })
        
        return result
    
    def get_pathology_vs_devices(self, sex: Optional[str] = None,
                                  view_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Analyse l'impact de la présence de Support Devices sur les diagnostics
        Grouped bar chart : pour chaque pathologie, comparer avec/sans devices
        """
        df = self._apply_filters(self._get_data(), sex, view_type)
        
        if 'Support Devices' not in df.columns:
            return []
        
        result = []
        
        for pathology in PATHOLOGY_COLUMNS:
            if pathology == 'Support Devices' or pathology not in df.columns:
                continue
            
            # Patients avec Support Devices = 1
            with_devices = df[df['Support Devices'] == 1.0]
            # Patients sans Support Devices (0 ou NaN)
            without_devices = df[(df['Support Devices'] == 0.0) | (df['Support Devices'].isna())]
            
            # Compter les positifs pour cette pathologie
            positive_with_devices = int((with_devices[pathology] == 1.0).sum())
            positive_without_devices = int((without_devices[pathology] == 1.0).sum())
            
            total_with = len(with_devices)
            total_without = len(without_devices)
            
            result.append({
                'pathology': pathology,
                'with_devices': positive_with_devices,
                'without_devices': positive_without_devices,
                'with_devices_percent': round(positive_with_devices / total_with * 100, 2) if total_with > 0 else 0,
                'without_devices_percent': round(positive_without_devices / total_without * 100, 2) if total_without > 0 else 0
            })
        
        # Trier par différence de prévalence
        result.sort(key=lambda x: abs(x['with_devices_percent'] - x['without_devices_percent']), reverse=True)
        
        return result


# Instance singleton du service
analytics_service = AnalyticsService()
