"""
Data Loader pour le dataset CheXpert avec pattern Singleton
Évite de recharger le CSV de 11 Go à chaque requête
"""
from typing import Optional
import pandas as pd
from pathlib import Path
import os
import logging
from dotenv import load_dotenv

from backend.constants import PATHOLOGY_COLUMNS, DEFAULT_DATA_PATH, DEFAULT_VALID_PATH

# Configuration du logger
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()


class CheXpertDataLoader:
    """
    Singleton pour charger et gérer le dataset CheXpert
    Le CSV n'est chargé qu'une seule fois au démarrage
    """
    _instance: Optional['CheXpertDataLoader'] = None
    _data: Optional[pd.DataFrame] = None
    _valid_data: Optional[pd.DataFrame] = None
    
    def __new__(cls) -> 'CheXpertDataLoader':
        """Implémentation du pattern Singleton"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_data(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Charge le CSV une seule fois, utilise le cache ensuite
        
        Args:
            csv_path: Chemin vers le fichier CSV (optionnel, utilise DATA_PATH par défaut)
        
        Returns:
            DataFrame pandas contenant les données
        """
        if self._data is None:
            # Utiliser le chemin depuis .env ou celui fourni
            if csv_path is None:
                csv_path = os.getenv('DATA_PATH', DEFAULT_DATA_PATH)
            
            # Vérifier que le fichier existe
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Le fichier CSV n'existe pas : {csv_path}")
            
            # Chargement initial
            logger.info(f"Chargement du dataset depuis {csv_path}...")
            self._data = pd.read_csv(csv_path)
            logger.info(f"Dataset chargé : {len(self._data):,} lignes")
            
            # Traitement des labels incertains
            self._process_uncertain_labels()
        
        return self._data
    
    def load_valid_data(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        """
        Charge le dataset de validation
        
        Args:
            csv_path: Chemin vers le fichier CSV de validation
        
        Returns:
            DataFrame pandas contenant les données de validation
        """
        if self._valid_data is None:
            if csv_path is None:
                csv_path = os.getenv('VALID_PATH', DEFAULT_VALID_PATH)
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Le fichier CSV de validation n'existe pas : {csv_path}")
            
            logger.info(f"Chargement du dataset de validation depuis {csv_path}...")
            self._valid_data = pd.read_csv(csv_path)
            logger.info(f"Dataset de validation chargé : {len(self._valid_data):,} lignes")
            
            # Traitement des labels incertains pour le dataset de validation aussi
            self._process_uncertain_labels_valid()
        
        return self._valid_data
    
    def _process_uncertain_labels(self) -> None:
        """
        Traite les valeurs -1 (incertitude) dans le dataset d'entraînement
        Stratégie : Remplacer -1 par NaN pour ne pas biaiser les statistiques
        """
        if self._data is None:
            return
        
        # Remplacer -1 par NaN pour les colonnes de pathologies
        for col in PATHOLOGY_COLUMNS:
            if col in self._data.columns:
                self._data[col] = self._data[col].replace(-1.0, pd.NA)
    
    def _process_uncertain_labels_valid(self) -> None:
        """
        Traite les valeurs -1 (incertitude) dans le dataset de validation
        """
        if self._valid_data is None:
            return
        
        for col in PATHOLOGY_COLUMNS:
            if col in self._valid_data.columns:
                self._valid_data[col] = self._valid_data[col].replace(-1.0, pd.NA)
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Retourne les données chargées sans forcer le chargement
        
        Returns:
            DataFrame ou None si non chargé
        """
        return self._data
    
    def normalize_image_path(self, path: str) -> str:
        """
        Normalise le chemin d'image depuis le CSV vers le système de fichiers local
        Les chemins dans le CSV commencent par 'CheXpert-v1.0-small/train/...'
        mais les fichiers sont dans 'data/train/...'
        
        Args:
            path: Chemin depuis le CSV
        
        Returns:
            Chemin normalisé pour le système de fichiers local
        """
        # Remplacer le préfixe du dataset par le chemin local
        if path.startswith('CheXpert-v1.0-small/'):
            return path.replace('CheXpert-v1.0-small/', 'data/')
        return path
    
    def reload(self) -> None:
        """
        Force le rechargement du dataset (utile pour les tests ou après modification)
        """
        self._data = None
        self._valid_data = None
