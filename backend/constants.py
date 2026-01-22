"""
Constantes partagées dans l'application
Évite la duplication de code et facilite la maintenance
"""
from typing import List

# Liste des colonnes de pathologies dans le dataset CheXpert
PATHOLOGY_COLUMNS: List[str] = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]

# Configuration par défaut
DEFAULT_DATA_PATH = 'data/train.csv'
DEFAULT_VALID_PATH = 'data/valid.csv'
DEFAULT_FLASK_PORT = 5050
DEFAULT_REACT_PORT = 5173
