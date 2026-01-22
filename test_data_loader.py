"""
Script de test pour valider le chargement du dataset
"""
import os
import sys

# Ajouter le répertoire backend au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from services.data_loader import CheXpertDataLoader

def test_data_loader():
    """Test le chargement du dataset"""
    print("=" * 60)
    print("Test du Data Loader")
    print("=" * 60)
    
    # Vérifier que le fichier existe
    csv_path = os.getenv('DATA_PATH', 'data/train.csv')
    print(f"\n1. Vérification du chemin: {csv_path}")
    print(f"   Fichier existe: {os.path.exists(csv_path)}")
    
    if not os.path.exists(csv_path):
        print(f"   ERREUR: Le fichier {csv_path} n'existe pas!")
        print(f"   Chemin absolu testé: {os.path.abspath(csv_path)}")
        return False
    
    # Test du Singleton
    print("\n2. Test du pattern Singleton")
    loader1 = CheXpertDataLoader()
    loader2 = CheXpertDataLoader()
    print(f"   loader1 id: {id(loader1)}")
    print(f"   loader2 id: {id(loader2)}")
    print(f"   Singleton fonctionne: {loader1 is loader2}")
    
    # Test du chargement
    print("\n3. Test du chargement des données")
    try:
        df = loader1.load_data()
        print(f"   ✓ Données chargées avec succès!")
        print(f"   Nombre de lignes: {len(df):,}")
        print(f"   Nombre de colonnes: {len(df.columns)}")
        print(f"   Colonnes: {list(df.columns[:5])}...")
        
        # Test du cache (deuxième appel devrait être instantané)
        print("\n4. Test du cache (deuxième chargement)")
        import time
        start = time.time()
        df2 = loader1.load_data()
        elapsed = time.time() - start
        print(f"   Temps de chargement (cache): {elapsed:.4f} secondes")
        print(f"   Même DataFrame: {df is df2}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Erreur lors du chargement: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_data_loader()
    sys.exit(0 if success else 1)
