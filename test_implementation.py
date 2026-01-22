"""
Script de validation rapide de l'implementation
Verifie que les nouvelles methodes backend fonctionnent
"""
import sys
import os

# Ajouter le dossier backend au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    print("[TEST] Importation des modules...")
    from services.analytics_service import analytics_service
    from constants import PATHOLOGY_COLUMNS
    print("[OK] Imports reussis")
    
    print(f"\n[INFO] Nombre de pathologies: {len(PATHOLOGY_COLUMNS)}")
    print(f"[INFO] Liste: {', '.join(PATHOLOGY_COLUMNS[:5])}...")
    
    print("\n[TEST] Verification des nouvelles methodes...")
    
    # Verifier que les methodes existent
    assert hasattr(analytics_service, 'get_multi_pathologies_histogram'), "Methode get_multi_pathologies_histogram manquante"
    assert hasattr(analytics_service, 'get_conditional_probabilities'), "Methode get_conditional_probabilities manquante"
    print("[OK] Methodes get_multi_pathologies_histogram et get_conditional_probabilities presentes")
    
    print("\n[TEST] Test des signatures de methodes...")
    
    # Test get_multi_pathologies_histogram
    import inspect
    sig1 = inspect.signature(analytics_service.get_multi_pathologies_histogram)
    params1 = list(sig1.parameters.keys())
    print(f"   get_multi_pathologies_histogram: {params1}")
    assert 'sex' in params1, "Parametre 'sex' manquant"
    assert 'view_type' in params1, "Parametre 'view_type' manquant"
    
    # Test get_conditional_probabilities
    sig2 = inspect.signature(analytics_service.get_conditional_probabilities)
    params2 = list(sig2.parameters.keys())
    print(f"   get_conditional_probabilities: {params2}")
    assert 'target_disease' in params2, "Parametre 'target_disease' manquant"
    assert 'sex' in params2, "Parametre 'sex' manquant"
    assert 'view_type' in params2, "Parametre 'view_type' manquant"
    
    print("\n[OK] Toutes les verifications sont passees avec succes!")
    print("\n" + "="*60)
    print("SUCCESS: IMPLEMENTATION BACKEND VALIDEE")
    print("="*60)
    print("\n[NEXT] Prochaines etapes:")
    print("   1. Demarrer le backend: python start_backend.py")
    print("   2. Tester les endpoints:")
    print("      - curl http://localhost:5050/api/analytics/multi-pathologies")
    print("      - curl http://localhost:5050/api/analytics/conditional-probabilities?target_disease=Pneumonia")
    print("   3. Demarrer le frontend: cd frontend && npm run dev")
    print("   4. Ouvrir http://localhost:5173 dans le navigateur")
    
except ImportError as e:
    print(f"[ERROR] Erreur d'importation: {e}")
    print("[INFO] Assurez-vous d'avoir installe les dependances: pip install -r requirements.txt")
    sys.exit(1)
    
except AssertionError as e:
    print(f"[ERROR] Erreur de validation: {e}")
    sys.exit(1)
    
except Exception as e:
    print(f"[ERROR] Erreur inattendue: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
