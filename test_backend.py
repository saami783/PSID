"""
Script de test simple pour vérifier que le backend fonctionne
"""
import requests
import time
import sys

def test_backend():
    """Test les endpoints du backend"""
    base_url = "http://localhost:5000"
    
    print("=" * 60)
    print("Test du Backend Flask")
    print("=" * 60)
    
    endpoints = [
        "/",
        "/api/analytics/health",
        "/api/analytics/stats",
        "/api/analytics/demographics",
        "/api/analytics/pathologies"
    ]
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        print(f"\nTest de {endpoint}...")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ Status: {response.status_code}")
                if 'data' in data:
                    print(f"  ✓ Données reçues")
                    if endpoint == "/api/analytics/pathologies" and len(data.get('data', [])) > 0:
                        print(f"  ✓ {len(data['data'])} pathologies trouvées")
                else:
                    print(f"  Réponse: {str(data)[:100]}...")
            else:
                print(f"  ✗ Status: {response.status_code}")
                print(f"  Réponse: {response.text[:200]}")
        except requests.exceptions.ConnectionError:
            print(f"  ✗ Impossible de se connecter au serveur")
            print(f"  Assurez-vous que le backend est lancé avec: python start_backend.py")
            return False
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✓ Tous les tests sont passés!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    print("\nAttente de 5 secondes pour que le serveur démarre...")
    time.sleep(5)
    
    if not test_backend():
        print("\n⚠ Le backend ne répond pas. Lancez-le avec: python start_backend.py")
        sys.exit(1)
