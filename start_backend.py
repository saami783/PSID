"""
Script pour démarrer le serveur Flask backend
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
os.chdir(project_root)

sys.path.insert(0, str(project_root))

csv_path = project_root / "data" / "train.csv"
if not csv_path.exists():
    print(f"ERREUR: Le fichier {csv_path} n'existe pas!")
    print(f"Répertoire courant: {os.getcwd()}")
    sys.exit(1)

print(f"Fichier CSV trouvé: {csv_path}")
print(f"Taille: {csv_path.stat().st_size / (1024*1024):.2f} MB")
print("Ce backend ne gère plus l'inférence; utilisez Gradio via http://127.0.0.1:7860/ pour les prédictions.")


from backend.app import create_app

app = create_app()
port = app.config['FLASK_PORT']
debug = app.config['FLASK_ENV'] == 'development'

print(f"\nDémarrage du serveur Flask sur le port {port}")
print(f"Mode: {app.config['FLASK_ENV']}")
print(f"CORS autorisé pour: {os.getenv('CORS_ORIGINS', 'http://localhost:5173')}")
print(f"\nAppuyez sur Ctrl+C pour arrêter le serveur\n")

app.run(host='0.0.0.0', port=port, debug=debug)
