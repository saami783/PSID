# DeepChex
Application d'analytics médicales et d'IA explicable sur le dataset CheXpert.

*Badges à configurer : Release · Tag · CircleCI · Code Climate · Codacy · Licence*

## À propos de DeepChex
DeepChex combine un dashboard d'exploration statistique et une interface d'inférence IA pour radiographies thoraciques. L'application croise 12 visualisations interactives, un filtrage avancé (sexe, type de vue, pathologie cible) et une API Flask pour rendre les biais et corrélations du dataset CheXpert visibles avant tout déploiement clinique. L'inférence est assurée par des modèles DenseNet121 (pulmonaire et cardio) avec explications Grad-CAM accessibles depuis l'interface Gradio.

## Fonctionnalités clés
- Dashboard React/Tailwind avec 12 graphiques (démographie, panorama clinique, fiabilité/ bruit, métadonnées techniques).
- API analytics Flask (/api/analytics/*) pour statistiques, cooccurrences, incertitude, pyramide des âges et distribution des vues.
- Filtres globaux (sexe, type de vue, pathologie) appliqués en temps réel sur l'ensemble des visualisations.
- Interface IA Gradio : scores par pathologie, seuils dédiés, heatmaps Grad-CAM et prétraitement adapté aux radios.
- Modèles PyTorch (DenseNet121) et pipeline de normalisation pour les fichiers DICOM exportés en PNG/JPEG.

## Construit avec
- Frontend : React 18 (Vite), Tailwind CSS, Recharts pour les visualisations.
- Backend : Flask 3, Pandas/Numpy pour l'analytics, python-dotenv pour la configuration.
- IA & données : PyTorch, Grad-CAM, dataset CheXpert small (`data/train.csv`, `data/valid.csv`), modèles `model_lungs_epoch_8.pth` et `best_cardio_model.pth`.
- Outils & ports : Vite (5173), Flask API (5050), Gradio (7860), Node 18+, Python 3.10+.

## Prérequis
- Node.js 18+ et npm.
- Python 3.10+ avec `pip`.
- Dataset CheXpert small placé dans `data/train.csv` (et `data/valid.csv` si utilisé).
- Poids de modèles disponibles à la racine (`model_lungs_epoch_8.pth`, `best_cardio_model.pth`).
- (Optionnel) `.env` pour `FLASK_ENV`, `FLASK_PORT`, `CORS_ORIGINS`, `DATA_PATH`, `VALID_PATH`.

## Manuel d'installation et d'utilisation
### Frontend (dashboard analytics)
```bash
cd frontend
npm install
npm run dev
```

### Backend analytique (Flask)
```bash
python3 -m venv .venv
source .venv/bin/activate  # sous Windows : .venv\\Scripts\\activate
pip install -r requirements.txt
export DATA_PATH=data/train.csv  # ou définir dans .env
python start_backend.py
```

### Interface IA (Gradio)
```bash
source .venv/bin/activate  # si déjà créé
python app.py
```

### Accéder aux services
- Dashboard : http://localhost:5173
- API analytics : http://localhost:5050/api/analytics (ex. `/stats`, `/demographics`, `/pathologies`, `/cooccurrence`)
- Interface IA : http://127.0.0.1:7860

## Contribution
- Fork du projet.
- Créer une branche feature (`git checkout -b feature/NouvelleFeature`).
- Commiter (`git commit -m "Ajout de NouvelleFeature"`).
- Pousser la branche (`git push origin feature/NouvelleFeature`).
- Ouvrir une Pull Request avec capture d'écran du dashboard ou de Gradio si pertinent.

## Licence
Licence à préciser (MIT/Apache 2.0 ou autre selon décision du projet).

## Auteurs
Équipe DeepChex / PSID — contributions bienvenues pour améliorer la robustesse clinique et l'explicabilité.
