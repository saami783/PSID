## À propos de DeepChex
Notre projet DeepCheX est une plateforme de diagnostic en ligne assistée par l'IA, pouvant être utilisée par des radiologues.

L'idée est qu'un radiologue puisse déposer une radiographie sur notre application afin de détecter ou non des pathologies cardio-respiratoires à partir de cette radiographie.

Notre application ne doit pas se limiter à une simple classification binaire (Sain/Malade). Elle vise à identifier précisément la présence de pathologies.

## Fonctionnalités 
- Dashboard avec 12 graphiques (démographie, panorama clinique, fiabilité/ bruit, métadonnées techniques).
- Interface IA Gradio : scores par pathologie, seuils dédiés, heatmaps Grad-CAM et prétraitement adapté aux radios.


## Construit avec
- Frontend : React, Tailwind CSS
- Backend : Flask
- IA & données : PyTorch, Grad-CAM, dataset CheXpert small (`data/train.csv`, `data/valid.csv`), modèles `model_lungs_epoch_8.pth` et `best_cardio_model.pth`.

## Prérequis
- Node.js 18+ et npm.
- Python 3.10+ avec `pip`.
- Dataset CheXpert small placé dans `data/train.csv` (et `data/valid.csv` si utilisé). Lien : https://www.kaggle.com/datasets/ashery/chexpert?resource=download
- Poids de modèles disponibles à la racine (`model_lungs_epoch_8.pth`, `best_cardio_model.pth`).

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
source .venv/bin/activate
pip install -r requirements.txt
export DATA_PATH=data/train.csv
python start_backend.py
```

### Interface IA (Gradio)
```bash
source .venv/bin/activate
python app.py
```

### Accéder aux services
- Dashboard : http://localhost:5173
- Interface IA : http://127.0.0.1:7860

