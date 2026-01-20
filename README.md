# PSID - Medical Analytics & AI

Projet M2 Informatique sur le dataset CheXpert (11 Go) avec architecture modulaire Flask + React.

## 📋 Table des Matières

- [Architecture](#architecture)
- [Choix Techniques](#choix-techniques)
- [Structure du Projet](#structure-du-projet)
- [Installation](#installation)
- [Lancement](#lancement)
- [API Endpoints](#api-endpoints)
- [Architecture Détaillée](#architecture-détaillée)
- [Bonnes Pratiques Appliquées](#bonnes-pratiques-appliquées)
- [Dépannage](#dépannage)

## 🏗️ Architecture

### Stack Technique

- **Backend**: Flask 3.0+ avec Blueprints, Clean Architecture
- **Frontend**: React 18 + Vite + Tailwind CSS + Recharts
- **Data**: Dataset CheXpert-v1.0-small (11 Go)
- **Python**: 3.10+
- **Node.js**: 18+ (LTS recommandé)

### Principes Architecturaux

- **Clean Architecture**: Séparation claire des couches (API, Services, Models, Utils)
- **Separation of Concerns**: Logique métier séparée de la présentation
- **DRY (Don't Repeat Yourself)**: Constantes partagées, pas de duplication
- **SOLID**: Principes respectés dans la structure modulaire
- **Singleton Pattern**: Data loader pour éviter les rechargements du CSV de 11 Go

## 🎯 Choix Techniques

### Backend (Flask)

**Pourquoi Flask plutôt que Django/FastAPI ?**
- Flask offre plus de flexibilité pour une API REST simple
- Blueprints permettent une modularité claire
- Plus léger et adapté à notre cas d'usage (API analytics)

**Architecture modulaire avec Blueprints**
- Routes organisées par domaine (`/api/analytics/`)
- Facilite l'ajout de nouveaux endpoints
- Séparation claire des responsabilités

**Singleton Pattern pour le Data Loader**
- Le CSV de 11 Go ne doit être chargé qu'une seule fois
- Cache en mémoire pour éviter les rechargements coûteux
- Améliore drastiquement les performances

**Type Hints partout**
- Meilleure maintenabilité
- Support IDE amélioré
- Documentation implicite du code

**Logging structuré**
- Utilisation du module `logging` au lieu de `print()`
- Logs configurables par niveau
- Facilite le debugging en production

### Frontend (React)

**Pourquoi React + Vite ?**
- Vite offre un démarrage ultra-rapide (vs Create React App)
- Hot Module Replacement (HMR) pour développement fluide
- Build optimisé pour la production

**Structure modulaire**
- `components/` : Composants UI réutilisables (Card, charts)
- `features/` : Logique métier par domaine (analytics)
- `services/` : Appels API centralisés
- `hooks/` : Logique réutilisable (useAnalytics)

**Séparation des préoccupations**
- Pas de logique métier dans les composants JSX
- Hooks personnalisés pour la logique complexe
- Service API découplé des composants

**Gestion d'erreurs robuste**
- `Promise.allSettled()` pour continuer même si une API échoue
- Affichage conditionnel des données disponibles
- Messages d'erreur utilisateur-friendly

**Tailwind CSS**
- Utility-first CSS pour développement rapide
- Pas de CSS custom à maintenir
- Design system cohérent

**Recharts**
- Bibliothèque de graphiques React native
- Performante et personnalisable
- Compatible avec notre stack

## 📁 Structure du Projet

```
Projet_PSID/
├── backend/                    # API Flask modulaire
│   ├── api/                    # Blueprints Flask
│   │   └── analytics/         # Routes analytics
│   │       └── routes.py      # Endpoints API
│   ├── services/               # Logique métier & Data Layer
│   │   └── data_loader.py     # Singleton pour charger le CSV
│   ├── models/                 # Modèles de données
│   │   └── chexpert.py        # TypedDict pour les réponses
│   ├── utils/                  # Utilitaires
│   │   └── response.py        # Formatage des réponses API
│   ├── constants.py           # Constantes partagées (PATHOLOGY_COLUMNS)
│   └── app.py                 # Point d'entrée Flask (factory pattern)
│
├── frontend/                   # Application React + Vite
│   ├── src/
│   │   ├── components/        # Composants UI purs
│   │   │   ├── ui/            # Composants de base (Card)
│   │   │   └── charts/        # Composants de graphiques
│   │   ├── features/          # Logique métier par domaine
│   │   │   └── analytics/     # Feature analytics
│   │   │       ├── AnalyticsDashboard.jsx
│   │   │       └── hooks/
│   │   │           └── useAnalytics.js  # Hook personnalisé
│   │   ├── services/          # Appels API
│   │   │   └── api.js         # Service API centralisé
│   │   ├── App.jsx            # Composant racine
│   │   └── main.jsx           # Point d'entrée React
│   ├── package.json
│   └── vite.config.js         # Configuration Vite
│
├── data/                       # Dataset CheXpert (11 Go - exclu du git)
│   ├── train.csv              # Dataset d'entraînement
│   ├── valid.csv              # Dataset de validation
│   ├── train/                 # Images d'entraînement
│   └── valid/                 # Images de validation
│
├── .env                        # Variables d'environnement (non versionné)
├── .env.example                # Template des variables d'environnement
├── .gitignore                  # Exclusions Git
├── requirements.txt            # Dépendances Python
├── start_backend.py            # Script de démarrage backend
└── README.md                   # Ce fichier
```

## 🚀 Installation

### Prérequis

- **Python 3.10+** : [Télécharger Python](https://www.python.org/downloads/)
- **Node.js 18+ (LTS)** : [Télécharger Node.js](https://nodejs.org/)
- **Git** : Pour cloner le projet

### Backend

1. **Créer un environnement virtuel Python** :
```bash
python -m venv venv
# Sur Windows:
venv\Scripts\activate
# Sur Linux/Mac:
source venv/bin/activate
```

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

3. **Configurer les variables d'environnement** :
```bash
# Copier le template
cp .env.example .env

# Éditer .env avec vos valeurs (optionnel, les valeurs par défaut fonctionnent)
```

Le fichier `.env` contient :
```env
FLASK_ENV=development
FLASK_PORT=5000
REACT_PORT=5173
DATA_PATH=data/train.csv
VALID_PATH=data/valid.csv
CORS_ORIGINS=http://localhost:5173
```

### Frontend

1. **Installer les dépendances Node.js** :
```bash
cd frontend
npm install
```

2. **Configuration** :
   - L'URL de l'API backend est configurée dans `frontend/src/services/api.js`
   - Par défaut : `http://localhost:5000`
   - Peut être modifiée via la variable d'environnement `VITE_API_URL`

## ▶️ Lancement

### Option 1 : Script automatique (Recommandé)

**Terminal 1 - Backend** :
```bash
python start_backend.py
```

**Terminal 2 - Frontend** :
```bash
cd frontend
npm run dev
```

### Option 2 : Lancement manuel

**Terminal 1 - Backend** :
```bash
cd backend
python app.py
```

**Terminal 2 - Frontend** :
```bash
cd frontend
npm run dev
```

### Vérification

1. **Backend** : Ouvrez http://localhost:5000/api/analytics/health
   - Devrait retourner : `{"status":"success","data":{"status":"healthy",...}}`

2. **Frontend** : Ouvrez http://localhost:5173
   - Vous devriez voir le dashboard avec les graphiques

### ⏱️ Temps d'attente

- **Backend** : 30-60 secondes au premier démarrage (chargement du CSV de 11 Go)
- **Frontend** : Quelques secondes pour compiler

Vous verrez dans le terminal backend :
```
Chargement du dataset depuis data/train.csv...
Dataset chargé : 223,414 lignes
🚀 Démarrage du serveur Flask sur le port 5000
```

## 🔌 API Endpoints

### Base URL
```
http://localhost:5000
```

### Endpoints Disponibles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/` | Informations sur l'API |
| `GET` | `/api/analytics/health` | Vérification de santé |
| `GET` | `/api/analytics/stats` | Statistiques générales du dataset |
| `GET` | `/api/analytics/demographics` | Statistiques démographiques |
| `GET` | `/api/analytics/pathologies` | Statistiques par pathologie |
| `GET` | `/api/analytics/correlation` | Matrice de corrélation (actuellement désactivé) |

### Format des Réponses

**Succès** :
```json
{
  "status": "success",
  "data": { ... },
  "meta": {}
}
```

**Erreur** :
```json
{
  "status": "error",
  "message": "Description de l'erreur"
}
```

## 🏛️ Architecture Détaillée

### Backend - Flux de Données

```
Requête HTTP
    ↓
Flask App (app.py)
    ↓
Blueprint (routes.py)
    ↓
Data Loader (Singleton) ← CSV chargé une seule fois
    ↓
Traitement des données
    ↓
Formatage réponse (response.py)
    ↓
Réponse JSON
```

### Backend - Couches

1. **API Layer** (`backend/api/`)
   - Gère les routes HTTP
   - Validation basique
   - Appelle les services

2. **Service Layer** (`backend/services/`)
   - Logique métier
   - Accès aux données
   - Singleton pour le cache

3. **Model Layer** (`backend/models/`)
   - Définition des structures de données
   - TypedDict pour validation de type

4. **Utils Layer** (`backend/utils/`)
   - Fonctions utilitaires
   - Formatage des réponses

### Frontend - Flux de Données

```
Composant (AnalyticsDashboard)
    ↓
Hook personnalisé (useAnalytics)
    ↓
Service API (api.js)
    ↓
Fetch vers Backend
    ↓
Traitement réponse
    ↓
Mise à jour état
    ↓
Rendu composants
```

### Frontend - Organisation

1. **Components** (`components/`)
   - Composants UI purs et réutilisables
   - Pas de logique métier
   - Props simples

2. **Features** (`features/`)
   - Logique métier par domaine
   - Composants complexes
   - Hooks personnalisés

3. **Services** (`services/`)
   - Appels API centralisés
   - Gestion des erreurs réseau
   - Configuration des URLs

## ✅ Bonnes Pratiques Appliquées

### Python/Flask

- ✅ **PEP 8** : Style de code respecté
- ✅ **Type Hints** : Toutes les fonctions typées (PEP 484)
- ✅ **Docstrings** : Documentation des fonctions (PEP 257)
- ✅ **Factory Pattern** : `create_app()` pour Flask
- ✅ **Blueprints** : Modularité des routes
- ✅ **Logging** : Module `logging` au lieu de `print()`
- ✅ **Constants** : Fichier `constants.py` pour éviter la duplication
- ✅ **Error Handling** : Gestion d'erreurs globale et locale
- ✅ **Environment Variables** : Configuration via `.env`

### React

- ✅ **Functional Components** : Pas de classes
- ✅ **Custom Hooks** : Logique réutilisable
- ✅ **Separation of Concerns** : UI / Logique / Services séparés
- ✅ **Error Handling** : `Promise.allSettled()` pour résilience
- ✅ **Service Layer** : API centralisée
- ✅ **Props Validation** : Vérification des props
- ✅ **Clean JSX** : Pas de logique métier dans le rendu

### Architecture

- ✅ **DRY** : Pas de duplication de code
- ✅ **SOLID** : Principes respectés
- ✅ **Clean Architecture** : Séparation des couches
- ✅ **Singleton Pattern** : Pour le cache des données
- ✅ **Configuration Management** : Variables d'environnement

## 🔍 Décisions Techniques Expliquées

### Pourquoi un Singleton pour le Data Loader ?

Le dataset CSV fait **11 Go**. Le charger à chaque requête serait :
- ⏱️ **Lent** : 30-60 secondes par requête
- 💾 **Gourmand** : Consommation mémoire excessive
- 🚫 **Inefficace** : Les données ne changent pas pendant l'exécution

**Solution** : Singleton qui charge une fois au démarrage et met en cache.

### Pourquoi `Promise.allSettled()` au lieu de `Promise.all()` ?

`Promise.all()` échoue si **une seule** API échoue, ce qui bloque toute l'application.

`Promise.allSettled()` continue même si certaines APIs échouent, permettant d'afficher les données disponibles.

### Pourquoi des constantes partagées ?

La liste `PATHOLOGY_COLUMNS` était dupliquée dans 4 fichiers. Si on ajoute une pathologie, il faut modifier 4 endroits → risque d'erreur.

**Solution** : Fichier `constants.py` avec une seule source de vérité.

### Pourquoi un fichier `response.py` ?

Toutes les réponses API suivent le même format. Au lieu de répéter le code partout, on centralise dans `response.py`.

**Avantages** :
- Cohérence des réponses
- Facile à modifier le format
- Réutilisable

### Pourquoi séparer les composants UI des features ?

**Composants UI** (`components/ui/Card.jsx`) :
- Réutilisables partout
- Pas de dépendance métier
- Faciles à tester

**Features** (`features/analytics/`) :
- Logique métier spécifique
- Peut utiliser plusieurs composants UI
- Organisé par domaine

## 🐛 Dépannage

### Backend ne démarre pas

**Problème** : `FileNotFoundError: Le fichier CSV n'existe pas`

**Solution** :
- Vérifiez que `data/train.csv` existe
- Vérifiez le chemin dans `.env` (`DATA_PATH`)

**Problème** : Port 5000 déjà utilisé

**Solution** :
- Changez `FLASK_PORT` dans `.env`
- Ou arrêtez le processus utilisant le port 5000

### Frontend ne charge pas les données

**Problème** : Erreur CORS

**Solution** :
- Vérifiez que le backend tourne sur le port 5000
- Vérifiez `CORS_ORIGINS` dans `.env`
- Ouvrez la console du navigateur (F12) pour voir les erreurs

**Problème** : `npm` n'est pas reconnu

**Solution** :
- Installez Node.js depuis https://nodejs.org/
- Redémarrez le terminal après installation

**Problème** : Erreur de politique d'exécution PowerShell

**Solution** :
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Graphiques ne s'affichent pas

**Problème** : Backend encore en chargement

**Solution** :
- Attendez 30-60 secondes que le CSV soit chargé
- Vérifiez les logs du backend : "Dataset chargé : 223,414 lignes"

**Problème** : Erreur dans la console du navigateur

**Solution** :
- Ouvrez la console (F12)
- Vérifiez les erreurs réseau ou JavaScript
- Vérifiez que Recharts est installé : `npm list recharts`

## 📝 Notes Importantes

- ⚠️ Le dataset `data/` (11 Go) est **exclu du git** via `.gitignore`
- ⚠️ Le fichier `.env` est **exclu du git** (contient des configurations sensibles)
- ⚠️ Le chargement initial du CSV peut prendre **30-60 secondes**
- ⚠️ Les valeurs `-1` (incertitude) sont traitées comme `NaN` dans les statistiques
- ⚠️ L'endpoint `/api/analytics/correlation` est actuellement désactivé (problème de sérialisation)

## 🔄 Pour Contribuer

### Structure du Code

1. **Ajouter un nouvel endpoint** :
   - Créer/modifier dans `backend/api/analytics/routes.py`
   - Utiliser `success_response()` et `error_response()` de `utils/response.py`

2. **Ajouter une nouvelle constante** :
   - Ajouter dans `backend/constants.py`
   - Importer où nécessaire

3. **Ajouter un nouveau composant React** :
   - Créer dans `frontend/src/components/` (UI) ou `features/` (métier)
   - Utiliser les composants existants (Card, etc.)

4. **Modifier le chargement des données** :
   - Modifier `backend/services/data_loader.py`
   - Utiliser `PATHOLOGY_COLUMNS` depuis `constants.py`

### Conventions de Code

- **Python** : Type hints partout, docstrings pour les fonctions publiques
- **React** : Functional components, hooks pour la logique
- **Noms** : Clairs et descriptifs, en anglais
- **Commentaires** : Expliquer le "pourquoi", pas le "quoi"

## 📚 Ressources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)
- [Recharts Documentation](https://recharts.org/)
- [Tailwind CSS Documentation](https://tailwindcss.com/)

## 👥 Auteurs

Projet M2 Informatique - PSID Medical Analytics & AI

---

**Dernière mise à jour** : Janvier 2026
