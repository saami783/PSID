import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from PIL import Image

# --- CONFIGURATION ---
DATA_DIR = "data"
# Mets ici ton meilleur modèle (Epoque 5 normalement)
MODEL_PATH = "models/tentative 4/chexpert_densenet_epoch_5.pth"
BATCH_SIZE = 32
THRESHOLD = 0.5  # Seuil de décision : au-dessus de 50% = Malade

# Les 5 cibles exactes
TARGET_COLS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


# --- 1. SETUP TECHNIQUE (Copier-Coller standard) ---
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CheXpertDenseNet(nn.Module):
    def __init__(self, num_classes=5):
        super(CheXpertDenseNet, self).__init__()
        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x): return self.model(x)


class CheXpertDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.label_cols = TARGET_COLS

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.root_dir, row['Path'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return torch.zeros(3, 320, 320), torch.zeros(len(self.label_cols))
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        if self.transform: image = self.transform(image)
        return image, labels


# --- 2. PRÉPARATION DES DONNÉES ---
def get_valid_df():
    csv_path = os.path.join(DATA_DIR, "valid.csv")
    if not os.path.exists(csv_path):
        print("❌ Erreur : valid.csv introuvable.")
        exit()

    df = pd.read_csv(csv_path)
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']
    if 'Path' in df.columns:
        df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small', 'data', regex=False)

    # Nettoyage pour la vérité terrain (Validation)
    df[TARGET_COLS] = df[TARGET_COLS].fillna(0)
    # Dans la validation officielle CheXpert, les incertains (-1) sont souvent traités
    # comme positifs (1) pour ne pas pénaliser le modèle s'il détecte une ambiguïté.
    df[TARGET_COLS] = df[TARGET_COLS].replace(-1, 1)

    return df


# --- 3. LE MOTEUR DE CONFUSION ---
def generate_matrices():
    device = get_device()
    print(f"🔍 Analyse de fiabilité sur {device}...")

    # Chargement Modèle
    model = CheXpertDenseNet(num_classes=5).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
    except Exception as e:
        print(f"❌ Erreur modèle : {e}")
        return

    # Chargement Data
    val_df = get_valid_df()
    print(f"📊 Nombre d'images de test : {len(val_df)}")

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = CheXpertDataset(val_df, root_dir='.', transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Collecte des prédictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Calcul en cours"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            # On binarise directement : Si proba > 0.5 -> 1 (Malade), sinon 0 (Sain)
            preds = (probs > THRESHOLD).float()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    print(f"\n{'=' * 60}")
    print(f"RÉSULTATS DÉTAILLÉS (Seuil = {THRESHOLD * 100}%)")
    print(f"{'=' * 60}\n")

    # Génération des matrices par maladie
    for i, disease in enumerate(TARGET_COLS):
        # Extraction des colonnes correspondantes
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]

        # Calcul de la matrice
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # Calcul des métriques
        total = tn + fp + fn + tp
        accuracy = (tp + tn) / total
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Capacité à trouver les malades
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Capacité à ne pas alarmer les sains

        print(f"🦠 MALADIE : {disease.upper()}")
        print(f"{'-' * 30}")
        print(f"Vrais Positifs (Succès Malade) : {tp}")
        print(f"Vrais Négatifs (Succès Sain)   : {tn}")
        print(f"Faux Positifs  (Fausse Alerte) : {fp}")
        print(f"Faux Négatifs  (Maladie Ratée) : {fn}")
        print(f"{'-' * 30}")
        print(f"📈 Précision Globale : {accuracy:.2%}")
        print(f"🚨 Sensibilité       : {sensitivity:.2%} (Chance de détecter la maladie si présente)")
        print(f"🛡️ Spécificité       : {specificity:.2%} (Chance de dire 'Sain' si le patient est sain)")
        print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    generate_matrices()