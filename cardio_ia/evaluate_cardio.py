import os
import torch
import torch.nn as nn  # <-- C'était l'import manquant
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, roc_auc_score
from tqdm import tqdm
from PIL import Image

MODEL_PATH = "../../../best_cardio_model.pth"  # on prends le meilleur modele
DATA_DIR = "../data"
BATCH_SIZE = 32
IMG_SIZE = 320
TARGET = 'Cardiomegaly'


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CardioValidDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.root_dir, row['Path'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.tensor([0.0])

        label = torch.tensor([row[TARGET]], dtype=torch.float32)
        if self.transform: image = self.transform(image)
        return image, label


def get_valid_df():
    csv_path = os.path.join(DATA_DIR, "valid.csv")
    if not os.path.exists(csv_path):
        print("valid.csv introuvable.")
        exit()

    df = pd.read_csv(csv_path)
    if 'Frontal/Lateral' in df.columns: df = df[df['Frontal/Lateral'] == 'Frontal']
    if 'Path' in df.columns: df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small', 'data', regex=False)

    # Stratégie Sécurité : Incertain (-1) = Malade (1) pour l'audit
    df[TARGET] = df[TARGET].replace(-1, 1).fillna(0).astype(int)
    return df


def evaluate():
    device = get_device()
    print(f"Audit du SPÉCIALISTE CARDIO (Best Model) sur {device}...")

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 1)
    model.to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"Le fichier {MODEL_PATH} est introuvable.")
        return

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur chargement : {e}")
        return

    val_df = get_valid_df()
    print(f"Validation sur {len(val_df)} images (Frontal).")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    loader = DataLoader(CardioValidDataset(val_df, '..', transform), batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Calcul..."):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(labels.numpy().flatten())

    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Score AUC
    try:
        auc_score = roc_auc_score(all_targets, all_probs)
        print(f"\nScore AUC Global : {auc_score:.4f}")
    except:
        print("\n⚠Impossible de calculer l'AUC (peut-être une seule classe présente ?)")
        auc_score = 0

    # Recherche Seuil Optimal
    print("\nRecherche du seuil optimal...")
    best_threshold = 0.5
    best_j = -1

    for th in np.arange(0.1, 0.95, 0.05):
        y_pred = (all_probs > th).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_targets, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-6)
        spec = tn / (tn + fp + 1e-6)
        j_score = sens + spec - 1
        if j_score > best_j:
            best_j = j_score
            best_threshold = th

    print(f"🏆 Seuil Optimal trouvé : {best_threshold:.2f}")

    y_final_pred = (all_probs > best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_targets, y_final_pred, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"MATRICE DE CONFUSION (Seuil Optimal : {best_threshold:.2f})")
    print(f"{'=' * 60}\n")
    print(f"MALADIE : CARDIOMEGALY")
    print(f"{'-' * 30}")
    print(f"Vrais Positifs (Bien détecté) : {tp}")
    print(f"Faux Négatifs  (Raté !)       : {fn}")
    print(f"Faux Positifs  (Fausse alerte) : {fp}")
    print(f"Vrais Négatifs (Sain confirmé) : {tn}")
    print(f"{'-' * 30}")
    print(f"Précision Globale : {accuracy:.2%}")
    print(f"Sensibilité       : {sensitivity:.2%} (Priorité)")
    print(f"Spécificité       : {specificity:.2%}")
    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    evaluate()