import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from PIL import Image

MODEL_PATH = "../models/tentative 5/model_lungs/model_lungs_epoch_8.pth"
DATA_DIR = "../data"
BATCH_SIZE = 32
IMG_SIZE = 320
THRESHOLD = 0.5

TARGETS = ['Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion']


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ValidationDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.labels = TARGETS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.root_dir, row['Path'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.zeros(len(self.labels))

        label_vals = row[self.labels].values.astype(float)
        labels = torch.tensor(label_vals, dtype=torch.float32)

        if self.transform: image = self.transform(image)
        return image, labels


def get_valid_df():
    csv_path = os.path.join(DATA_DIR, "valid.csv")
    if not os.path.exists(csv_path):
        print("valid.csv introuvable.")
        exit()

    df = pd.read_csv(csv_path)
    if 'Frontal/Lateral' in df.columns: df = df[df['Frontal/Lateral'] == 'Frontal']
    if 'Path' in df.columns: df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small', 'data', regex=False)

    # Alignement avec la stratégie U-Ones
    df[TARGETS] = df[TARGETS].replace(-1, 1).fillna(0)
    return df


def evaluate():
    device = get_device()
    print(f"Audit du PNEUMOLOGUE (Epoch 8) sur {device}...")

    # 1. DÉFINITION EXACTE COMME L'ENTRAÎNEMENT
    # Pas de classe "LungSpecialist" wrapper pour éviter les erreurs de clés
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, len(TARGETS))
    model.to(device)

    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur chargement modèle : {e}")
        return

    val_df = get_valid_df()
    print(f"Validation sur {len(val_df)} images.")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    loader = DataLoader(ValidationDataset(val_df, '..', transform), batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_targets = []

    # 4. Inférence
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Calcul..."):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            preds = (probs > THRESHOLD).float()

            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # 5. Affichage
    print(f"\n{'=' * 60}")
    print(f"MATRICES DE CONFUSION (Seuil: {THRESHOLD * 100}%)")
    print(f"{'=' * 60}\n")

    for i, disease in enumerate(TARGETS):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        print(f"MALADIE : {disease.upper()}")
        print(f"{'-' * 30}")
        print(f"Vrais Positifs (Bien détecté) : {tp}")
        print(f"Faux Négatifs  (Raté !)       : {fn}")
        print(f"Faux Positifs  (Fausse alerte) : {fp}")
        print(f"Vrais Négatifs (Sain confirmé) : {tn}")
        print(f"{'-' * 30}")
        print(f"Précision Globale : {accuracy:.2%}")
        print(f"Sensibilité       : {sensitivity:.2%} (Ne pas rater !)")
        print(f" Spécificité       : {specificity:.2%}")
        print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    evaluate()