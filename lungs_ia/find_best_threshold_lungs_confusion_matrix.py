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
        labels = torch.tensor(row[self.labels].values.astype(float), dtype=torch.float32)
        if self.transform: image = self.transform(image)
        return image, labels


def get_valid_df():
    df = pd.read_csv(os.path.join(DATA_DIR, "valid.csv"))
    if 'Frontal/Lateral' in df.columns: df = df[df['Frontal/Lateral'] == 'Frontal']
    if 'Path' in df.columns: df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small', 'data', regex=False)
    df[TARGETS] = df[TARGETS].replace(-1, 1).fillna(0)
    return df


def find_thresholds():
    device = get_device()
    print(f"🎚️ Optimisation des Seuils sur {device}...")

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, len(TARGETS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    val_df = get_valid_df()
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    loader = DataLoader(ValidationDataset(val_df, '..', transform), batch_size=BATCH_SIZE, shuffle=False)

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Calcul des probabilités"):
            probs = torch.sigmoid(model(images.to(device))).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    print(f"\n{'=' * 70}")
    print(f"MEILLEURS SEUILS RECOMMANDÉS (Objectif: Sensibilité > 90%)")
    print(f"{'=' * 70}")

    # On teste plein de seuils
    thresholds_to_test = np.arange(0.1, 0.95, 0.05)

    for i, disease in enumerate(TARGETS):
        print(f"\n{disease.upper()}")
        print(f"{'-' * 30}")
        print(f"{'Seuil':<10} | {'Sensib.':<10} | {'Spécif.':<10} | {'Précision':<10}")

        best_th = 0.5
        best_score = 0  # On cherche le meilleur Index de Youden (Sens + Spec - 1)

        for th in thresholds_to_test:
            y_true = all_targets[:, i]
            y_pred = (all_probs[:, i] > th).astype(float)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            acc = (tp + tn) / (tp + tn + fp + fn)

            # Score de Youden
            youden = sens + spec - 1
            if youden > best_score:
                best_score = youden
                best_th = th

            # On affiche les lignes intéressantes (autour de 0.5 ou haute sensibilité)
            if abs(th - 0.5) < 0.01 or (sens > 0.90 and sens < 0.96):
                print(f"{th:.2f}       | {sens:.2%}     | {spec:.2%}     | {acc:.2%}")

        print(f"{'-' * 30}")
        print(f"Seuil Optimal (Youden) : {best_th:.2f}")


if __name__ == "__main__":
    find_thresholds()