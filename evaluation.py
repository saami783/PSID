import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from PIL import Image

# --- CONFIGURATION ---
DATA_DIR = "data"
BATCH_SIZE = 16

# Liste des checkpoints (Assure-toi qu'ils existent)
CHECKPOINTS = [f"chexpert_densenet_epoch_{i}.pth" for i in range(1, 16)]

# --- MODIFICATION CRITIQUE : LES 5 CLASSES ---
TARGET_COLS = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion'
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CheXpertDenseNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=False):  # <--- 5 CLASSES
        super(CheXpertDenseNet, self).__init__()
        self.model = models.densenet121(weights=None)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x): return self.model(x)


class CheXpertDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.label_cols = TARGET_COLS  # Utilise les 5 colonnes

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = os.path.join(self.root_dir, row['Path'])
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            return torch.zeros(3, 320, 320), torch.zeros(len(self.label_cols))

        # Important : on ne charge que les valeurs des 5 colonnes cibles
        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        if self.transform: image = self.transform(image)
        return image, labels


def get_clean_valid_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']
    if 'Path' in df.columns:
        df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small', 'data', regex=False)

    # Nettoyage standard pour la validation
    df[TARGET_COLS] = df[TARGET_COLS].fillna(0)
    # Pour la validation (Ground Truth), on considère souvent Incertain comme Positif
    # pour ne pas pénaliser le modèle s'il détecte quelque chose d'ambigu.
    df[TARGET_COLS] = df[TARGET_COLS].replace(-1, 1)
    return df


def test_model(model_path, dataloader, device):
    print(f"--- Test : {model_path} ---")
    model = CheXpertDenseNet(num_classes=len(TARGET_COLS)).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"⚠️ Erreur chargement : {e}")
        return 0

    model.eval()
    all_targets, all_probs = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Calcul...", leave=False):
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    aucs = []
    print(f"{'Pathologie':<20} | AUC")
    print("-" * 30)
    for i, col in enumerate(TARGET_COLS):
        try:
            if np.unique(all_targets[:, i]).size > 1:
                score = roc_auc_score(all_targets[:, i], all_probs[:, i])
                aucs.append(score)
                print(f"{col:<20} | {score:.4f}")
        except:
            pass

    mean_auc = np.mean(aucs) if aucs else 0
    print(f"\n🏆 MOYENNE AUC : {mean_auc:.4f}\n")
    return mean_auc


if __name__ == '__main__':
    device = get_device()
    val_csv = os.path.join(DATA_DIR, "valid.csv")

    data_transforms = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_df = get_clean_valid_dataframe(val_csv)
    val_dataset = CheXpertDataset(val_df, root_dir='.', transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    best_score = 0
    best_model = ""

    for checkpoint in CHECKPOINTS:
        if os.path.exists(checkpoint):
            score = test_model(checkpoint, val_loader, device)
            if score > best_score:
                best_score = score
                best_model = checkpoint

    print(f"\n🥇 MEILLEUR : {best_model} (AUC: {best_score:.4f})")