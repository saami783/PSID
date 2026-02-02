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

# --- CONSTANTES ---
DATA_DIR = "data"
BATCH_SIZE = 16
# Liste des 5 modèles que tu as générés
CHECKPOINTS = [
    "chexpert_densenet_epoch_1.pth",
    "chexpert_densenet_epoch_2.pth",
    "chexpert_densenet_epoch_3.pth",
    "chexpert_densenet_epoch_4.pth",
    "chexpert_densenet_epoch_5.pth"
    "chexpert_densenet_epoch_6.pth",
    "chexpert_densenet_epoch_7.pth",
    "chexpert_densenet_epoch_8.pth",
    "chexpert_densenet_epoch_9.pth",
    "chexpert_densenet_epoch_10.pth"
    "chexpert_densenet_epoch_11.pth",
    "chexpert_densenet_epoch_12.pth",
    "chexpert_densenet_epoch_13.pth",
    "chexpert_densenet_epoch_14.pth",
    "chexpert_densenet_epoch_15.pth"
]

TARGET_COLS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]


# --- 1. FONCTIONS ET CLASSES (Identiques au Train) ---

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CheXpertDenseNet(nn.Module):
    def __init__(self, num_classes=14, pretrained=False):
        super(CheXpertDenseNet, self).__init__()
        # Note: pretrained=False ici car on va charger NOS poids à nous
        self.model = models.densenet121(weights=None)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)


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
            return torch.zeros(3, 224, 224), torch.zeros(len(self.label_cols))

        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        if self.transform:
            image = self.transform(image)
        return image, labels


def get_clean_valid_dataframe(csv_path):
    # Logique simplifiée pour la validation
    df = pd.read_csv(csv_path)
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']
    if 'Path' in df.columns:
        df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small', 'data', regex=False)

    df[TARGET_COLS] = df[TARGET_COLS].fillna(0)
    df[TARGET_COLS] = df[TARGET_COLS].replace(-1, 0)
    df[TARGET_COLS] = df[TARGET_COLS].astype(float)  # Sécurité
    return df


# --- 2. MOTEUR D'ÉVALUATION ---

def test_model(model_path, dataloader, device):
    print(f"--- Test du modèle : {model_path} ---")

    # 1. Charger l'architecture
    model = CheXpertDenseNet(num_classes=len(TARGET_COLS)).to(device)

    # 2. Charger les poids sauvegardés
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"⚠️ Erreur chargement {model_path}: {e}")
        return 0

    model.eval()  # Mode évaluation (très important)

    all_targets = []
    all_probs = []

    # 3. Boucle de prédiction (Sans gradient = Rapide)
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Calcul..."):
            images = images.to(device)

            # Prédiction
            outputs = model(images)
            # Sigmoid pour avoir des probabilités entre 0 et 1
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    # 4. Concaténation des résultats
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # 5. Calcul de l'AUC pour chaque maladie
    aucs = []
    print("\n📊 Résultats par pathologie (AUC) :")
    for i in range(len(TARGET_COLS)):
        try:
            # On ne calcule l'AUC que s'il y a au moins un cas positif dans la validation
            if np.unique(all_targets[:, i]).size > 1:
                score = roc_auc_score(all_targets[:, i], all_probs[:, i])
                aucs.append(score)
                # Affichage uniquement pour les pathologies principales
                if i < 14:
                    print(f"   - {TARGET_COLS[i]:<30} : {score:.4f}")
        except:
            pass

    mean_auc = np.mean(aucs)
    print(f"\n🏆 MOYENNE AUC pour ce modèle : {mean_auc:.4f}\n")
    print("-" * 30)
    return mean_auc


# --- 3. MAIN ---

if __name__ == '__main__':
    device = get_device()
    print(f"Evaluation sur : {device}")

    # Préparation Données Validation
    data_transforms = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_csv = os.path.join(DATA_DIR, "valid.csv")
    if not os.path.exists(val_csv):
        print("❌ Fichier valid.csv introuvable !")
        exit()

    val_df = get_clean_valid_dataframe(val_csv)
    print(f"Validation set : {len(val_df)} images")

    val_dataset = CheXpertDataset(val_df, root_dir='.', transform=data_transforms)
    # Important : shuffle=False pour l'évaluation
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Comparatif
    best_score = 0
    best_model = ""

    for checkpoint in CHECKPOINTS:
        if os.path.exists(checkpoint):
            score = test_model(checkpoint, val_loader, device)
            if score > best_score:
                best_score = score
                best_model = checkpoint
        else:
            print(f"Fichier {checkpoint} manquant, ignoré.")

    print(f"\n🥇 LE MEILLEUR MODÈLE EST : {best_model}")
    print(f"   Avec un score AUC moyen de : {best_score:.4f}")