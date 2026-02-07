import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# --- CONSTANTES ---
DATA_DIR = "data"
ORIGINAL_CSV = os.path.join(DATA_DIR, "train.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "train_clean.csv")

IMG_SIZE = 320
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-5

# MODIFICATION 1 : Restriction aux 5 pathologies officielles "CheXpert 5"
# Les papiers confirment que c'est le standard de compétition pour l'évaluation.
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


def prepare_data(original_path, clean_path):
    # On force la régénération pour être sûr d'avoir la stratégie U-Ones et les bonnes colonnes
    if os.path.exists(clean_path):
        # On vérifie si le fichier existant a les bonnes colonnes
        df_check = pd.read_csv(clean_path, nrows=1)
        if all(col in df_check.columns for col in TARGET_COLS) and len(df_check.columns) < 20:
            return pd.read_csv(clean_path)

    print(f"🧹 Régénération du dataset (U-Ones + CheXpert 5)...")
    df = pd.read_csv(original_path)
    if 'Frontal/Lateral' in df.columns: df = df[df['Frontal/Lateral'] == 'Frontal']
    if 'Path' in df.columns: df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small', 'data', regex=False)

    # -1 devient 1 (Stratégie U-Ones)
    # Efficace pour Atelectasis et Edema selon l'analyse des papiers [cite: 1214, 1216]
    df[TARGET_COLS] = df[TARGET_COLS].replace(-1, 1)
    df[TARGET_COLS] = df[TARGET_COLS].fillna(0)
    for col in TARGET_COLS: df[col] = df[col].astype('float32')

    # On ne sauvegarde que les colonnes nécessaires pour alléger
    cols_to_keep = ['Path'] + TARGET_COLS
    df = df[cols_to_keep]

    df.to_csv(clean_path, index=False)
    return df


# --- 2. DATASET ---
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
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), torch.zeros(len(self.label_cols))

        labels = torch.tensor(row[self.label_cols].values.astype(np.float32))

        if self.transform:
            image = self.transform(image)

        return image, labels


class CheXpertDenseNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):  # Modifié pour 5 classes
        super(CheXpertDenseNet, self).__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x): return self.model(x)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(dataloader, desc="Training")

    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        if torch.isnan(loss):
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())

    return running_loss / len(dataloader.dataset)


if __name__ == '__main__':
    device = get_device()
    print(f"\n🚀 Démarrage OPTIMISÉ (CheXpert 5 + Weighted Loss)")
    print(f"Device: {device} | Batch: {BATCH_SIZE} | Res: {IMG_SIZE}")

    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_df = prepare_data(ORIGINAL_CSV, CLEAN_CSV)
        print(f"Dataset : {len(train_df)} images.")
        print(f"Cibles : {TARGET_COLS}")

        train_dataset = CheXpertDataset(train_df, root_dir='.', transform=train_transforms)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    except Exception as e:
        print(f"Erreur : {e}")
        exit()

    print("Initialisation du modèle...")
    model = CheXpertDenseNet(num_classes=len(TARGET_COLS)).to(device)

    # MODIFICATION 2 : Calcul des poids positifs (pos_weight)
    # Cela force le modèle à se concentrer sur les cas rares (malades) plutôt que les sains
    print("⚖️ Calcul des poids pour équilibrer les classes...")
    pos_counts = train_df[TARGET_COLS].sum()
    neg_counts = len(train_df) - pos_counts
    # Formule : Poids = (Nombre Négatifs) / (Nombre Positifs)
    # Si une maladie est rare (beaucoup de négatifs), son poids sera élevé.
    pos_weights_value = neg_counts / (pos_counts + 1e-5)
    pos_weights_tensor = torch.tensor(pos_weights_value.values, dtype=torch.float32).to(device)

    print("Poids calculés :")
    for i, col in enumerate(TARGET_COLS):
        print(f"  - {col}: x{pos_weights_tensor[i].item():.2f}")

    # Injection des poids dans la Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    print(f"\n🏁 C'est parti pour {NUM_EPOCHS} époques...")

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Époque {epoch + 1}/{NUM_EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        if np.isnan(train_loss):
            print("❌ NaN détecté sur la moyenne. Arrêt.")
            break

        print(f"Train Loss: {train_loss:.4f}")
        scheduler.step(train_loss)

        torch.save(model.state_dict(), f"chexpert_densenet_epoch_{epoch + 1}.pth")
        print(f"💾 Sauvegardé.")

    print("\n✅ Terminé.")