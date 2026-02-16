import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# script qui me permet d'entraîner le modèle pour les 4 pathologies pulmonaires

TARGETS = ['Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion']
DATA_DIR = "../data"
ORIGINAL_CSV = os.path.join(DATA_DIR, "train.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "train_lungs.csv")
IMG_SIZE = 320
BATCH_SIZE = 32
NUM_EPOCHS = 8


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_data():
    if os.path.exists(CLEAN_CSV):
        print(f"Chargement de {CLEAN_CSV}...")
        return pd.read_csv(CLEAN_CSV)

    print("Préparation dataset POUMONS (Filtre Frontal + U-Ones)...")
    df = pd.read_csv(ORIGINAL_CSV)

    # Filtre Frontal uniquement
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']

    # Correction des chemins
    if 'Path' in df.columns:
        df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small', 'data', regex=False)

    # Stratégie U-Ones (-1 -> 1) : On considère l'incertain comme malade
    # C'est la stratégie recommandée par le papier CheXpert pour Atelectasis et Edema
    df[TARGETS] = df[TARGETS].replace(-1, 1).fillna(0).astype('float32')

    # On ne garde que l'essentiel
    df = df[['Path'] + TARGETS]
    df.to_csv(CLEAN_CSV, index=False)
    return df


class LungDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.labels = TARGETS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['Path']
        labels = torch.tensor(row[self.labels].values.astype('float32'))

        try:
            img = Image.open(image_path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, labels
        except:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), labels


def train():
    device = get_device()
    print(f"Entraînement PNEUMOLOGUE (4 classes) sur {device}")

    # Data Augmentation "LIBÉRALE" (Les textures supportent la rotation)
    # Rotation de 15° autorisée
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    df = prepare_data()
    print(f"Images d'entraînement : {len(df)}")

    loader = DataLoader(LungDataset(df, transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Modèle Multi-Label (4 sorties)
    model = models.densenet121(weights='DEFAULT')
    model.classifier = nn.Linear(1024, len(TARGETS))
    model.to(device)

    # Calcul des Poids pour la Loss (Équilibrage)
    print("⚖Calcul des poids...")
    pos_counts = df[TARGETS].sum()
    pos_weights = (len(df) - pos_counts) / (pos_counts + 1e-5)
    print(f"Poids: {pos_weights.values}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights.values).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1)

    for epoch in range(NUM_EPOCHS):
        model.train()
        loop = tqdm(loader, desc=f"Ep {epoch + 1}/{NUM_EPOCHS}")
        epoch_loss = 0

        for imgs, lbls in loop:
            imgs, lbls = imgs.to(device), lbls.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Sauvegarde à chaque époque
        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        save_name = f"model_lungs_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), save_name)
        print(f"Sauvegardé : {save_name} (Loss: {avg_loss:.4f})")


if __name__ == "__main__":
    train()