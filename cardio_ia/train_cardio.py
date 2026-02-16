import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import roc_auc_score
from PIL import Image
from tqdm import tqdm

# script qui me permet d'entraîner le modèle pour la cardiomegalie

TARGET = 'Cardiomegaly'
DATA_DIR = "../data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VALID_CSV = os.path.join(DATA_DIR, "valid.csv")
CLEAN_TRAIN_CSV = os.path.join(DATA_DIR, "train_cardio_final.csv")

IMG_SIZE = 320
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_train_data():
    if os.path.exists(CLEAN_TRAIN_CSV): return pd.read_csv(CLEAN_TRAIN_CSV)

    print("Préparation TRAIN (Soft Labels)...")
    df = pd.read_csv(TRAIN_CSV)
    if 'Frontal/Lateral' in df.columns: df = df[df['Frontal/Lateral'] == 'Frontal']
    if 'Path' in df.columns: df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small', 'data', regex=False)

    # Soft Labels : 0.7 pour incertain
    df[TARGET] = df[TARGET].replace(-1, 0.7).fillna(0).astype('float32')
    df = df[['Path', TARGET]]
    df.to_csv(CLEAN_TRAIN_CSV, index=False)
    return df


def prepare_valid_data():
    df = pd.read_csv(VALID_CSV)
    if 'Frontal/Lateral' in df.columns: df = df[df['Frontal/Lateral'] == 'Frontal']
    if 'Path' in df.columns: df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small', 'data', regex=False)
    df[TARGET] = df[TARGET].replace(-1, 1).fillna(0).astype('float32')
    return df


class CardioDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['Path']
        label = torch.tensor([row[TARGET]], dtype=torch.float32)
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, label
        except:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), label

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    loop = tqdm(loader, desc="Training")
    for imgs, lbls in loop:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return epoch_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Validating"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(lbls.numpy())
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except:
        auc = 0.5
    return auc


if __name__ == "__main__":
    device = get_device()
    print(f"Démarrage CARDIOLOGUE (Sauvegarde complète) sur {device}")

    train_transform = transforms.Compose([
        transforms.Resize((340, 340)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomRotation(5),
        transforms.RandomAffine(0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_df = prepare_train_data()
    valid_df = prepare_valid_data()

    train_loader = DataLoader(CardioDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2)
    valid_loader = DataLoader(CardioDataset(valid_df, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = models.densenet121(weights='DEFAULT')
    model.classifier = nn.Linear(1024, 1)
    model.to(device)

    # Poids (x1.5 pour la sensibilité)
    n_pos = (train_df[TARGET] > 0).sum()
    pos_weight = (len(train_df) - n_pos) / (n_pos + 1e-5)

    final_weight_tensor = torch.tensor([pos_weight * 1.5], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=final_weight_tensor)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_auc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

        avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_auc = validate(model, valid_loader, device)
        scheduler.step()

        print(f"Train Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}")

        epoch_name = f"model_cardio_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), epoch_name)
        print(f"Sauvegardé : {epoch_name}")

        # 2. Mise à jour du meilleur modele
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "../../../best_cardio_model.pth")
            print(f"NOUVEAU RECORD ! (Copié dans 'best_cardio_model.pth')")

    print("\nTerminé.")