import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = "data/"
# Chemins exacts de tes modèles
MODEL_CARDIO = "best_cardio_model.pth"
MODEL_LUNGS = "model_lungs_epoch_8.pth"

# Paramètres
BATCH_SIZE = 32
IMG_SIZE = 320
TH_CARDIO = 0.30
TH_LUNGS = 0.50


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()


# --- CHARGEMENT DONNÉES ---
def get_valid_df():
    # Chargement et nettoyage identiques à l'entraînement
    df = pd.read_csv(os.path.join(DATA_DIR, "valid.csv"))
    if 'Frontal/Lateral' in df.columns: df = df[df['Frontal/Lateral'] == 'Frontal']
    if 'Path' in df.columns: df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small', 'data', regex=False)

    # Nettoyage U-Ones pour validation stricte
    targets = ['Cardiomegaly', 'Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion']
    df[targets] = df[targets].replace(-1, 1).fillna(0)
    return df


class AuditDataset(Dataset):
    def __init__(self, df, root):
        self.df = df
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(os.path.join(self.root, row['Path'])).convert('RGB')
        except:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE))

        # On récupère les labels réels
        l_cardio = float(row['Cardiomegaly'])
        l_lungs = torch.tensor(row[['Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion']].values.astype(float),
                               dtype=torch.float32)

        return self.transform(img), l_cardio, l_lungs


# --- MOTEUR D'INFÉRENCE ---
def compute_predictions():
    print("🚀 Démarrage de l'inférence sur le jeu de validation...")

    # 1. Charger Modèles
    m_cardio = models.densenet121(weights=None)
    m_cardio.classifier = nn.Linear(1024, 1)
    if os.path.exists(MODEL_CARDIO):
        m_cardio.load_state_dict(torch.load(MODEL_CARDIO, map_location=device))
        m_cardio.to(device).eval()
        print("✅ Modèle Cardio chargé.")
    else:
        m_cardio = None

    m_lungs = models.densenet121(weights=None)
    m_lungs.classifier = nn.Linear(1024, 4)
    if os.path.exists(MODEL_LUNGS):
        m_lungs.load_state_dict(torch.load(MODEL_LUNGS, map_location=device))
        m_lungs.to(device).eval()
        print("✅ Modèle Poumons chargé.")
    else:
        m_lungs = None

    # 2. Dataset
    df = get_valid_df()
    loader = DataLoader(AuditDataset(df, ''), batch_size=BATCH_SIZE, shuffle=False)

    # 3. Boucle de calcul
    results = {
        'cardio_true': [], 'cardio_prob': [],
        'lungs_true': [], 'lungs_prob': []
    }

    with torch.no_grad():
        for img, lc, ll in tqdm(loader):
            img = img.to(device)

            # Cardio
            if m_cardio:
                out = torch.sigmoid(m_cardio(img)).cpu().numpy().flatten()
                results['cardio_prob'].extend(out)
                results['cardio_true'].extend(lc.numpy())

            # Lungs
            if m_lungs:
                out = torch.sigmoid(m_lungs(img)).cpu().numpy()
                results['lungs_prob'].append(out)
                results['lungs_true'].append(ll.numpy())

    return results


# --- VISUALISATION ---
def plot_results():
    data = compute_predictions()

    # === 1. CARDIO : ROC & CONFUSION ===
    if data['cardio_true']:
        y_true = np.array(data['cardio_true'])
        y_prob = np.array(data['cardio_prob'])

        # A. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='#d35400', lw=3, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.fill_between(fpr, tpr, alpha=0.1, color='#d35400')
        plt.title('Courbe ROC : Cardiomégalie', fontsize=14)
        plt.xlabel('Taux Faux Positifs')
        plt.ylabel('Taux Vrais Positifs (Sensibilité)')
        plt.legend(loc="lower right")
        plt.savefig('graph_dynamic_roc_cardio.png', dpi=300)

        # B. Confusion Matrix
        y_pred = (y_prob > TH_CARDIO).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False, annot_kws={"size": 16})
        plt.title(f'Matrice Confusion Cardio (Seuil {TH_CARDIO})')
        plt.xlabel('Prédiction')
        plt.ylabel('Réalité')
        plt.xticks([0.5, 1.5], ['Sain', 'Malade'])
        plt.yticks([0.5, 1.5], ['Sain', 'Malade'])
        plt.savefig('graph_dynamic_cm_cardio.png', dpi=300)
        print("✅ Graphiques Cardio générés.")

    # === 2. LUNGS : CONFUSION MATRICES ===
    if data['lungs_true']:
        y_true = np.concatenate(data['lungs_true'])
        y_prob = np.concatenate(data['lungs_prob'])
        y_pred = (y_prob > TH_LUNGS).astype(int)

        targets = ['Atélectasie', 'Condensation', 'Œdème', 'Épanchement']
        colors = ['Blues', 'Reds', 'Greens', 'Purples']

        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        for i, ax in enumerate(axes):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])

            # Calcul automatique des métriques pour le titre
            sens = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0

            sns.heatmap(cm, annot=True, fmt='d', cmap=colors[i], cbar=False, ax=ax, annot_kws={"size": 14})
            ax.set_title(f"{targets[i]}\nSensibilité: {sens:.1%}", fontweight='bold')
            ax.set_xlabel('Prédiction')
            if i == 0: ax.set_ylabel('Vérité Terrain')
            ax.set_xticklabels(['Sain', 'Malade'])
            ax.set_yticklabels(['Sain', 'Malade'])

        plt.suptitle(f"Matrices de Confusion : Pathologies Pulmonaires (Seuil {TH_LUNGS})", fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig('graph_dynamic_cm_lungs.png', dpi=300)
        print("✅ Graphiques Poumons générés.")


if __name__ == "__main__":
    plot_results()