import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
# Mets le chemin de ton modèle Epoque 15 ici
MODEL_PATH = "models/tentative 4/chexpert_densenet_epoch_5.pth"
# Image à tester
IMAGE_PATH = "data/valid/patient64632/study1/view1_frontal.jpg"

# --- MODIFICATION CRITIQUE : SEULEMENT LES 5 CLASSES OFFICIELLES ---
LABELS = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Pleural Effusion'
]

# --- 1. DÉFINITION DU MODÈLE ---
class CheXpertDenseNet(nn.Module):
    def __init__(self, num_classes=5): # <--- CHANGÉ À 5
        super(CheXpertDenseNet, self).__init__()
        self.model = models.densenet121(weights=None)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)

def predict(image_path, model_path):
    # 1. Configuration Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Chargement du modèle sur {device}...")

    # 2. Charger le modèle (Dimension 5)
    model = CheXpertDenseNet(num_classes=len(LABELS)).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    except FileNotFoundError:
        print(f"❌ Erreur: Le fichier {model_path} est introuvable.")
        return
    except RuntimeError as e:
        print(f"❌ Erreur de dimension : {e}")
        print("Vérifie que tu utilises bien un modèle entraîné sur 5 classes.")
        return

    # 3. Préparer l'image
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Erreur image : {e}")
        return

    # 4. Prédiction
    print(f"Analyse de l'image : {image_path} ...")
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

    # 5. Affichage
    print("\n--- RÉSULTATS DU DIAGNOSTIC (CheXpert 5) ---")
    results = list(zip(LABELS, probabilities))
    results.sort(key=lambda x: x[1], reverse=True)

    for label, prob in results:
        # Rouge si > 50%, Vert sinon
        color = "\033[91m" if prob > 0.5 else "\033[92m"
        reset = "\033[0m"
        print(f"{color}{label:<20} : {prob * 100:.2f}%{reset}")

if __name__ == '__main__':
    predict(IMAGE_PATH, MODEL_PATH)