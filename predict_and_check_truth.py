import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import sys

MODEL_LUNGS_PATH = "models/tentative 5/model_lungs/model_lungs_epoch_8.pth"
MODEL_CARDIO_PATH = "best_cardio_model.pth"
VALID_CSV = "data/valid.csv"

DEFAULT_IMAGE = "data/valid/patient64545/study1/view1_frontal.jpg"
# DEFAULT_IMAGE = "data/valid/patient64546/study1/view1_frontal.jpg"
# DEFAULT_IMAGE = "data/valid/patient64548/study1/view1_frontal.jpg"


# DEFAULT_IMAGE = "data/valid/patient64552/study1/view1_frontal.jpg"
# DEFAULT_IMAGE = "data/valid/patient64555/study1/view1_frontal.jpg"

# DEFAULT_IMAGE = "data/valid/patient64543/study1/view1_frontal.jpg"
# DEFAULT_IMAGE = "data/valid/patient64541/study1/view1_frontal.jpg"

# Tes Seuils calibrés
THRESHOLDS = {
    'Atelectasis': 0.45,
    'Consolidation': 0.50,
    'Edema': 0.45,
    'Pleural Effusion': 0.25,
    'Cardiomegaly': 0.30
}

LUNG_TARGETS = ['Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion']
CARDIO_TARGET = ['Cardiomegaly']
ALL_TARGETS = LUNG_TARGETS + CARDIO_TARGET


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(path, num_classes, device):
    if not os.path.exists(path): return None
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, num_classes)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except:
        return None


def get_ground_truth(image_path):
    """Cherche la vérité terrain dans le CSV"""
    if not os.path.exists(VALID_CSV):
        print("Fichier valid.csv introuvable.")
        return None

    df = pd.read_csv(VALID_CSV)

    # On nettoie les chemins pour être sûr que ça matche
    # L'image path est souvent "data/valid/..."
    # Le CSV est souvent "CheXpert-v1.0-small/valid/..."
    # On va comparer uniquement la fin : "patientXXXXX/studyX/viewX_frontal.jpg"

    suffix = "/".join(image_path.split("/")[-3:])

    row = df[df['Path'].str.contains(suffix, regex=False)]

    if row.empty:
        print(f"⚠Image non trouvée dans le CSV : {suffix}")
        return None

    return row.iloc[0]


def check_truth(image_path):
    device = get_device()
    print(f"\n--- VÉRIFICATION VÉRITÉ TERRAIN ---")
    print(f"Patient : {image_path.split('/')[-3]}")

    # 1. Récupérer la Vérité
    truth_row = get_ground_truth(image_path)
    if truth_row is None: return

    # 2. Faire la Prédiction (IA)
    lung_model = load_model(MODEL_LUNGS_PATH, 4, device)
    cardio_model = load_model(MODEL_CARDIO_PATH, 1, device)

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
    except:
        print("Erreur image.")
        return

    predictions = {}

    # Inférence
    if lung_model:
        with torch.no_grad():
            preds = torch.sigmoid(lung_model(img_t)).cpu().numpy()[0]
            for i, label in enumerate(LUNG_TARGETS): predictions[label] = preds[i]

    if cardio_model:
        with torch.no_grad():
            pred = torch.sigmoid(cardio_model(img_t)).cpu().numpy()[0][0]
            predictions[CARDIO_TARGET[0]] = pred

    # 3. Comparaison et Affichage
    print(f"\nTABLEAU COMPARATIF")
    print("-" * 85)
    print(f"{'PATHOLOGIE':<20} | {'IA (Proba)':<12} | {'IA (Diag)':<12} | {'MÉDECIN':<12} | {'VERDICT'}")
    print("-" * 85)

    correct_count = 0

    for label in ALL_TARGETS:

        prob = predictions.get(label, 0.0)
        th = THRESHOLDS.get(label, 0.5)
        ia_sick = prob > th

        # Données Médecin (CSV)
        # Dans valid.csv, 1.0 = Malade, 0.0 = Sain. (Parfois -1, on le traite comme incertain/malade)
        real_val = truth_row[label]
        doctor_sick = (real_val == 1.0)

        # Formatage affichage
        ia_text = "DETECTÉ" if ia_sick else "Négatif"
        doc_text = "MALADE" if doctor_sick else "Sain"

        # Verdict (Comparaison)
        if ia_sick == doctor_sick:
            verdict = "CORRECT"
            color = "\033[92m"  # Vert
            correct_count += 1
        else:
            if ia_sick and not doctor_sick:
                verdict = "Faux Positif"  # L'IA s'inquiète pour rien
                color = "\033[93m"  # Jaune
            else:
                verdict = "RATÉ !"  # L'IA a raté une maladie
                color = "\033[91m"  # Rouge

        reset = "\033[0m"

        print(f"{color}{label:<20} | {prob * 100:5.1f}%      | {ia_text:<12} | {doc_text:<12} | {verdict}{reset}")

    print("-" * 85)
    score = (correct_count / 5) * 100
    print(f"🎯 Score sur ce patient : {correct_count}/5 ({score:.0f}%)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_path = sys.argv[1]
        check_truth(user_path)
    else:
        check_truth(DEFAULT_IMAGE)