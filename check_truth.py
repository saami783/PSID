import pandas as pd
import os

# CONFIGURATION
IMAGE_PATH = "patient64632/study1/view1_frontal.jpg"  # L'image que tu testes
CSV_PATH = "data/valid.csv"

# Les 5 cibles de ton modèle
MODEL_TARGETS = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']


def check_truth():
    if not os.path.exists(CSV_PATH):
        print("Fichier valid.csv introuvable.")
        return

    df = pd.read_csv(CSV_PATH)
    # Recherche partielle dans le chemin
    row = df[df['Path'].str.contains(IMAGE_PATH, regex=False)]

    if row.empty:
        print(f"❌ Patient {IMAGE_PATH} non trouvé dans le CSV de validation.")
        return

    print(f"--- VÉRITÉ TERRAIN (Radiologue) ---")
    print(f"Patient : {IMAGE_PATH}")

    # On regarde toutes les colonnes possibles
    all_cols = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]

    found = False
    for col in all_cols:
        val = row.iloc[0][col]

        # Est-ce une maladie ciblée par ton IA ?
        is_target = "🎯" if col in MODEL_TARGETS else "  "

        if val == 1.0:
            print(f"{is_target} ✅ {col:<25} : PRÉSENT")
            found = True
        elif val == -1.0:
            print(f"{is_target} ❓ {col:<25} : INCERTAIN")
            found = True

    if not found:
        print("Rien de signalé (Patient sain ou données manquantes).")


if __name__ == "__main__":
    check_truth()