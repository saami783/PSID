import pandas as pd
import os

# Charge le fichier de validation
df = pd.read_csv("data/valid.csv")

# Le patient que tu as testé
image_path = "patient64632/study1/view1_frontal.jpg"

# On cherche la ligne correspondante dans le CSV
# (On cherche une correspondance partielle car le CSV contient "CheXpert-v1.0-small/...")
row = df[df['Path'].str.contains(image_path)]

if not row.empty:
    print(f"--- VÉRITÉ TERRAIN (Radiologue) pour {image_path} ---")

    # On affiche les colonnes qui valent 1.0 (Présent)
    target_cols = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
    ]

    found_something = False
    for col in target_cols:
        val = row.iloc[0][col]
        if val == 1.0:
            print(f"✅ {col} : PRÉSENT (1.0)")
            found_something = True
        elif val == -1.0:
            print(f"❓ {col} : INCERTAIN (-1.0)")

    if not found_something:
        print("Rien de signalé (Tout est à 0 ou vide)")
else:
    print("Patient non trouvé dans le CSV.")