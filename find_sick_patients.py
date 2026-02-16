import pandas as pd
import os

# ce script permet de trouver des exemples de patients malades dans le dataset de validation
CSV_PATH = "data/valid.csv"
TARGETS = ['Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion', 'Cardiomegaly']


def find_sick():
    if not os.path.exists(CSV_PATH):
        print("Fichier csv introuvable.")
        return

    print("🔍 Recherche de patients malades dans le dataset de validation...\n")

    df = pd.read_csv(CSV_PATH)

    # On ne veut que les vues Frontales (car nos modèles sont entraînés dessus)
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']

    # Correction du chemin pour qu'il matche ton dossier local
    if 'Path' in df.columns:
        df['Path'] = df['Path'].str.replace('CheXpert-v1.0-small', 'data', regex=False)

    # Pour chaque maladie, on cherche 3 exemples positifs (1.0)
    for disease in TARGETS:
        print(f"EXEMPLES POUR : {disease.upper()}")
        print("-" * 50)

        # On cherche ceux qui ont 1.0 (Malade confirmé)
        sick_patients = df[df[disease] == 1.0]

        if len(sick_patients) == 0:
            print("Aucun cas trouvé dans valid.csv (bizarre...)")
        else:
            # On prend les 3 premiers
            for i in range(min(3, len(sick_patients))):
                path = sick_patients.iloc[i]['Path']
                print(f"python check_truth.py {path}")

        print("\n")


if __name__ == "__main__":
    find_sick()