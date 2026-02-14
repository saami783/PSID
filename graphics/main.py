import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.figsize': (12, 8)})

csv_path = '../data/train.csv'

labels = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

print("Chargement des données...")
try:
    df = pd.read_csv(csv_path)
    print(f"Données chargées : {len(df)} images.")
except FileNotFoundError:
    print(f"ERREUR : Le fichier '{csv_path}' est introuvable.")
    # On quitte ou on génère des données factices pour tester le code
    exit()

# =============================================================================
# GRAPHIQUE 1 : Distribution détaillée (Positif, Incertain, Négatif, Non Mentionné)
# =============================================================================
print("Génération du graphique 1 : Distribution détaillée...")

counts_detailed = []
for label in labels:
    pos = (df[label] == 1.0).sum()  # cas positifs
    unc = (df[label] == -1.0).sum()  # cas incertains
    neg = (df[label] == 0.0).sum()  # cas négatifs explicites (0)
    nan = df[label].isna().sum()  # NaN

    counts_detailed.append({
        'Label': label,
        'Positif (1.0)': pos,
        'Incertain (-1.0)': unc,
        'Négatif (0.0)': neg,
        'Non Mentionné (NaN)': nan
    })

df_counts = pd.DataFrame(counts_detailed).set_index('Label')

# Couleurs : Vert (Pos), Jaune (Inc), Rouge (Neg), Gris Clair (NaN)
colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#ecf0f1']

ax1 = df_counts.plot(kind='bar', stacked=True, figsize=(14, 8), color=colors, width=0.8, edgecolor='black',
                     linewidth=0.5)

plt.title('Distribution des étiquettes dans CheXpert', fontsize=18)
plt.ylabel('Nombre d\'images', fontsize=14)
plt.xlabel('Pathologie', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend(title='Type d\'étiquette', loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('chexpert_distribution_detaillee.png', dpi=300)
plt.show()

# =============================================================================
# GRAPHIQUE 2 : Nombre de cas Positifs par Pathologie
# =============================================================================
print("Génération du graphique 2 : Focus Positifs...")

pos_counts = df[labels].apply(lambda x: (x == 1.0).sum()).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
ax2 = sns.barplot(x=pos_counts.values, y=pos_counts.index, palette='Blues_r')

plt.title('Prévalence des pathologies (cas positifs uniquement)', fontsize=18)
plt.xlabel('Nombre d\'images positives', fontsize=14)
plt.ylabel('Pathologie', fontsize=14)

for i, v in enumerate(pos_counts.values):
    ax2.text(v + (max(pos_counts.values) * 0.01), i, f"{v:,}", color='black', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('chexpert_counts_positifs.png', dpi=300)
plt.show()

# =============================================================================
# GRAPHIQUE 3 : Multi-label Cardinalité (Pathologies par patient)
# =============================================================================
print("Génération du graphique 3 : Cardinalité...")

plt.figure(figsize=(10, 6))
labels_per_patient = (df[labels] == 1.0).sum(axis=1)

ax3 = sns.countplot(x=labels_per_patient, palette='viridis')

plt.title('Nombre de pathologies positives par radiographie', fontsize=18)
plt.xlabel('Nombre de pathologies détectées simultanément', fontsize=14)
plt.ylabel('Nombre de patients', fontsize=14)

for container in ax3.containers:
    ax3.bar_label(container)

plt.tight_layout()
plt.savefig('chexpert_cardinalite.png', dpi=300)
plt.show()

# =============================================================================
# GRAPHIQUE 4 : Matrice de Co-occurrence (Corrélation)
# =============================================================================
print("Génération du graphique 4 : Corrélation...")

df_binary = df[labels].replace(-1.0, 0).fillna(0)

plt.figure(figsize=(12, 10))
corr_matrix = df_binary.corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
            vmin=-0.1, vmax=0.6, square=True, linewidths=.5, cbar_kws={"shrink": .7})

plt.title('Matrice de corrélation entre pathologies', fontsize=18)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('chexpert_correlation.png', dpi=300)
plt.show()


# =============================================================================
# GRAPHIQUE 5 : GRAPHIQUE 2 : Démographie (Age et Sexe)
# =============================================================================
fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df['Age'], bins=30, kde=True, ax=ax2, color='skyblue')
ax2.set_title("Distribution de l'Âge des Patients")
ax2.set_xlabel('Âge')

sex_counts = df['Sex'].value_counts()

sex_counts_clean = sex_counts[sex_counts.index.isin(['Male', 'Female'])]

sex_counts_clean.plot.pie(
    autopct='%1.1f%%',
    ax=ax3,
    startangle=90,
    colors=['#66b3ff','#ff9999']
)

ax3.set_ylabel('')
ax3.set_title('Répartition par Sexe')

plt.tight_layout()
plt.savefig('chexpert_demographics.png')
plt.show()

# Configuration du style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Chargement (à adapter selon votre chemin)
try:
    df = pd.read_csv('../data/train.csv')
except:
    # Données factices pour l'exemple si fichier absent
    import numpy as np
    df = pd.DataFrame({
        'Frontal/Lateral': np.random.choice(['Frontal', 'Lateral', 'Unknown'], 1000, p=[0.8, 0.19, 0.01]),
        'AP/PA': np.random.choice(['AP', 'PA', 'LL', np.nan], 1000, p=[0.4, 0.4, 0.01, 0.19])
    })

# Création d'une figure avec 2 sous-graphiques côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# =============================================================================
# GRAPHIQUE 6 : Frontal vs Lateral
# =============================================================================
# 1. Filtrage strict : on ne garde que 'Frontal' et 'Lateral'
df_view = df[df['Frontal/Lateral'].isin(['Frontal', 'Lateral'])]

# 2. Graphique
sns.countplot(x='Frontal/Lateral', data=df_view, ax=ax1, palette='viridis', order=['Frontal', 'Lateral'])

# 3. Esthétique
ax1.set_title('Type de vue : Frontal vs Lateral', fontsize=14)
ax1.set_xlabel('')
ax1.set_ylabel("Nombre d'images")

# Ajout des chiffres sur les barres
for container in ax1.containers:
    ax1.bar_label(container, fontsize=11)


# =============================================================================
# GRAPHIQUE 7 : Projection AP vs PA
# =============================================================================
# Note : AP/PA ne concerne généralement que les vues Frontales.
# Les vues Latérales ont souvent un NaN ou rien ici.

# 1. Filtrage strict : on ne garde que 'AP' et 'PA'
# Cela élimine automatiquement les NaN, les LL, RL et les Lateral
df_proj = df[df['AP/PA'].isin(['AP', 'PA'])]

# 2. Graphique
sns.countplot(x='AP/PA', data=df_proj, ax=ax2, palette='mako', order=['AP', 'PA'])

# 3. Esthétique
ax2.set_title('Projection : Anteroposterior (AP) vs Posteroanterior (PA)', fontsize=14)
ax2.set_xlabel('')
ax2.set_ylabel('') # On enlève le label Y pour alléger car c'est la même échelle

# Ajout des chiffres sur les barres
for container in ax2.containers:
    ax2.bar_label(container, fontsize=11)

plt.tight_layout()
plt.savefig('chexpert_views_split.png', dpi=300)
plt.show()

print("Terminé. Les images ont été sauvegardées.")