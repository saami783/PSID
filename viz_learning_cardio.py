import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Données extraites de tes screenshots Cardio 1 & 2
epochs = np.arange(1, 11)
# Source: train cardio modele 1.jpg et 2.jpg
train_loss = [1.0165, 0.9685, 0.9433, 0.9189, 0.8871, 0.8438, 0.7833, 0.7108, 0.6459, 0.6030]
val_auc = [0.8314, 0.7988, 0.8236, 0.8405, 0.7605, 0.7998, 0.7742, 0.7458, 0.7355, 0.7443]


def plot_cardio_curve():
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- Axe Gauche : Perte (Train Loss) ---
    color_loss = '#e74c3c'  # Rouge
    ax1.set_xlabel('Époques', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Perte d\'entraînement (Loss)', color=color_loss, fontsize=12, fontweight='bold')
    ax1.plot(epochs, train_loss, marker='o', linestyle='-', color=color_loss, label='Train Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_loss)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Axe Droit : Performance (Val AUC) ---
    ax2 = ax1.twinx()
    color_auc = '#2ecc71'  # Vert
    ax2.set_ylabel('Score Validation (AUC)', color=color_auc, fontsize=12, fontweight='bold')
    ax2.plot(epochs, val_auc, marker='s', linestyle='--', color=color_auc, label='Val AUC', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color_auc)
    ax2.set_ylim(0.70, 0.86)  # Zoom pour bien voir les variations

    # --- Annotations Clés ---
    # 1. Le Pic (Best Model)
    best_epoch = 4
    best_val = 0.8405
    plt.axvline(x=best_epoch, color='black', linestyle=':', alpha=0.6)

    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9)
    ax2.annotate(f'BEST MODEL\nAUC = {best_val}',
                 xy=(best_epoch, best_val), xytext=(best_epoch, 0.855),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 ha='center', fontsize=10, fontweight='bold', bbox=bbox_props)

    # 2. Zone de Sur-apprentissage
    ax1.axvspan(5, 10, color='grey', alpha=0.1)
    ax1.text(7.5, 0.95, "Zone de Sur-apprentissage\n(Perte ↓ mais AUC ↘)",
             ha='center', color='grey', fontweight='bold')

    plt.title("Analyse d'Entraînement Cardio : Détection du point optimal", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('courbe_cardio_finale.png', dpi=300)
    print("✅ Généré : courbe_cardio_finale.png")


if __name__ == "__main__":
    plot_cardio_curve()