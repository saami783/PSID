import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# --- CONFIGURATION ---
MODEL_LUNGS_PATH = "model_lungs_epoch_8.pth"
MODEL_CARDIO_PATH = "best_cardio_model.pth"

THRESHOLDS = {
    'Atelectasis': 0.45,
    'Consolidation': 0.50,
    'Edema': 0.45,
    'Pleural Effusion': 0.25,
    'Cardiomegaly': 0.30
}

LUNG_TARGETS = ['Atelectasis', 'Consolidation', 'Edema', 'Pleural Effusion']
CARDIO_TARGET = ['Cardiomegaly']


# --- SETUP TECHNIQUE ---
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


device = get_device()


def load_model(path, num_classes):
    if not os.path.exists(path): return None
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, num_classes)
    try:
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except:
        return None


lung_model = load_model(MODEL_LUNGS_PATH, 4)
cardio_model = load_model(MODEL_CARDIO_PATH, 1)

# --- CSS MODERNE & ÉPURÉ (Style SaaS) ---
modern_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;700&display=swap');

body { font-family: 'Plus Jakarta Sans', sans-serif !important; background-color: #f8fafc; }

/* Conteneur Global */
.gradio-container { max-width: 1000px !important; margin: 0 auto; }

/* Header */
.header-container { text-align: center; margin-bottom: 30px; padding: 20px 0; }
.header-title { font-size: 2.2rem; font-weight: 800; color: #0f172a; letter-spacing: -0.5px; }
.header-subtitle { color: #64748b; font-size: 1.1rem; margin-top: 5px; font-weight: 500; }

/* Carte de Résultat */
.result-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
    border: 1px solid #e2e8f0;
}

/* Lignes de Pathologie */
.pathology-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 0;
    border-bottom: 1px solid #f1f5f9;
}
.pathology-row:last-child { border-bottom: none; }

.pathology-info { display: flex; align-items: center; gap: 12px; }
.icon-box { 
    width: 40px; height: 40px; 
    border-radius: 10px; 
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
}
.pathology-name { font-weight: 600; color: #334155; font-size: 1rem; }
.threshold-info { font-size: 0.75rem; color: #94a3b8; }

/* Barres de progression */
.progress-container { width: 120px; height: 6px; background: #f1f5f9; border-radius: 10px; overflow: hidden; }
.progress-fill { height: 100%; border-radius: 10px; transition: width 0.5s ease; }

/* Badges */
.badge {
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.badge-safe { background: #dcfce7; color: #166534; }
.badge-danger { background: #fee2e2; color: #991b1b; }
.badge-neutral { background: #f1f5f9; color: #475569; }

/* Résumé Global */
.summary-banner {
    margin-top: 20px;
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-weight: 700;
    font-size: 1rem;
}
.summary-safe { background: #ecfdf5; color: #047857; border: 1px solid #d1fae5; }
.summary-danger { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }

/* Bouton */
button.primary-btn { 
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important; 
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.3) !important;
    transition: transform 0.1s;
}
button.primary-btn:hover { transform: translateY(-1px); }
"""


# --- LOGIQUE D'ANALYSE ---
def predict(image):
    if image is None: return "Veuillez charger une image."

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_t = transform(image).unsqueeze(0).to(device)
    results = {}

    if lung_model:
        with torch.no_grad():
            preds = torch.sigmoid(lung_model(img_t)).cpu().numpy()[0]
            for i, label in enumerate(LUNG_TARGETS): results[label] = preds[i]
    if cardio_model:
        with torch.no_grad():
            pred = torch.sigmoid(cardio_model(img_t)).cpu().numpy()[0][0]
            results[CARDIO_TARGET[0]] = pred

    # Génération HTML Épuré
    html = '<div class="result-card">'

    sorted_res = sorted(results.items(), key=lambda x: x[1], reverse=True)
    any_sick = False

    # Mapping des icônes pour le style
    icons = {
        'Cardiomegaly': '❤️',
        'Pleural Effusion': '💧',
        'Edema': '🌫️',
        'Consolidation': '🧱',
        'Atelectasis': '🫁'
    }

    for label, prob in sorted_res:
        th = THRESHOLDS.get(label, 0.5)
        is_sick = prob > th
        if is_sick: any_sick = True

        # Styles conditionnels
        badge_cls = "badge-danger" if is_sick else "badge-safe"
        badge_txt = "DÉTECTÉ" if is_sick else "NORMAL"
        bar_color = "#ef4444" if is_sick else "#22c55e"  # Rouge vs Vert vif
        icon = icons.get(label, '⚕️')
        bg_icon = "#fee2e2" if is_sick else "#dcfce7"  # Fond icone

        html += f"""
        <div class="pathology-row">
            <div class="pathology-info">
                <div class="icon-box" style="background: {bg_icon};">{icon}</div>
                <div>
                    <div class="pathology-name">{label}</div>
                    <div class="threshold-info">Seuil: {th * 100:.0f}%</div>
                </div>
            </div>

            <div style="display:flex; align-items:center; gap:15px;">
                <div style="text-align:right;">
                    <span style="font-weight:700; color:#334155;">{prob * 100:.1f}%</span>
                </div>
                <div class="progress-container">
                    <div class="progress-fill" style="width: {prob * 100}%; background: {bar_color};"></div>
                </div>
                <div class="badge {badge_cls}">{badge_txt}</div>
            </div>
        </div>
        """

    # Résumé final propre
    if any_sick:
        html += """
        <div class="summary-banner summary-danger">
            ⚠️ Anomalies détectées. Une vérification radiologique est nécessaire.
        </div>
        """
    else:
        html += """
        <div class="summary-banner summary-safe">
            ✅ Analyse Négative. Aucun signe pathologique détecté.
        </div>
        """

    html += '</div>'  # Fin card
    return html


# --- INTERFACE ---
with gr.Blocks(css=modern_css, title="MedAI Clean") as demo:
    # Header
    gr.HTML("""
        <div class="header-container">
            <div class="header-title">MedAI <span style="color:#2563eb;">Diagnostics</span></div>
            <div class="header-subtitle">Intelligence Artificielle d'Aide au Diagnostic Thoracique</div>
        </div>
    """)

    with gr.Row():
        # Colonne Gauche : Input
        with gr.Column(scale=4):
            input_image = gr.Image(
                type="pil",
                label="",
                height=400,
                show_label=False
            )
            analyze_btn = gr.Button("Lancer l'Analyse", elem_classes=["primary-btn"])

        # Colonne Droite : Output
        with gr.Column(scale=5):
            output = gr.HTML(label=None)

    analyze_btn.click(fn=predict, inputs=input_image, outputs=output)

if __name__ == "__main__":
    demo.launch()