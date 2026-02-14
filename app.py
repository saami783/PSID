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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

body { 
    font-family: 'Plus Jakarta Sans', sans-serif !important; 
    background: radial-gradient(circle at 20% 20%, #111827, #0b1224 55%, #0b1224 100%);
    color: #e2e8f0;
}
.gradio-container { max-width: 1180px !important; margin: 0 auto; padding: 28px 16px 38px; }

/* Hero */
.hero-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(79,70,229,0.22));
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 20px;
    padding: 22px 22px 18px;
    margin-bottom: 18px;
    display: grid;
    grid-template-columns: 2fr 1.3fr;
    gap: 18px;
    box-shadow: 0 25px 70px -40px rgba(59,130,246,0.9);
}
.eyebrow { letter-spacing: 0.12em; text-transform: uppercase; color: #bfdbfe; font-weight: 700; font-size: 0.85rem; margin-bottom: 6px; }
.hero-title { font-size: 2.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px; margin: 0 0 8px 0; }
.hero-lead { color: #e2e8f0; opacity: 0.9; line-height: 1.55; }
.pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.pill { padding: 8px 12px; border-radius: 999px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.08); color: #dbeafe; font-size: 0.85rem; }
.hero-steps { display: grid; gap: 10px; }
.hero-step { background: rgba(15,23,42,0.5); border: 1px solid rgba(148,163,184,0.2); border-radius: 12px; padding: 12px 14px; color: #e2e8f0; font-weight: 600; display: flex; gap: 10px; align-items: center; }
.hero-step span { color: #93c5fd; font-weight: 700; }

/* Panels */
.panel { 
    background: rgba(15,23,42,0.65); 
    border: 1px solid rgba(148,163,184,0.2); 
    border-radius: 18px; 
    padding: 18px; 
    box-shadow: 0 20px 60px -45px rgba(0,0,0,0.8);
}
.section-title { font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px; }
.muted { color: #94a3b8; font-size: 0.95rem; line-height: 1.5; }
.content-row { gap: 16px; }

/* Upload zone */
.upload-zone {
    border: 1.5px dashed #334155;
    background: rgba(17,24,39,0.65);
    border-radius: 16px;
    min-height: 340px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
    overflow: hidden;
}
.upload-zone:hover { border-color: #60a5fa; box-shadow: 0 15px 60px -40px #3b82f6; }
.upload-zone img { object-fit: contain; }
.drag-note { margin-top: 10px; color: #cbd5e1; background: rgba(59,130,246,0.08); border: 1px solid rgba(59,130,246,0.2); border-radius: 12px; padding: 12px 14px; line-height: 1.5; }

/* Résultats */
.result-wrapper { margin-top: 4px; }
.result-card {
    background: #0f172a;
    border-radius: 16px;
    padding: 20px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.02), 0 10px 40px -30px rgba(0,0,0,0.9);
    border: 1px solid rgba(148,163,184,0.25);
}
.result-placeholder { color: #94a3b8; text-align: center; padding: 16px 10px; line-height: 1.6; }

/* Lignes de Pathologie */
.pathology-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 0;
    border-bottom: 1px solid #1f2937;
}
.pathology-row:last-child { border-bottom: none; }

.pathology-info { display: flex; align-items: center; gap: 12px; }
.icon-box { 
    width: 40px; height: 40px; 
    border-radius: 10px; 
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.pathology-name { font-weight: 700; color: #f8fafc; font-size: 1rem; letter-spacing: -0.01em; }
.threshold-info { font-size: 0.75rem; color: #94a3b8; }

/* Barres de progression */
.progress-container { width: 120px; height: 6px; background: #1f2937; border-radius: 10px; overflow: hidden; }
.progress-fill { height: 100%; border-radius: 10px; transition: width 0.5s ease; }

/* Badges */
.badge {
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.badge-safe { background: #14532d33; color: #22c55e; border: 1px solid #14532d66; }
.badge-danger { background: #7f1d1d33; color: #fca5a5; border: 1px solid #7f1d1d66; }
.badge-neutral { background: #111827; color: #cbd5e1; border: 1px solid #1f2937; }

/* Résumé Global */
.summary-banner {
    margin-top: 14px;
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    font-weight: 700;
    font-size: 0.95rem;
}
.summary-safe { background: #14532d40; color: #86efac; border: 1px solid #14532d66; }
.summary-danger { background: #7f1d1d33; color: #fecaca; border: 1px solid #7f1d1d66; }

/* Bouton */
button.primary-btn { 
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important; 
    border: none !important;
    box-shadow: 0 8px 20px -10px rgba(37, 99, 235, 0.8) !important;
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
                    <span style="font-weight:700; color:#e2e8f0;">{prob * 100:.1f}%</span>
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
    gr.HTML("""
        <div class="hero-card">
            <div>
                <div class="eyebrow">Analyse thoracique assistée</div>
                <div class="hero-title">MedAI Diagnostics</div>
                <p class="hero-lead">Glissez-déposez une radiographie (ou cliquez pour importer) et obtenez une lecture automatique des pathologies pulmonaires et cardiaques. Le drag & drop préserve la qualité native et accélère la prise en charge.</p>
                <div class="pill-row">
                    <div class="pill">Prétraitement optimisé pour radios</div>
                    <div class="pill">Score transparent par pathologie</div>
                    <div class="pill">Résultat en quelques secondes</div>
                </div>
            </div>
            <div class="hero-steps">
                <div class="hero-step"><span>1</span> Déposez la radio directement dans la zone d'upload (drag & drop).</div>
                <div class="hero-step"><span>2</span> Le modèle conserve la résolution et les contrastes sans compression superflue.</div>
                <div class="hero-step"><span>3</span> Les probabilités par pathologie apparaissent instantanément.</div>
            </div>
        </div>
    """)

    with gr.Row(elem_classes=["content-row"]):
        with gr.Column(scale=5):
            gr.HTML("""
                <div class="panel">
                    <div class="section-title">Charger une radiographie</div>
                    <p class="muted">Le glisser-déposer est recommandé : il évite les manipulations intermédiaires, réduit le risque de compression et garde les métadonnées intactes pour une lecture fidèle.</p>
                </div>
            """)
            input_image = gr.Image(
                type="pil",
                label="",
                height=380,
                show_label=False,
                elem_classes=["upload-zone"]
            )
            analyze_btn = gr.Button("Analyser la radio", elem_classes=["primary-btn"])
            gr.HTML("""
                <div class="drag-note">
                    <strong>Pourquoi utiliser le drag & drop ?</strong><br/>
                    - Import direct depuis le PACS ou votre explorateur, sans conversion.<br/>
                    - Préserve la finesse des contrastes, cruciale pour détecter les opacités.<br/>
                    - Gain de temps : un geste et l'analyse démarre.
                </div>
            """)

        with gr.Column(scale=5):
            output = gr.HTML(
                label=None,
                elem_classes=["result-wrapper"],
                value="""
                    <div class="result-card">
                        <div class="result-placeholder">
                            Déposez une radiographie pour générer instantanément le score des pathologies clés (atélectasie, épanchement pleural, œdème, cardiomégalie...).
                        </div>
                    </div>
                """
            )

    analyze_btn.click(fn=predict, inputs=input_image, outputs=output)

if __name__ == "__main__":
    demo.launch()
