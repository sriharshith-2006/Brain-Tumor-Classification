import streamlit as st
import torch
import os
import gdown
from torchvision import transforms
from PIL import Image
from model import CNN_Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH="saved_models/best_model.pth"
MODEL_URL="https://drive.google.com/uc?id=1ylfCWm6Dfl13rZLYeYoCmQFxx3MJwe-8"
if not os.path.exists(MODEL_PATH):
    os.makedirs("saved_models",exist_ok=True)
    with st.spinner("Downloading Model...."):
        gdown.download(MODEL_URL,MODEL_PATH,quiet=False)
classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Base text color fix — default Streamlit text was inheriting a light
       color that's invisible on a white page (this was the "black box" /
       disappearing confidence text). Every element gets a readable dark
       color unless something more specific overrides it below. */
    .stApp, .stApp p, .stApp span, .stApp li, .stApp label,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] small,
    [data-testid="stAlert"] p {
        color: #2b2440;
    }

    /* Streamlit's own top toolbar renders dark by default — match it to
       the page instead of leaving a black bar. */
    [data-testid="stHeader"] {
        background: #fffaf7;
    }

    html, body {
        background-color: #f7f4ff !important;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    .main, .block-container {
        background: transparent !important;
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(480px 480px at 8% 6%, rgba(244,114,182,0.30), transparent 65%),
            radial-gradient(520px 520px at 95% 12%, rgba(56,189,248,0.28), transparent 65%),
            radial-gradient(460px 460px at 88% 92%, rgba(52,211,153,0.26), transparent 65%),
            radial-gradient(420px 420px at 4% 88%, rgba(250,204,21,0.22), transparent 65%),
            linear-gradient(135deg, #fdf4ff 0%, #f5f3ff 30%, #eff6ff 60%, #f0fdfa 100%) !important;
        background-attachment: fixed;
    }

    [data-testid="stAlert"] {
        background: #eef4ff;
        border: 1px solid #dbeafe;
        border-radius: 12px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(200deg, #0f766e 0%, #0e7490 45%, #1e3a8a 100%);
    }
    [data-testid="stSidebar"] * { color: #ecfeff !important; }

    /* ---------- Hero ---------- */
    .hero { text-align: center; padding: 2rem 1rem 0.5rem 1rem; }
    .hero .pill {
        display: inline-block;
        background: linear-gradient(90deg, #fbcfe8, #ddd6fe);
        color: #6d28d9;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.05em;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        margin-bottom: 1rem;
    }
    .hero h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.7rem;
        line-height: 1.15;
        color: #18122b;
        margin-bottom: 0.6rem;
    }
    .hero h1 .grad {
        background: linear-gradient(90deg, #ec4899, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p.sub {
        font-size: 1.12rem;
        color: #5b5470;
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Thin rainbow accent strip along the very top of the page */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 5px;
        background: linear-gradient(90deg, #ec4899, #8b5cf6, #06b6d4, #34d399, #facc15);
        z-index: 999;
    }

    /* ---------- Card ---------- */
    .glass-card {
        background: #ffffff;
        border: 1px solid #eee7fb;
        border-top: 3px solid transparent;
        border-image: linear-gradient(90deg, #ec4899, #8b5cf6, #06b6d4) 1;
        border-radius: 0 0 22px 22px;
        padding: 1.9rem;
        margin-top: 1.4rem;
        box-shadow: 0 12px 30px rgba(124,58,237,0.12);
    }
    .glass-card h3 {
        font-family: 'Poppins', sans-serif;
        color: #18122b; font-weight: 700; margin-bottom: 0.5rem;
    }
    .glass-card p { color: #5b5470; line-height: 1.65; font-size: 1rem; }
    .glass-card strong { color: #7c3aed; }

    /* ---------- Feature tiles ---------- */
    .tile {
        border-radius: 18px;
        padding: 1.3rem 1.1rem;
        height: 100%;
        color: white;
        box-shadow: 0 10px 22px rgba(0,0,0,0.10);
        transition: transform 0.2s ease;
    }
    .tile:hover { transform: translateY(-4px); }
    .tile .emoji { font-size: 1.6rem; margin-bottom: 0.3rem; display:block; }
    .tile h4 { font-family: 'Poppins', sans-serif; font-size: 1rem; margin: 0.1rem 0 0.3rem 0; }
    .tile p { font-size: 0.85rem; color: rgba(255,255,255,0.92); margin: 0; line-height: 1.4; }
    .t1 { background: linear-gradient(150deg, #f472b6, #db2777); }
    .t2 { background: linear-gradient(150deg, #fb923c, #ea580c); }
    .t3 { background: linear-gradient(150deg, #34d399, #059669); }
    .t4 { background: linear-gradient(150deg, #38bdf8, #2563eb); }

    /* ---------- Upload dropzone ---------- */
    .drop-header {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-bottom: 1rem;
    }
    .drop-icon {
        font-size: 1.6rem;
        background: linear-gradient(135deg, #fce7f3, #ede9fe);
        border-radius: 14px;
        width: 52px; height: 52px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .drop-header h4 {
        font-family: 'Poppins', sans-serif;
        color: #18122b; font-size: 1.05rem; margin: 0;
    }
    .drop-header p { color: #8a83a0; font-size: 0.86rem; margin: 0.15rem 0 0 0; }

    section[data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #fdf2fb, #f0f9ff);
        border: 2.5px dashed #c026d3;
        border-radius: 16px;
        transition: border-color 0.2s ease, background 0.2s ease;
    }
    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #8b5cf6;
        background: linear-gradient(135deg, #fce7f3, #ede9fe);
    }
    /* The built-in "Browse files" button was inheriting an invisible
       light-on-light color — give it a solid, clearly visible fill. */
    section[data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(90deg, #ec4899, #8b5cf6) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    section[data-testid="stFileUploaderDropzone"] button:hover {
        opacity: 0.92;
    }
    section[data-testid="stFileUploaderDropzone"] button p { color: #ffffff !important; }

    .preview-frame {
        margin-top: 1.4rem;
        padding-top: 1.4rem;
        border-top: 1px dashed #e9e3f7;
    }
    .file-chip {
        display: inline-block;
        margin-top: 0.7rem;
        background: #f3effc;
        color: #6d28d9;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
    }

    .result-card {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        background: linear-gradient(135deg, #ecfdf5, #ecfeff);
        border: 1px solid #a7f3d0;
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
    }
    .result-card .result-icon {
        font-size: 1.6rem;
        background: white;
        border-radius: 12px;
        width: 46px; height: 46px;
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 10px rgba(5,150,105,0.15);
        flex-shrink: 0;
    }
    .result-card .result-label {
        font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em;
        color: #059669; font-weight: 600;
    }
    .result-card .result-value {
        font-family: 'Poppins', sans-serif;
        font-size: 1.25rem; font-weight: 700; color: #064e3b;
    }
    .conf-chip {
        display: inline-block;
        margin-top: 0.6rem;
        background: #fff7ed;
        color: #c2410c;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(90deg, #ec4899, #8b5cf6, #06b6d4);
        background-size: 200% auto;
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.7rem 1.7rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        box-shadow: 0 6px 18px rgba(139,92,246,0.30);
        transition: 0.3s ease-in-out;
    }
    .stButton > button:hover {
        background-position: right center;
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 22px rgba(236,72,153,0.35);
    }

    .result-badge {
        display: inline-block;
        padding: 0.7rem 1.5rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 1.15rem;
        background: linear-gradient(90deg, #34d399, #06b6d4);
        color: #052e16;
        margin-top: 0.6rem;
        box-shadow: 0 6px 18px rgba(52,211,153,0.30);
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #f472b6, #facc15, #34d399);
    }

    .footer {
        text-align: center;
        color: #9a93ad;
        font-size: 0.85rem;
        margin-top: 2.8rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)
st.sidebar.markdown("## 🧠 Brain Tumor AI")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Upload and predict"],
    label_visibility="visible"
)
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ For research/educational purposes only. Not a substitute for professional medical diagnosis.")
if page == "Home":
    st.markdown("""
    <div class="hero">
        <span class="pill">🧠 AI-Powered Medical Imaging</span>
        <h1>See what's really<br>going on <span class="grad">inside the scan</span></h1>
        <p class="sub">Upload a brain MRI and get an instant, AI-driven read on whether
        it shows a glioma, meningioma, pituitary tumor — or nothing at all.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>👋 Welcome</h3>
        <p>This app runs a <strong>Convolutional Neural Network</strong> trained on thousands
        of MRI scans to spot patterns the human eye can miss. Just upload a scan on the
        <strong>Upload and predict</strong> page and get a result — class, confidence, and
        all — in seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    col1, col2, col3, col4 = st.columns(4)
    tiles = [
        ("t1", "📤", "Drag & Drop", "Upload any JPG or PNG MRI scan instantly"),
        ("t2", "⚡", "Instant Results", "Predictions in seconds, powered by deep learning"),
        ("t3", "🎯", "4 Tumor Types", "Glioma · Meningioma · Pituitary · No Tumor"),
        ("t4", "📊", "Confidence Score", "See exactly how sure the model really is"),
    ]
    for col, (cls, icon, title, desc) in zip([col1, col2, col3, col4], tiles):
        with col:
            st.markdown(f"""
            <div class="tile {cls}">
                <span class="emoji">{icon}</span>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">Built using Streamlit · CNN-based Brain Tumor Classifier</div>',
        unsafe_allow_html=True
    )
elif page == "Upload and predict":
    st.markdown("""
    <div class="hero">
        <span class="pill">🔬 Scan Analysis</span>
        <h1>Upload your <span class="grad">MRI scan</span></h1>
        <p class="sub">Drop an image below and let the model do the rest.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="drop-header">
            <div class="drop-icon">📁</div>
            <div>
                <h4>Drag &amp; drop your scan</h4>
                <p>JPG or PNG, up to a 200 MB — or click to browse</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    upload = st.file_uploader(
        "Upload MRI scan here",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG",
        label_visibility="collapsed"
    )

    if upload is not None:
        image = Image.open(upload).convert("RGB")
        st.markdown('<div class="preview-frame">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, use_container_width=True)
            st.markdown(
                f'<div class="file-chip">🖼️ {upload.name}</div>',
                unsafe_allow_html=True
            )
        with col2:
            st.write("")
            st.write("")
            predict_clicked = st.button("🔍 Analyze Scan", use_container_width=True)
            if predict_clicked:
                with st.spinner("Analyzing scan..."):
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor()
                    ])
                    model = CNN_Model().to(device)
                    model.load_state_dict(
                        torch.load(MODEL_PATH, map_location=device)
                    )
                    model.eval()
                    input_image = transform(image).unsqueeze(0)
                    input_image = input_image.to(device)

                    with torch.no_grad():
                        outputs = model(input_image)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, prediction = torch.max(probabilities, dim=1)

                pred_label = classes[prediction.item()]
                conf_pct = confidence.item() * 100

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-icon">🧠</div>
                    <div class="result-text">
                        <div class="result-label">Predicted Tumor Class</div>
                        <div class="result-value">{pred_label}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.write("")
                st.progress(confidence.item())
                st.markdown(
                    f'<div class="conf-chip">Confidence: {conf_pct:.2f}%</div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<p style="text-align:center; color:#9a93ad; margin-top:0.5rem;">'
            'No scan uploaded yet — choose a file above to get started.</p>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="footer">Built using Streamlit · CNN-based Brain Tumor Classifier</div>',
        unsafe_allow_html=True
    )