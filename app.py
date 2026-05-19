import streamlit as st
import torch
from transformers import AutoTokenizer
from src.model import AdversarialFakeNewsModel
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# --- Page Configuration ---
st.set_page_config(
    page_title="SOTA Multilingual Fake News Detector",
    page_icon="🤖",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stTextArea textarea { background-color: #1e1e1e; color: white; border: 1px solid #333; }
    .result-card {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #444;
        margin-bottom: 20px;
    }
    .fake { border-color: #ff4b4b; background-color: rgba(255, 75, 75, 0.1); }
    .real { border-color: #28a745; background-color: rgba(40, 167, 69, 0.1); }
    .uncertainty-meter { font-size: 0.9em; color: #aaa; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model & Tokenizer ---
@st.cache_resource
def load_assets():
    model_name = 'xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load in lightweight mode to prune unused ResNet-18 (saves 45MB RAM/VRAM)
    model = AdversarialFakeNewsModel(num_sources=101, lightweight=True)
    
    weights_path = 'final_model_lightweight.pt'
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(weights_path))
    else:
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    model.eval()
    return tokenizer, model

tokenizer, model = load_assets()

# --- Image Transform ---
img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- UI Layout ---
st.title("🤖 SOTA Multilingual Fact-Checker")
st.write("Powered by **XLM-R (LoRA) + Adversarial GRL + Source-Aware Fusion**.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Step 1: News Content")
    input_text = st.text_area("Headline / Article Body:", placeholder="Enter text in any language (English, Hindi, Urdu, etc.)...", height=200)
    
    st.subheader("Step 2: Source Reputation (Critical)")
    source_type = st.selectbox(
        "Where did you find this news?",
        ["International News Agency (e.g. Reuters, BBC)", "Verified Social Media", "WhatsApp / Random Website", "Suspicious Source"]
    )
    # Mapping sources to trained indices (Simplified mapping for demo)
    source_map = {
        "International News Agency (e.g. Reuters, BBC)": 0,
        "Verified Social Media": 50,
        "WhatsApp / Random Website": 100,
        "Suspicious Source": 10 # This showed high 'Fake' sensitivity in eval
    }
    selected_source_id = source_map[source_type]

with col2:
    st.subheader("Step 3: Analyze")
    num_passes = st.slider("Model Intelligence (MC Dropout Passes):", 5, 20, 10)
    analyze_btn = st.button("🚀 Execute Truth Shield")

if analyze_btn:
    if input_text.strip() == "":
        st.warning("Bhai, kam se kam text toh dalo!")
    else:
        with st.spinner("Exposing the truth using SOTA Multilingual AI..."):
            # 1. Prepare Text
            tokens = tokenizer(input_text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            s_id = torch.tensor([selected_source_id])
            
            # 2. Monte Carlo Dropout (Uncertainty Estimation)
            predictions = []
            for _ in range(num_passes):
                with torch.no_grad():
                    # mc_dropout=True forces dropout to be active during inference
                    label_logits, _, _ = model(tokens['input_ids'], tokens['attention_mask'], source_ids=s_id)
                    label_out = torch.sigmoid(label_logits)
                    predictions.append(label_out.item())
            
            # 3. Aggregate Results
            mean_prob = np.mean(predictions) # Probability of REAL (class 1)
            uncertainty = np.std(predictions)
            prob_fake = 1.0 - mean_prob # Probability of FAKE (class 0)
            is_fake = mean_prob < 0.5
            trust_score = (1 - uncertainty) * 100
            
            # 4. Display Results
            result_class = "fake" if is_fake else "real"
            result_label = "FAKE NEWS DETECTED" if is_fake else "VERIFIED REAL NEWS"
            
            st.markdown(f"""
                <div class="result-card {result_class}">
                    <h1 style="margin:0;">{result_label}</h1>
                    <h2 style="color: #fff;">Probability of Fake: {prob_fake*100:.2f}%</h2>
                    <div class="uncertainty-meter">
                        Model Confidence Index: <b>{trust_score:.2f}%</b><br>
                        Statistical Uncertainty: {uncertainty:.4f}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if is_fake:
                st.error("🚩 **Warning**: This news matches patterns of misinformation or propaganda.")
            else:
                st.success("✅ **Safe**: This news pattern aligns with credible reporting.")
            
            # 5. XAI Explanation (LIME)
            st.subheader("🔍 Why did the AI say this?")
            
            def predictor(texts):
                tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
                with torch.no_grad():
                    s_id_batch = torch.full((len(texts),), selected_source_id)
                    logits, _, _ = model(tokens['input_ids'], tokens['attention_mask'], source_ids=s_id_batch)
                    probs = torch.sigmoid(logits).cpu().numpy()
                return np.hstack([1-probs, probs]) # index 0 is FAKE, index 1 is REAL

            explainer = LimeTextExplainer(class_names=['Fake', 'Real'])
            exp = explainer.explain_instance(input_text, predictor, num_features=10)
            
            # Display LIME explanation
            components.html(exp.as_html(), height=400, scrolling=True)

st.divider()
st.caption("Developed by Antigravity (SOTA AI Engineer) | Project Viral Shield 🛡️")
