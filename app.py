import streamlit as st
from PIL import Image
import io
import pandas as pd
import streamlit.components.v1 as components
import altair as alt
import numpy as np

# Import the "AI Brain" directly for Cloud Compatibility
try:
    from src.inference.wrapper import InferenceEngine
except ImportError:
    st.error("Inference module not found. Check repository structure.")

# Helper function for smooth scrolling
def scroll_to(element_id):
    components.html(
        f"""
        <script>
            var element = window.parent.document.getElementById('{element_id}');
            if (element) {{
                element.scrollIntoView({{behavior: 'smooth', block: 'start'}});
            }}
        </script>
        """,
        height=0,
    )

# --- Initialize the Engine (One-time Load) ---
@st.cache_resource
def load_engine():
    return InferenceEngine()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed to initialize AI Engine: {e}")
    engine = None

st.set_page_config(page_title="Baseer AI", page_icon="👁️", layout="centered")

# Visual Styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    div.stButton > button {
        width: 100%; border-radius: 5px; height: 3em;
        background-color: #238636; color: white; font-weight: 600;
    }
    .instruction-text { font-size: 0.85rem; color: #8b949e; line-height: 1.4; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("👁️ Baseer AI")
    st.markdown("---")
    st.markdown("### 📋 Usage Guidelines")
    st.markdown("<div class='instruction-text'>1. Upload a clear, centered photo.<br>2. Click 'Analyze' to begin.</div>", unsafe_allow_html=True)
    
    st.markdown("### 🎯 Supported Classes")
    st.markdown("<div class='instruction-text'><b>Transport:</b> Plane, Car, Ship, Truck<br><b>Animals:</b> Bird, Cat, Deer, Dog, Frog, Horse</div>", unsafe_allow_html=True)
    
    st.write("---")
    st.info("System: Online (v1.0)\n\nDeveloped by Muhammad Hamza Saqib")

# --- Main Interface ---
st.title("Baseer AI")
st.caption("Discerning Eye (بصیر) — Specialized Object Recognition")
st.write("---")

uploaded_file = st.file_uploader("Drop an image below for analysis", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.markdown("<div id='analyze_anchor'></div>", unsafe_allow_html=True)
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    
    scroll_to("analyze_anchor")
    
    if st.button("RUN SPECTRAL ANALYSIS"):
        if engine is None:
            st.error("AI Engine is not initialized. Please check logs.")
        else:
            with st.spinner("Processing tensors via Direct Link..."):
                try:
                    # Convert Image to Numpy for the engine
                    img_array = np.array(image.convert("RGB"))
                    
                    # Run Inference Directly (No URL needed!)
                    res = engine.predict(img_array)
                    st.session_state['result'] = res
                except Exception as e:
                    st.error(f"Inference Error: {e}")

# --- Step 2: Results ---
if 'result' in st.session_state:
    st.markdown("<div id='result_anchor'></div>", unsafe_allow_html=True)
    st.write("---")
    
    res = st.session_state['result']
    conf_pct = res['confidence'] * 100
    
    col1, col2 = st.columns([1, 1])
    col1.metric("Predicted Identity", res['prediction'].title())
    col2.metric("Confidence Level", f"{conf_pct:.1f}%")
    
    chart_df = pd.DataFrame({
        'Category': [res['prediction'].title(), 'Others'],
        'Probability': [res['confidence'], 1.0 - res['confidence']]
    })
    
    chart = alt.Chart(chart_df).mark_bar(color='#58a6ff').encode(
        x=alt.X('Category', sort=None),
        y=alt.Y('Probability', scale=alt.Scale(domain=[0, 1])),
        tooltip=['Category', 'Probability']
    ).properties(height=300)
    
    st.altair_chart(chart, use_container_width=True)
    scroll_to("result_anchor")
    
st.write("---")
st.markdown("<p style='text-align: center; color: grey; font-size: 0.8rem;'>Baseer AI Infrastructure v1.0 | Licensed Apache 2.0</p>", unsafe_allow_html=True)
