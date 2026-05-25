import streamlit as st
import numpy as np
from PIL import Image
import io
import pandas as pd
import streamlit.components.v1 as components
import altair as alt

# --- MANDATORY FIRST COMMAND ---
st.set_page_config(page_title="Baseer AI", page_icon="👁️", layout="centered")

# --- Imports & Engine Initialization ---
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


# One-time Load of the AI Brain
@st.cache_resource
def load_engine():
    try:
        return InferenceEngine()
    except Exception as e:
        return f"ERROR: {e}"


engine_res = load_engine()

# Visual Styling
st.markdown(
    """
<style>
    .main { background-color: #0e1117; }
    div.stButton > button {
        width: 100%; border-radius: 5px; height: 3em;
        background-color: #238636; color: white; font-weight: 600;
    }
    .instruction-text { font-size: 0.85rem; color: #8b949e; line-height: 1.4; }
</style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.title("👁️ Baseer AI")
    st.markdown("---")
    st.markdown("### 📋 Usage Guidelines")
    st.markdown(
        "<div class='instruction-text'>1. Upload a photo.<br>2. Click 'Analyze'.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### 🎯 Supported Classes")
    st.markdown(
        "<div class='instruction-text'><b>Transport:</b> Plane, Car, Ship, Truck<br><b>Animals:</b> Bird, Cat, Deer, Dog, Frog, Horse</div>",
        unsafe_allow_html=True,
    )
    st.write("---")
    st.info("System: Online (v1.0)\n\nDeveloped by Muhammad Hamza Saqib")

# --- Main Interface ---
st.title("Baseer AI")
st.caption("Discerning Eye (بصیر) — Specialized Object Recognition")
st.write("---")

# Error Handling for Engine
if isinstance(engine_res, str):
    st.error(f"AI Engine failed to initialize: {engine_res}")
    engine = None
else:
    engine = engine_res

uploaded_file = st.file_uploader(
    "Drop an image below for analysis", type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    st.markdown("<div id='analyze_anchor'></div>", unsafe_allow_html=True)
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)
    scroll_to("analyze_anchor")

    if st.button("RUN SPECTRAL ANALYSIS"):
        if engine is None:
            st.error("Engine offline.")
        else:
            with st.spinner("Processing tensors..."):
                try:
                    img_array = np.array(image.convert("RGB"))
                    res = engine.predict(img_array)
                    st.session_state["result"] = res
                except Exception as e:
                    st.error(f"Inference Error: {e}")

# --- Results ---
if "result" in st.session_state:
    st.markdown("<div id='result_anchor'></div>", unsafe_allow_html=True)
    st.write("---")
    res = st.session_state["result"]
    col1, col2 = st.columns(2)
    col1.metric("Predicted Identity", res["prediction"].title())
    col2.metric("Confidence Level", f"{res['confidence']*100:.1f}%")

    chart_df = pd.DataFrame(
        {
            "Category": [res["prediction"].title(), "Others"],
            "Probability": [res["confidence"], 1.0 - res["confidence"]],
        }
    )
    chart = (
        alt.Chart(chart_df)
        .mark_bar(color="#58a6ff")
        .encode(
            x=alt.X("Category", sort=None),
            y=alt.Y("Probability", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Category", "Probability"],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)
    scroll_to("result_anchor")

st.write("---")
st.markdown(
    "<p style='text-align: center; color: grey; font-size: 0.8rem;'>Baseer AI Infrastructure v1.0 | Licensed Apache 2.0</p>",
    unsafe_allow_html=True,
)
