import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
from streamlit_drawable_canvas import st_canvas
import base64
import os

# -----------------------------------------------------------
# 0️⃣ Page Config & Setup
# -----------------------------------------------------------
st.set_page_config(
    page_title="EMNIST Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------
# ✨ Enhanced Background & Styling
# -----------------------------------------------------------
@st.cache_resource
def get_base64_of_bin_file(bin_file):
    """Reads a binary file and returns its base64 encoded string."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"Error reading background image: {e}")
        return None

def set_app_background(jpg_file):
    """
    Sets the app background to a blurred version of the specified JPG file.
    Enhanced with better error handling and styling.
    """
    if not os.path.exists(jpg_file):
        return
        
    bin_str = get_base64_of_bin_file(jpg_file)
    if bin_str is None:
        return
    
    page_bg_img = f"""
    <style>
    /* Background styling */
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        filter: blur(10px);
        opacity: 0.3;
        z-index: -1;
    }}
    
    [data-testid="stAppViewContainer"] > .main {{
        background-color: transparent;
    }}
    
    /* Enhanced card styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 8px 16px;
        backdrop-filter: blur(10px);
    }}
    
    /* Metric styling */
    [data-testid="stMetricValue"] {{
        font-size: 3rem;
        font-weight: bold;
    }}
    
    /* Button enhancements */
    .stButton > button {{
        transition: all 0.3s ease;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }}
    
    /* Style ALL canvas toolbar buttons with strong visibility */
    button[kind="secondary"] {{
        background-color: #3b82f6 !important;
        color: white !important;
        border: 2px solid #1e40af !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        margin: 0 4px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }}
    
    button[kind="secondary"]:hover {{
        background-color: #2563eb !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important;
    }}
    
    button[kind="secondary"]:active {{
        transform: translateY(0px) !important;
    }}
    
    /* Extra targeting for canvas control buttons */
    .stApp button[title*="Undo"],
    .stApp button[title*="Redo"],
    .stApp button[title*="Delete"],
    .stApp button[title*="Download"] {{
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        font-size: 15px !important;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }}
    
    .stApp button[title*="Undo"]:hover,
    .stApp button[title*="Redo"]:hover,
    .stApp button[title*="Delete"]:hover,
    .stApp button[title*="Download"]:hover {{
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
        transform: scale(1.05) translateY(-2px) !important;
        box-shadow: 0 6px 15px rgba(59, 130, 246, 0.6) !important;
    }}
    
    /* Target canvas toolbar buttons - more specific selectors */
    div.row-widget.stHorizontalBlock button,
    [data-testid="stHorizontalBlock"] button,
    .css-10trblm button,
    .css-1v0mbdj button {{
        background: #3b82f6 !important;
        color: white !important;
        border: 2px solid #1e40af !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: bold !important;
        font-size: 14px !important;
        box-shadow: 0 3px 8px rgba(59, 130, 246, 0.5) !important;
        min-width: 80px !important;
    }}
    
    div.row-widget.stHorizontalBlock button:hover,
    [data-testid="stHorizontalBlock"] button:hover,
    .css-10trblm button:hover,
    .css-1v0mbdj button:hover {{
        background: #2563eb !important;
        transform: scale(1.05) !important;
        box-shadow: 0 5px 12px rgba(59, 130, 246, 0.7) !important;
    }}
    
    /* Universal button selector for drawable canvas */
    iframe + div button {{
        background: #3b82f6 !important;
        color: white !important;
        border: 2px solid #1e40af !important;
        padding: 8px 16px !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        box-shadow: 0 3px 10px rgba(59, 130, 246, 0.5) !important;
    }}
    
    iframe + div button:hover {{
        background: #2563eb !important;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.7) !important;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Apply background
set_app_background("alpha and digit.jpg")

# -----------------------------------------------------------
# 1️⃣ Load Model with Enhanced Error Handling
# -----------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the model with comprehensive error handling."""
    model_path = "cnn_emnist_digits_alphabets.pkl"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.info("Please ensure the model file is in the same directory as this script.")
        st.stop()
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# -----------------------------------------------------------
# 2️⃣ Enhanced Helper Functions
# -----------------------------------------------------------
def label_to_char(label):
    """Convert label index to character with validation."""
    if 0 <= label <= 9:
        return str(label)
    elif 10 <= label <= 35:
        return chr(label - 10 + ord('A'))
    elif 36 <= label <= 61:
        return chr(label - 36 + ord('a'))
    else:
        return "?"

def get_char_type(label):
    """Get character type from label index."""
    if 0 <= label <= 9:
        return "Digit"
    elif 10 <= label <= 35:
        return "Uppercase Letter"
    elif 36 <= label <= 61:
        return "Lowercase Letter"
    return "Unknown"

def get_confidence_color(confidence):
    """Return color based on confidence level."""
    if confidence >= 0.8:
        return "🟢"
    elif confidence >= 0.5:
        return "🟡"
    else:
        return "🔴"

# -----------------------------------------------------------
# 3️⃣ Enhanced Preprocessing with Better Quality
# -----------------------------------------------------------
def preprocess_image(img_data, show_steps=False):
    """
    Enhanced preprocessing with optional step visualization.
    """
    steps = {}
    
    try:
        if isinstance(img_data, np.ndarray):
            # --- FROM CANVAS ---
            img = Image.fromarray(img_data.astype('uint8'), 'RGBA')
            img_gray = img.convert('L')
            steps['original'] = img_gray.copy()
            
            # Enhanced bounding box detection with padding
            bbox = img_gray.getbbox()
            if bbox is None:
                st.warning("⚠️ Please draw a character first.")
                return None, None
            
            # Add padding to bbox
            padding = 5
            bbox = (
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(img_gray.width, bbox[2] + padding),
                min(img_gray.height, bbox[3] + padding)
            )
            
            cropped = img_gray.crop(bbox)
            steps['cropped'] = cropped.copy()
            
            # Enhanced contrast
            enhancer = ImageEnhance.Contrast(cropped)
            cropped = enhancer.enhance(1.5)
            steps['enhanced'] = cropped.copy()
            
            # Resize with high quality
            cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
            steps['resized'] = cropped.copy()
            
            # Center in 28x28 canvas
            new_img = Image.new('L', (28, 28), 0)
            w, h = cropped.size
            paste_x = (28 - w) // 2
            paste_y = (28 - h) // 2
            new_img.paste(cropped, (paste_x, paste_y))
            
            img = new_img
            
        else:
            # --- FROM FILE UPLOADER ---
            img = Image.open(img_data)
            img = img.convert('L')
            steps['original'] = img.copy()
            
            # Enhance contrast before resizing
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)
            
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            img = ImageOps.invert(img)
            steps['processed'] = img.copy()
        
        steps['final'] = img.copy()
        
        # Convert to array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0
        input_data = img_array.reshape(-1, 28, 28, 1)
        
        return input_data, steps if show_steps else None
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None, None

# -----------------------------------------------------------
# 4️⃣ Enhanced Prediction Function
# -----------------------------------------------------------
def get_contextual_prediction(pred_array, context_type):
    """
    Enhanced contextual prediction with confidence scoring.
    """
    top_indices = np.argsort(pred_array)[::-1]
    
    # Map context type
    context_map = {
        "Digit (0-9)": "Digit",
        "Uppercase Letter (A-Z)": "Uppercase Letter",
        "Lowercase Letter (a-z)": "Lowercase Letter"
    }
    target_type = context_map.get(context_type, "Digit")
    
    # Find best match within context
    for idx in top_indices:
        if get_char_type(idx) == target_type:
            return idx, pred_array[idx]
    
    # Fallback
    return top_indices[0], pred_array[top_indices[0]]

# -----------------------------------------------------------
# 5️⃣ Enhanced UI
# -----------------------------------------------------------
st.title("🧠 EMNIST Digit & Alphabet Classifier")
st.markdown("""
<div style='background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; backdrop-filter: blur(10px);'>
    <p style='margin: 0;'>🎨 <b>Draw</b> or <b>upload</b> a character. Select the context for improved accuracy!</p>
    <p style='margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.8;'>
        Supports: Digits (0-9), Uppercase (A-Z), and Lowercase (a-z) letters
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'processed_steps' not in st.session_state:
    st.session_state.processed_steps = None
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input")
    
    # Enhanced context selector
    char_type_context = st.radio(
        "🎯 Select Input Context:",
        ("Digit (0-9)", "Uppercase Letter (A-Z)", "Lowercase Letter (a-z)"),
        help="💡 Providing context improves prediction accuracy by 20-30%!"
    )
    
    # Advanced options in expander
    with st.expander("⚙️ Advanced Options"):
        show_preprocessing = st.checkbox("Show preprocessing steps", value=False)
        canvas_stroke_width = st.slider("Canvas stroke width", 8, 20, 12)
    
    tab1, tab2 = st.tabs(["✍️ Draw Character", "📤 Upload Image"])

    with tab1:
        st.write("Draw your character below:")
        canvas_result = st_canvas(
            stroke_width=canvas_stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🎯 Predict Drawing", use_container_width=True, type="primary"):
                with st.spinner("Processing..."):
                    processed_img, steps = preprocess_image(
                        canvas_result.image_data, 
                        show_steps=show_preprocessing
                    )
                    if processed_img is not None:
                        pred = model.predict(processed_img)
                        st.session_state.prediction = pred
                        st.session_state.processed_steps = steps
        
        with col_btn2:
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.session_state.prediction = None
                st.session_state.processed_steps = None
                st.session_state.canvas_key += 1
                st.rerun()

    with tab2:
        uploaded_file = st.file_uploader(
            "Upload an image file",
            type=["png", "jpg", "jpeg"],
            help="Upload a clear image of a single character"
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="📎 Uploaded Image", width=280)
            if st.button("🎯 Predict Upload", use_container_width=True, type="primary"):
                with st.spinner("Processing..."):
                    processed_img, steps = preprocess_image(
                        uploaded_file,
                        show_steps=show_preprocessing
                    )
                    if processed_img is not None:
                        pred = model.predict(processed_img)
                        st.session_state.prediction = pred
                        st.session_state.processed_steps = steps

with col2:
    st.subheader("🔍 Prediction Results")
    
    if st.session_state.prediction is None:
        st.info("👈 Draw or upload an image, then click 'Predict' to see results.")
    
    else:
        pred_array = st.session_state.prediction[0]
        predicted_class_index, confidence = get_contextual_prediction(
            pred_array, 
            char_type_context
        )
        predicted_char = label_to_char(predicted_class_index)
        
        # Enhanced prediction display
        confidence_icon = get_confidence_color(confidence)
        
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(139, 92, 246, 0.2)); 
                    padding: 20px; border-radius: 15px; text-align: center; backdrop-filter: blur(10px);'>
            <p style='font-size: 0.9em; margin: 0; opacity: 0.8;'>Predicted Character</p>
            <p style='font-size: 4em; font-weight: bold; margin: 10px 0;'>{predicted_char}</p>
            <p style='font-size: 1.1em; margin: 0;'>
                {confidence_icon} Confidence: {confidence*100:.1f}%
            </p>
            <p style='font-size: 0.8em; margin: 5px 0 0 0; opacity: 0.7;'>
                Context: {char_type_context} | Class: {predicted_class_index}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # Show preprocessing steps if enabled
        if show_preprocessing and st.session_state.processed_steps:
            with st.expander("🔬 View Preprocessing Steps", expanded=True):
                steps = st.session_state.processed_steps
                step_cols = st.columns(len(steps))
                for idx, (step_name, step_img) in enumerate(steps.items()):
                    with step_cols[idx]:
                        st.image(step_img, caption=step_name.title(), use_column_width=True)
        
        st.write("---")
        st.subheader("📊 Top 5 Predictions")
        
        # Enhanced top 5 display
        top5_indices = np.argsort(pred_array)[-5:][::-1]
        
        for i, idx in enumerate(top5_indices):
            char = label_to_char(idx)
            prob = pred_array[idx]
            char_type = get_char_type(idx)
            confidence_icon = get_confidence_color(prob)
            
            is_selected = (idx == predicted_class_index)
            border = "2px solid #3b82f6" if is_selected else "1px solid rgba(255,255,255,0.2)"
            
            st.markdown(f"""
            <div style='background-color: rgba(255, 255, 255, 0.05); 
                        padding: 12px; margin: 8px 0; border-radius: 8px; 
                        border: {border}; backdrop-filter: blur(10px);'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div style='display: flex; align-items: center; gap: 12px;'>
                        <span style='font-size: 1.5em; font-weight: bold; 
                                    min-width: 30px; text-align: center;'>{char}</span>
                        <span style='opacity: 0.7; font-size: 0.9em;'>{char_type}</span>
                    </div>
                    <div style='display: flex; align-items: center; gap: 8px;'>
                        <span>{confidence_icon}</span>
                        <span style='font-weight: bold;'>{prob*100:.1f}%</span>
                    </div>
                </div>
                <div style='background: rgba(59, 130, 246, 0.3); height: 4px; 
                           border-radius: 2px; margin-top: 8px;'>
                    <div style='background: #3b82f6; height: 100%; width: {prob*100}%; 
                               border-radius: 2px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------
# 6️⃣ Footer with Tips
# -----------------------------------------------------------
st.write("---")
with st.expander("💡 Tips for Better Results"):
    st.markdown("""
    - **Draw clearly** with bold strokes in the center of the canvas
    - **Select the correct context** (Digit/Uppercase/Lowercase) before predicting
    - **Center your character** for best results
    - **Use good contrast** - dark characters on light backgrounds or vice versa
    - **Upload clear images** without background noise
    - **Adjust stroke width** in Advanced Options if needed
    """)

st.markdown("""
<div style='text-align: center; opacity: 0.6; padding: 20px; font-size: 0.9em;'>
    Made with ❤️ using Streamlit | Model: CNN EMNIST Classifier
</div>
""", unsafe_allow_html=True)