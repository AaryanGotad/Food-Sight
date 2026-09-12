import streamlit as st
import tensorflow as tf
from PIL import Image
import threading
from pathlib import Path
from utils import utilities
import copy

# -----------------( PAGE CONFIGURATION )-----------------
st.set_page_config(
    page_title="FoodSight | See what's on your plate",
    page_icon="🍽️",
    layout="centered"
)

# -----------------( CUSTOM CSS / THEMING )-----------------
st.markdown("""
    <style>
        /* Apple-inspired typography */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }
        
        /* Main title styling */
        .hero-title {
            font-size: 3.5rem;
            font-weight: 800;
            letter-spacing: -0.05rem;
            color: #1c1c1e;
            text-align: center;
            margin-bottom: 0px;
            padding-bottom: 0px;
        }
        
        .hero-subtitle {
            font-size: 1.4rem;
            font-weight: 400;
            color: #8e8e93;
            text-align: center;
            margin-top: 0px;
            margin-bottom: 1.5rem;
        }

        /* Image corner radii */
        img {
            border-radius: 16px;
        }
        
        /* Progress bar warm accent overriding */
        .stProgress > div > div > div > div {
            background-color: #FF6B6B; 
        }

        /* Percentage badge styling */
        .confidence-badge {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1c1c1e;
        }

        /* Footer styling */
        .footer {
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid #e5e5ea;
            text-align: center;
            font-size: 0.85rem;
            color: #8e8e93;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------( THREAD SAFETY & MODEL LOADING )-----------------
MODEL_LOCK = threading.Lock()

@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / 'model' / 'FoodSight.keras'
    return tf.keras.models.load_model(model_path)

@st.cache_data
def load_class_names():
    """Loads all 101 class names from file."""
    possible_paths = [
        Path(__file__).resolve().parent / 'class_names.txt',
        Path(__file__).resolve().parent / 'utils' / 'class_names.txt',
        Path('class_names.txt'),
        Path('utils/class_names.txt')
    ]
    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
    return []

with st.spinner('Waking up FoodSight...'):
    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

class_names = load_class_names()

def load_sample_image(class_raw_name):
    """Loads a sample image from assets/samples if available."""
    sample_path = Path(__file__).resolve().parent / 'assets' / 'samples' / f"{class_raw_name}.jpg"
    if sample_path.exists():
        return Image.open(sample_path).convert('RGB')
    return None

# -----------------( SESSION STATE INITIALIZATION )-----------------
if "saved_image" not in st.session_state:
    st.session_state['saved_image'] = None

# -----------------( 1. HERO SECTION )-----------------
st.markdown("<h1 class='hero-title'>FoodSight</h1>", unsafe_allow_html=True)
st.markdown("<p class='hero-subtitle'>See what's on your plate.</p>", unsafe_allow_html=True)

# -----------------( 2. EXPLORE 101 CLASSES & SAMPLE SELECTOR )-----------------
st.markdown("### Explore the Menu & Test Samples")
st.caption("FoodSight is trained to recognize 101 distinct dishes. Select a sample below to test the classifier:")

if class_names:
    formatted_classes = {c.replace('_', ' ').title(): c for c in class_names}
    
    # Quick-select feature pills for popular items
    popular_samples = ["Pad Thai", "Hamburger", "Ramen", "Pizza", "Sushi", "Tacos", "Macarons"]
    selected_pill = st.pills(
        label="Popular Samples",
        options=popular_samples,
        selection_mode="single",
        label_visibility="collapsed"
    )
    
    # Searchable selectbox for all 101 classes
    selected_class_formatted = st.selectbox(
        "Or search through all 101 supported classes:",
        options=["Select a dish..."] + list(formatted_classes.keys())
    )
    
    # Handle sample selection from pills or selectbox
    target_sample = None
    if selected_pill:
        target_sample = formatted_classes.get(selected_pill)
    elif selected_class_formatted != "Select a dish...":
        target_sample = formatted_classes.get(selected_class_formatted)
        
    if target_sample:
        sample_img = load_sample_image(target_sample)
        if sample_img is not None:
            if st.button(f"Load Sample: {target_sample.replace('_', ' ').title()}", type="secondary"):
                st.session_state['saved_image'] = sample_img
                st.rerun()
        else:
            st.info(f"Sample preview selected for **{target_sample.replace('_', ' ').title()}**. (Add `{target_sample}.jpg` to `assets/samples/` to enable local testing).")

st.divider()

# -----------------( 3. IMAGE INPUT & PREVIEW )-----------------
if st.session_state['saved_image'] is None:
    st.markdown("### Show FoodSight what's on your plate")
    st.caption("Upload a food photo or drop an image file below.")
    
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert('RGB')
        st.session_state['saved_image'] = pil_image
        st.rerun()

else:
    # Image Preview & Actions
    col_img, col_actions = st.columns([2, 1])
    
    with col_img:
        st.image(st.session_state['saved_image'], width='stretch')
        
    with col_actions:
        st.markdown("### Ready for Analysis")
        st.caption("Image prepped for the EfficientNet-B0 classifier.")
        
        predict_button = st.button("Identify This", type="primary", width='stretch')
        
        if st.button("Delete and Upload Another", width='stretch'):
            st.session_state['saved_image'] = None
            st.rerun()

    # -----------------( 4. ANALYSIS & PREPROCESSING VISUALIZATION )-----------------
    if predict_button:
        st.divider()
        with st.spinner('Identifying...'):
            processed_image = utilities.preprocess_image(st.session_state['saved_image'])

            with MODEL_LOCK:
                predictions = model.predict(processed_image)

            top_5_preds = utilities.top_k_preds(predictions)
            
            for i in range(len(top_5_preds)):
                top_5_preds[i]['Probability'] = float(top_5_preds[i]['confidence'])
                top_5_preds[i]['confidence'] = round(top_5_preds[i]['confidence'] * 100, 2)
                
            state = utilities.determine_prediction_state(top_5_preds)

        # -----------------( "WHAT THE MODEL SEES" - PLACED BEFORE PREDICTION RESULTS )-----------------
        st.markdown("### What the model sees")
        st.caption("Before classification, your image is standardized to a 224 × 224 RGB tensor. This is the exact input fed into the neural network.")
        
        vis_col1, vis_col2, vis_col3 = st.columns([2, 1, 2])
        
        with vis_col1:
            st.image(st.session_state['saved_image'], width='stretch')
            st.markdown("<p style='text-align:center; font-size: 0.8rem; color: #8e8e93;'>Original Image</p>", unsafe_allow_html=True)
            
        with vis_col2:
            st.markdown("<div style='display: flex; height: 100%; align-items: center; justify-content: center; padding-top: 2rem;'><h2 style='color:#8e8e93;'>→</h2></div>", unsafe_allow_html=True)
            
        with vis_col3:
            resized_preview = st.session_state['saved_image'].resize((224, 224))
            st.image(resized_preview, width='stretch')
            st.markdown("<p style='text-align:center; font-size: 0.8rem; color: #8e8e93;'>224 × 224 Processed Tensor</p>", unsafe_allow_html=True)

        st.divider()

        # -----------------( 5. PREDICTION RESULTS & RAW OUTPUTS )-----------------
        st.success('Identified!')

        # Original Raw Model Outputs Popover Implementation
        left_col, middle_col, right_col = st.columns([1, 2, 1])
        with right_col:
            with st.popover('Raw Model Outputs'):                    
                raw_model_outputs = copy.deepcopy(top_5_preds)
                for output in raw_model_outputs:
                    output.pop('confidence', None)
                st.dataframe(raw_model_outputs)

        # Primary Result Display with Explicit Percentage
        top_label = top_5_preds[0]['label'].replace('_', ' ').title()
        top_conf = top_5_preds[0]['confidence']
        top_prob = top_5_preds[0]['Probability']

        if state == "CONFIDENT":
            st.write(f"We think it's a **{top_label}** with a confidence of **{top_conf}%**")
            st.progress(top_prob)
            
        elif state == "UNCERTAIN":
            st.warning("FoodSight isn't completely sure. These are the classes it considered most likely:")
            st.write(f"1. **{top_label}** — **{top_conf}%**")
            st.progress(top_prob)
            
            second_label = top_5_preds[1]['label'].replace('_', ' ').title()
            second_conf = top_5_preds[1]['confidence']
            second_prob = top_5_preds[1]['Probability']
            st.write(f"2. **{second_label}** — **{second_conf}%**")
            st.progress(second_prob)
            
        elif state == "CONFUSED":
            st.error("FoodSight is having trouble identifying this dish confidently.")
            st.markdown("The highest prediction confidence is too low. The image may fall outside the 101 categories or contain obscure framing.")

        st.space('medium')

        # Top 5 Breakdown Table with Percentages & Progress Column
        with st.expander('Top 5 Predictions'):
            st.dataframe(
                top_5_preds,
                column_config={
                    'label': st.column_config.TextColumn("Label"),
                    'confidence': st.column_config.NumberColumn(
                        'Confidence %',
                        format="%.2f%%"
                    ),
                    'Probability': st.column_config.ProgressColumn(
                        'Confidence Score',
                        help='Model prediction confidence level',
                        format="%.2f",
                        min_value=0.0,
                        max_value=1.0
                    ),
                },
                hide_index=True,
                width='stretch'
            )

# -----------------( 6. GUIDELINES & LIMITATIONS )-----------------
st.write("")
st.write("")
st.markdown("### Give FoodSight its best shot")
guide_col1, guide_col2 = st.columns(2)

with guide_col1:
    st.markdown("""
    **Photo Guidelines**
    * **Keep it centered:** Ensure the food is the primary subject.
    * **Good lighting:** Natural light helps distinguish textures.
    * **Clear the frame:** Avoid covering the dish with hands or heavy packaging.
    * **Clear angle:** Capture the dish from an angle where key ingredients are visible.
    """)

with guide_col2:
    st.markdown("""
    **Why might FoodSight be wrong?**
    * **Fixed 101 Dataset:** FoodSight only recognizes [101 specific dishes](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). Unlisted foods will yield a best-guess match.
    * **Visual Similarity:** Dishes like chocolate cake and chocolate mousse share almost identical visual features.
    * **Pattern Recognition:** The model learns pixel patterns, not ingredients or taste.
    """)

# -----------------( 7. FOOTER )-----------------
st.markdown("""
    <div class="footer">
        <strong>FoodSight</strong><br>
        © 2026 FoodSight | Designed & Developed by Aaryan Gotad<br>
        Engineered with ❤️ at IIT Guwahati
    </div>
""", unsafe_allow_html=True)