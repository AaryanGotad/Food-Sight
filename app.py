import streamlit as st

import tensorflow as tf
from PIL import Image

import threading
from pathlib import Path
from utils import utilities
import copy

# thread safety setup
MODEL_LOCK = threading.Lock()

# -----------------( DL MODEL LOADING )-----------------
@st.cache_resource
def load_model():
    """
    Loads the model once gloabally and caches it across all sessions.
    """
    model_path = Path(__file__).resolve().parent / 'model' / 'FoodSight.keras'
    return tf.keras.models.load_model(model_path)

left_col, middle_col, right_col = st.columns([1, 2, 1])

with middle_col:
    # initializing model
    try:
        with st.spinner('Loading model...'):
            model = load_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

# -----------------( TITLE & HEADING )-----------------
st.markdown(
    """
        <h1 
            style='text-align: center;
                   font-size: 4rem;
                   letter-spacing: 0.03rem;
                   font-family: "Copperplate";
                   text-decoration: underline;
                   margin-bottom: 0.05em;'>
            FoodSight
        </h1>
    """,
    unsafe_allow_html=True)

st.markdown(
    """
        <h2 
            style='text-align: center;
                    font-size: 1.95rem;
                    letter-spacing: 0.15rem;
                    font-family: "Roboto";'>
            See what's on your plate
        </h2>
    """,
    unsafe_allow_html=True)

st.write("")

# -----------------( IMAGE UPLOAD LOGIC )-----------------
# Initializing persistent session state for the target image
if "saved_image" not in st.session_state:
    st.session_state['saved_image'] = None

if st.session_state['saved_image'] is None:
    st.subheader('Provide an Image')

    # rendering input options if no image is currently stored
    uploaded_file = st.file_uploader(
        label="Upload a photo",
        type=['png', 'jpg', 'jpeg']
    )

    # st.markdown(
    #     """
    #         <h2 
    #             style='text-align: center;
    #                     font-size: 1.05rem;
    #                     letter-spacing: 0.15rem;
    #                     font-family: "Roboto";'>
    #             OR
    #         </h2>
    #     """,
    #     unsafe_allow_html=True)

    # # CAPTURE AN IMAGE
    # camera_file = st.camera_input("Capture a Photo")

    # if either widget recieved a new file
    # active_input = camera_file if camera_file is not None else uploaded_file
    active_input = uploaded_file

    if active_input is not None:
        # save file to state, then force a fresh rerun
        pil_image = Image.open(active_input)

        st.session_state['saved_image'] = pil_image
        st.rerun()

else:
    # render preview only
    st.subheader('Uploaded Photo')

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.image(st.session_state['saved_image'])

    with right_col:
        # clare the image and bring the input elements back
        if st.button("Delete and Upload Another"):
            st.session_state['saved_image'] = None
            st.rerun()

        st.space('small')

        predict_button = st.button('Identify This')
    
    if predict_button:
        with st.spinner('Identifying...'):
            # preprocessing image
            processd_image = utilities.preprocess_image(st.session_state['saved_image'])

            with MODEL_LOCK:
                predictions = model.predict(processd_image)

            # top 5 predictions with class names and confidence probabilities
            top_5_preds = utilities.top_k_preds(predictions)

            # displaying results
            st.success('Identified!')

            for i in range(len(top_5_preds)):
                top_5_preds[i]['Probability'] = float(top_5_preds[i]['confidence'])
                top_5_preds[i]['confidence'] = round(top_5_preds[i]['confidence'] * 100, 2)

            left_col, middle_col, right_col = st.columns([1, 2, 1])

            with right_col:
                # displaying raw model probability values
                with st.popover('Raw Model Outputs'):                    

                    raw_model_outputs = copy.deepcopy(top_5_preds)
                    for output in raw_model_outputs:
                        output.pop('confidence', None)

                    st.dataframe(raw_model_outputs)  

            st.write('We think it\'s a ', top_5_preds[0]['label'],
                    'with a confidence of', round(top_5_preds[0]['Probability'] * 100, 2), '%')

            st.progress(top_5_preds[0]['Probability'])

            st.space('medium')

            with st.expander('Top 5 Predictions'):

                st.dataframe(
                    top_5_preds,
                    column_config={
                            'label': st.column_config.TextColumn("Label"),
                            'confidence': st.column_config.NumberColumn(
                                'Confidence %',
                                format="%.2f"
                            ),
                            'Probability': st.column_config.ProgressColumn(
                                'Confidence Score',
                                help='Model predication confidence level',
                                format="%.2f", # to show numeric percemtage/decimal text beside the bar
                                min_value=0.0,
                                max_value=1.0
                            ),
                        },
                        hide_index=True
                    )
           


            
   