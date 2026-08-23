import streamlit as st

import tensorflow as tf
from PIL import Image

import threading
from utils import utilities

# thread safety setup
MODEL_LOCK = threading.Lock()

# -----------------( DL MODEL LOADING )-----------------
@st.cache_resource
def load_model():
    """
    Loads the model once gloabally and caches it across all sessions.
    """
    return tf.keras.models.load_model('/workspaces/Food-Sight/model/FoodSight.keras')

# initializing model
model = load_model()

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

st.space('small')

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

    st.markdown(
        """
            <h2 
                style='text-align: center;
                        font-size: 1.05rem;
                        letter-spacing: 0.15rem;
                        font-family: "Roboto";'>
                OR
            </h2>
        """,
        unsafe_allow_html=True)

    # CAPTURE AN IMAGE
    camera_file = st.camera_input("Capture a Photo")

    # if either widget recieved a new file
    active_input = camera_file if camera_file is not None else uploaded_file

    if active_input is not None:
        # save file to state, then force a fresh rerun
        pil_image = Image.open(active_input)

        st.session_state['saved_image'] = pil_image
        st.rerun()

else:
    # render preview only
    st.subheader('Uploaded Photo')
    st.image(st.session_state['saved_image'])

    # clare the image and bring the input elements back
    if st.button("Delete and Upload Another"):
        st.session_state['saved_image'] = None
        st.rerun()

    if st.button('Identify This'):
        with st.spinner('Identifying...'):
            # preprocessing image
            processd_image = utilities.preprocess_image(st.session_state['saved_image'])

            with MODEL_LOCK:
                predictions = model.predict(processd_image)

            top_5_preds = utilities.top_k_preds(predictions)

            # output results
            st.success('Identified!')
            st.write('Raw model output:', top_5_preds)
   