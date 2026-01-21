import streamlit as st
import requests
import numpy as np
from PIL import Image

API_URL = "http://127.0.0.1:8000/file"

st.set_page_config(page_title="Transistor Defect Detection", layout="centered")

st.title("🔍 Transistor Defect Detection")
st.write("Upload an image to run defect detection.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Run inference"):
        with st.spinner("Running model..."):
            response = requests.post(
                API_URL,
                files={"file": uploaded_file.getvalue()}
            )

        result = response.json()["result"]
        st.success("Inference complete")
        st.write("Model output:")
        mask = np.array(result)

        # ensure 2D
        if mask.ndim == 2:
            
            # convert 0/1 → 0/255
            mask_img = (mask * 255).astype(np.uint8)

            image = Image.fromarray(mask_img, mode="L")

            st.image(image, caption="Predicted defect mask", use_container_width=True)
        else:
            st.error(f"API error: {response.status_code}")


#streamlit run streamlit_app.py
