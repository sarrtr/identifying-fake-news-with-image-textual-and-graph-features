import streamlit as st
import requests
import base64
import os
import streamlit.components.v1 as components

API_URL = os.environ.get("API_URL", "http://0.0.0.0:8000")

# Configure the Streamlit page settings
st.set_page_config(page_title="Multimodal Fake News Detector", layout="centered")
st.title("Multimodal Fake-News Detector with XAI")

with st.form("input_form"):
    # Text input for news content
    text = st.text_area("News text / headline", height=120)
    
    # Image upload functionality
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    
    # Configuration slider for LIME explainer sensitivity
    lime_samples = st.slider("LIME samples (image explainer)", min_value=50, max_value=1000, value=200, step=50)
    
    # Form submission button
    submitted = st.form_submit_button("Analyze")

if submitted:
    # Validate that both text and image are provided
    if not text or not uploaded_file:
        st.warning("Please provide both text and an image.")
    else:
        # Prepare files and data for API request
        files = {
            "image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        data = {"text": text, "lime_samples": str(lime_samples)}
        
        # Send request to backend with loading indicator
        with st.spinner("Sending to backend and waiting for results..."):
            try:
                resp = requests.post(API_URL+'/predict', data=data, files=files, timeout=600)
                resp.raise_for_status()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        # Parse JSON response from backend
        result = resp.json()

        # Display original input data
        st.subheader("Input")
        st.write("Text:")
        st.write(result.get("text"))
        st.write("Image:")
        st.image(uploaded_file, use_container_width=True)

        # Show model prediction results
        st.subheader("Model prediction:")
        label = result.get("label")
        confidence = result.get("confidence")
        st.markdown(f"**{label.upper()}**  (confidence: {confidence:.3f})")

        # Display text explanation using LIME
        st.subheader("Text explanation")
        lime_list = result.get("lime_text_list", [])
        if lime_list:
            st.write("Top tokens and weights:")
            for token, weight in lime_list:
                st.write(f"{token} → {weight:.4f}")

        # Show detailed LIME HTML explanation
        lime_html = result.get("lime_text_html", "")
        if lime_html:
            st.markdown("Detailed LIME:")
            with st.container():
                components.html(
                    lime_html,
                    height=500,  
                    scrolling=False,
                    width=800
                )

        # Display image explanations using Grad-CAM
        st.subheader("Image explanations")
        gradcam_b64 = result.get("gradcam_image")
        if gradcam_b64:
            header, b64 = gradcam_b64.split(",", 1) if gradcam_b64.startswith("data:") else (None, gradcam_b64)
            img_bytes = base64.b64decode(b64)
            st.image(img_bytes, use_container_width=True)
        else:
            st.write("No image explanation returned.")

        st.success("Done")