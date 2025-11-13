# streamlit_app.py
import streamlit as st
import requests
import base64
import os
import streamlit.components.v1 as components

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Multimodal Fake News Detector", layout="centered")

st.title("Multimodal Fake-News Detector with XAI")

with st.form("input_form"):
    text = st.text_area("News text / headline", height=120)
    uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    lime_samples = st.slider("LIME samples (image explainer)", min_value=50, max_value=1000, value=200, step=50)
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not text or not uploaded_file:
        st.warning("Please provide both text and an image.")
    else:
        # send to backend
        files = {
            "image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        data = {"text": text, "lime_samples": str(lime_samples)}
        with st.spinner("Sending to backend and waiting for results..."):
            try:
                resp = requests.post(API_URL+'/predict', data=data, files=files, timeout=600)
                resp.raise_for_status()
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.stop()

        result = resp.json()

        # display original
        st.subheader("Input")
        st.write("Text:")
        st.write(result.get("text"))
        st.write("Model prediction:")
        label = result.get("label")
        confidence = result.get("confidence")
        st.markdown(f"**{label.upper()}**  (confidence: {confidence:.3f})")

        # show original uploaded image
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

        # Show XAI outputs
        st.subheader("Text explanation (LIME)")
        lime_list = result.get("lime_text_list", [])
        if lime_list:
            st.write("Top tokens and weights (local surrogate):")
            for token, weight in lime_list:
                st.write(f"{token} → {weight:.4f}")

        lime_html = result.get("lime_text_html", "")
        if lime_html:
            st.markdown("Detailed LIME (HTML):")
            components.html(lime_html, height=200, scrolling=True)

        st.subheader("Image explanations")
        gradcam_b64 = result.get("gradcam_image")
        if gradcam_b64:
            # decode and show
            header, b64 = gradcam_b64.split(",", 1) if gradcam_b64.startswith("data:") else (None, gradcam_b64)
            img_bytes = base64.b64decode(b64)
            st.image(img_bytes, caption="Grad-CAM + LIME overlay (combined view)", use_container_width=True)
        else:
            st.write("No image explanation returned.")

        st.success("Done")
