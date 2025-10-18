import streamlit as st
from main import process_profile
import os, tempfile

st.title("📄 AI CV Generator")
st.write("Upload a JSON profile to automatically generate a professional CV using LLMs.")

uploaded_file = st.file_uploader("Upload your JSON profile", type=["json"])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Processing your CV... please wait ⏳")
    output_path = process_profile(tmp_path)
    st.success("✅ CV generated successfully!")

    with open(output_path, "rb") as f:
        st.download_button("⬇️ Download CV", f, file_name="Generated_CV.docx")
