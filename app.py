import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os
import tensorflow as tf
from fpdf import FPDF
from datetime import datetime

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_model()
class_names = ['Fake', 'Real']

results_log = []

# Convert audio to spectrogram
def audio_to_spectrogram_image(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)

    fig = plt.figure(figsize=(3, 3))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp_img.name, bbox_inches='tight', pad_inches=0)
    plt.close()
    return tmp_img.name

# Preprocess for model
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Single file PDF report
def generate_pdf_result(label, spectrogram_path, audio_filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="AudioShield - Fake Audio Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction: {label}", ln=True)
    pdf.cell(200, 10, txt=f"File: {audio_filename}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.image(spectrogram_path, x=10, y=70, w=180)
    pdf_path = spectrogram_path.replace(".png", "_report.pdf")
    pdf.output(pdf_path)
    return pdf_path

# Summary PDF for all results
def generate_summary_pdf(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="AudioShield - Summary Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for r in results:
        pdf.cell(200, 10, txt=f"{r['filename']} --> {r['prediction']}", ln=True)
    tmp_report = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp_report.name)
    return tmp_report.name

# UI
st.title("AudioShield - Fake Audio Detection")
st.markdown("Upload one or more **.wav** audio files to classify as Real or Fake.")

uploaded_files = st.file_uploader("Upload Audio Files (.wav)", type=["wav"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        audio_bytes = uploaded_file.read()

        st.subheader(f"🎧 {uploaded_file.name}")
        st.audio(audio_bytes, format="audio/wav")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_audio_path = tmp_file.name

        spectrogram_path = audio_to_spectrogram_image(tmp_audio_path)
        st.image(spectrogram_path, caption="Generated Spectrogram", use_column_width=True)

        input_img = preprocess_image(spectrogram_path)
        pred = model.predict(input_img)
        fake_prob = pred[0][0]  # index 0 = Fake
        

        if fake_prob >= 0.99:
            final_label = "Real"
        else:
            final_label = "Fake"

        st.write(f"### Prediction: **{final_label}**")

        # PDF (single report)
        pdf_path = generate_pdf_result(final_label, spectrogram_path, uploaded_file.name)
        st.download_button("📄 Download This Report", open(pdf_path, "rb"), file_name=f"{uploaded_file.name}_Report.pdf")

        results_log.append({
            "filename": uploaded_file.name,
            "prediction": final_label
        })

        os.remove(tmp_audio_path)

# Summary PDF button
if results_log:
    summary_path = generate_summary_pdf(results_log)
    st.download_button("📄 Download All Results", open(summary_path, "rb"), file_name="AudioShield_Summary.pdf")
