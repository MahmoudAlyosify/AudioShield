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
import cv2

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model (5-22).h5")

model = load_model()
class_names = ['Fake', 'Real']

# Convert audio to spectrogram and return as image path
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

# Prepare image for prediction
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Generate PDF Report
def generate_pdf_result(pred_label, confidence, spectrogram_path, audio_filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="AudioShield - Fake Audio Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Prediction: {pred_label}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(200, 10, txt=f"File: {audio_filename}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.image(spectrogram_path, x=10, y=70, w=180)
    pdf_path = spectrogram_path.replace(".png", "_report.pdf")
    pdf.output(pdf_path)
    return pdf_path

def generate_grad_cam(model, image_path, class_index):
    img = preprocess_image(image_path)[0]
    img_input = np.expand_dims(img, axis=0)

    # تمرير input لتفعيل النموذج
    _ = model.predict(img_input)

    # تحديد اسم آخر طبقة Convolution
    last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]

    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_input)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    conv_outputs = conv_outputs[0]
    for i in range(pooled_grads.shape[0]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # تحميل الصورة الأصلية
    img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    grad_cam_path = image_path.replace(".png", "_gradcam.png")
    cv2.imwrite(grad_cam_path, superimposed_img)
    return grad_cam_path

# Streamlit UI
st.title("AudioShield - Fake Audio Detection")
st.markdown("Upload a **.wav** audio file to classify as Real or Fake.")

uploaded_file = st.file_uploader("Upload Audio (.wav)", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_audio_path = tmp_file.name

    spectrogram_path = audio_to_spectrogram_image(tmp_audio_path)
    st.image(spectrogram_path, caption="Generated Spectrogram", use_column_width=True)

    if st.button("Analyze"):
        input_img = preprocess_image(spectrogram_path)
        pred = model.predict(input_img)
        class_index = np.argmax(pred)
        confidence = pred[0][class_index]

        st.write(f"### Prediction: **{class_names[class_index]}**")
        st.write(f"Confidence: {confidence * 100:.2f}%")

        # Grad-CAM
        grad_cam_img = generate_grad_cam(model, spectrogram_path, class_index)
        st.image(grad_cam_img, caption="Grad-CAM: What the model focused on", use_column_width=True)

        # PDF
        pdf_path = generate_pdf_result(class_names[class_index], confidence * 100, spectrogram_path, uploaded_file.name)
        st.download_button(label="📄 Download PDF Report", data=open(pdf_path, "rb"), file_name="AudioShield_Report.pdf")

    # Cleanup
    os.remove(tmp_audio_path)
