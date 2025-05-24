# AudioShield: Deepfake Voice Detection

**Voice Integrity. Audio Security.**

AudioShield is an AI-powered tool that detects deepfake or tampered audio using spectrogram analysis and deep learning. It helps users and organizations verify the authenticity of voice recordings in critical settings such as call centers, financial services, and virtual meetings.

---

## 📑 Table of Contents

- [Features](#features)
- [Project Video](#project-video)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Usage](#usage)
- [Dataset](#dataset)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Team](#team)

---

## 🚀 Features

- Detects AI-generated and synthetic voices
- Classifies audio as **Real** or **Fake** with a confidence score
- Converts audio to mel-spectrograms using consistent preprocessing
- Displays Grad-CAM heatmaps to explain model focus
- Generates downloadable PDF reports with visual results
- Built with a simple and intuitive Streamlit interface

---

## 🎥 Project Video

This is a brief video explaining the motivation, design, and implementation of AudioShield:

https://github.com/user-attachments/assets/8686ceb7-0578-41b2-a39c-34e4a52245e3

---

## 🎥 Run Demo

This is a brief video explaining the motivation, design, and implementation of AudioShield:

https://github.com/user-attachments/assets/a3c27aa0-a010-44d6-8d40-b52b279aa711

---

## 🧠 How It Works

1. **Upload Audio**  
   Accepts `.wav` or `.mp3` files from the user.

2. **Preprocessing**  
   Audio is converted to a 224×224 mel-spectrogram using `librosa`.

3. **Classification**  
   A trained CNN model classifies the input as Real or Fake.

4. **Explainability**  
   Grad-CAM heatmaps show where the model focused when making its decision.

5. **Reporting**  
   The result, confidence, and spectrogram image are saved into a downloadable PDF.

---

## ⚙️ Tech Stack

| Layer         | Technology             |
|---------------|-------------------------|
| Language      | Python                  |
| spectrograms  | Librosa, OpenCV         |
| ML / DL       | TensorFlow / Keras      |
| Interface     | Streamlit               |
| Explainability| Grad-CAM                |
| Reports       | FPDF                    |

---

## 💻 Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/MahmoudAlyosify/AudioShield.git
   cd AudioShield


2. Install required packages:

   ```bash
   pip install -r requirements.txt


3. Run the app:

   ```bash
   streamlit run app.py


---

## 🗂 Dataset

The AudioShield model was trained on **81,000 mel-spectrogram images** generated from real and fake audio:

* **Training**:
  30,000 real + 30,000 fake

* **Validation**:
  8,000 real + 8,000 fake

* **Testing**:
  2,500 real + 2,500 fake

You can access the dataset from:

* **Kaggle**:
  [AudioShield - Fake vs Real Spectrogram Dataset](https://www.kaggle.com/datasets/mahmoudalyosify/audioshield-fake-real-audio-spectrogram-dataset)

* **Hugging Face**:
  [AudioShield Dataset on Hugging Face](https://huggingface.co/datasets/mahmoudalyosify/AudioShield_Fake_Real_Audio_Spectrogram_Dataset)

---

## 🧭 Future Improvements

* Real-time detection of voice deepfakes during calls
* Support for multilingual and accented audio
* Cloud deployment and REST API
* Integration with messaging and social platforms
* Semantic verification using speech-to-text and NLP

---

## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0**.

---

## 👥 Team

| Name     | Role                |
| -------- | ------------------- |
| Mahmoud  |  ML & Development   |
| Emad     |  ML & Development   |
| Ahmed    |  ML & Development   |
| Elsayed  |  ML & Development   |
| Abdullah |  ML & Development   |


