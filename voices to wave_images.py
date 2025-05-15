import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


input_dir = "E:/voices"

output_dir = "E:/img"


os.makedirs(output_dir, exist_ok=True)


sr = 22050
n_fft = 2048
hop_length = 512
n_mels = 128


for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        audio_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_path = os.path.join(output_dir, output_filename)

        try:

            y, _ = librosa.load(audio_path, sr=sr)

            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            S_DB = librosa.power_to_db(S, ref=np.max)


            plt.figure(figsize=(20, 15), dpi=150)
            librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, cmap='magma')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"✅ Saved: {output_path}")

        except Exception as e:
            print(f"❌ Failed to process {audio_path}: {e}")
