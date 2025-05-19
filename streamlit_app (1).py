import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

# Параметры модели
SR = 11025
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 40
FMIN = 20
FMAX = 5000
FRAME_LENGTH = 256
BPM_RANGE = range(30, 286)

# Загрузка модели
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.h5')

model = load_model()

# Инициализация LabelEncoder
le = LabelEncoder()
le.fit(list(BPM_RANGE))

# Функции обработки аудио (из вашего кода)
def extract_mel_spectrogram(file_path, augment=False):
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)
        required_samples = FRAME_LENGTH * HOP_LENGTH
        if len(y) < required_samples:
            y = np.pad(y, (0, max(0, required_samples - len(y))), mode='constant')
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX, window='hamming'
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db, 1.0
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, 1.0

def get_spectrogram_windows(mel_spec, frame_length=FRAME_LENGTH, hop_size=128):
    windows = []
    for start in range(0, mel_spec.shape[1] - frame_length + 1, hop_size):
        window = mel_spec[:, start:start + frame_length]
        if window.shape[1] == frame_length:
            windows.append(window)
    return np.array(windows)

def estimate_global_tempo(model, mel_spec, le):
    windows = get_spectrogram_windows(mel_spec)
    if len(windows) == 0:
        return None
    windows = windows[..., np.newaxis]
    windows = (windows - windows.min()) / (windows.max() - windows.min())
    predictions = model.predict(windows)
    avg_predictions = np.mean(predictions, axis=0)
    tempo_class = np.argmax(avg_predictions)
    return le.inverse_transform([tempo_class])[0]

# Streamlit интерфейс
st.title("Music Tempo Detection")
st.write("Upload an audio file to detect its tempo (BPM).")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Сохранение файла
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Обработка и предсказание
    with st.spinner("Processing..."):
        mel_spec, _ = extract_mel_spectrogram(file_path)
        if mel_spec is not None:
            tempo = estimate_global_tempo(model, mel_spec, le)
            if tempo is not None:
                st.success(f"Predicted Tempo: {tempo} BPM")
            else:
                st.error("Could not estimate tempo.")
        else:
            st.error("Error processing audio file.")

    # Удаление временного файла
    os.remove(file_path)