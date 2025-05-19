from flask import Flask, request, render_template, jsonify
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import scipy.interpolate
from sklearn.preprocessing import LabelEncoder

# Параметры из вашей модели
SR = 11025
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 40
FMIN = 20
FMAX = 5000
FRAME_LENGTH = 256
BPM_RANGE = range(30, 286)
NUM_CLASSES = len(BPM_RANGE)

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Создание папки для загрузок, если она не существует
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Загрузка модели
model = load_model('model.h5')  # Укажите путь к вашей сохраненной модели

# Инициализация LabelEncoder
le = LabelEncoder()
le.fit(list(BPM_RANGE))

# Функция извлечения Mel-спектограммы (из вашего кода)
def extract_mel_spectrogram(file_path, augment=False):
    try:
        # Загрузка аудио
        y, sr = librosa.load(file_path, sr=SR, mono=True)
        
        # Проверка, что y не None и является массивом
        if y is None or not isinstance(y, np.ndarray):
            print(f"Ошибка: аудиофайл {file_path} не загружен, y is {y}")
            return None, 1.0
        
        # Проверка длины аудио
        required_samples = FRAME_LENGTH * HOP_LENGTH
        if len(y) < required_samples:
            y = np.pad(y, (0, max(0, required_samples - len(y))), mode='constant')
        
        scale_factor = 1.0
        # Извлечение Mel-спектограммы
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX, window='hamming'
        )
        # Логарифмическая нормализация
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db, scale_factor
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")
        return None, 1.0

# Функция для получения окон спектограммы (из вашего кода)
def get_spectrogram_windows(mel_spec, frame_length=FRAME_LENGTH, hop_size=128):
    windows = []
    for start in range(0, mel_spec.shape[1] - frame_length + 1, hop_size):
        window = mel_spec[:, start:start + frame_length]
        if window.shape[1] == frame_length:
            windows.append(window)
    return np.array(windows)

# Функция глобальной оценки темпа (из вашего кода)
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Извлечение Mel-спектограммы
        mel_spec, _ = extract_mel_spectrogram(file_path)
        if mel_spec is None:
            os.remove(file_path)
            return jsonify({'error': 'Failed to process audio file'}), 500
        
        # Оценка BPM
        bpm = estimate_global_tempo(model, mel_spec, le)
        os.remove(file_path)  # Удаление файла после обработки
        
        if bpm is None:
            return jsonify({'error': 'Unable to estimate BPM'}), 500
        
        return jsonify({'bpm': int(bpm)})

if __name__ == '__main__':
    app.run(debug=True)