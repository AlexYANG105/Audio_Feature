from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
from python_speech_features import mfcc
import tempfile
import os
from gammatone.gtgram import gtgram
from pydub import AudioSegment
import parselmouth
import whisper
import soundfile as sf
import noisereduce as nr
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

# Load Whisper model once at startup
whisper_model = whisper.load_model("base")

def compute_gtcc(y, sr, n_filters=20, win_time=0.025, hop_time=0.01):
    frame_length = int(win_time * sr)
    nfft = 2 ** (frame_length - 1).bit_length()
    if nfft < frame_length:
        nfft = frame_length
    gtg = gtgram(y, sr, win_time, hop_time, n_filters, nfft)
    log_gtg = np.log(gtg + 1e-8)
    gtcc_feat = librosa.feature.mfcc(S=log_gtg, n_mfcc=13)
    return gtcc_feat

def generate_waveform_image(y, sr):
    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(np.linspace(0, len(y)/sr, num=len(y)), y, color='#0074D9')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Waveform')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.read()
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode('utf-8')
    return img_b64

def generate_spectrogram_image(y, sr):
    plt.switch_backend('Agg')
    fig, ax = plt.subplots(figsize=(5, 2))
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    ax.set_title('Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img_bytes = buf.read()
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode('utf-8')
    return img_b64

def extract_features(y, sr, wav_path):
    # --- Transcription ---
    try:
        transcription_result = whisper_model.transcribe(wav_path)
        # This line for Cantonese
        # Transcription_result = whisper_model.transcribe(wav_path, language="zh")
        transcription = transcription_result["text"]
    except Exception as e:
        print("Whisper transcription error:", e)
        transcription = ""

    # --- Parselmouth features ---
    try:
        snd = parselmouth.Sound(wav_path)
        duration = snd.get_total_duration()
        pitch = snd.to_pitch()
        f0_values = pitch.selected_array['frequency']
        f0 = float(np.nan_to_num(np.mean(f0_values[f0_values > 0]))) if np.any(f0_values > 0) else 0
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        num_pulses = parselmouth.praat.call(point_process, "Get number of points")
        if point_process is not None and num_pulses > 1:
            try:
                jitter = float(np.nan_to_num(parselmouth.praat.call([snd, point_process], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)))
            except Exception as e:
                print("Jitter extraction error:", e)
                jitter = 0
            try:
                shimmer = float(np.nan_to_num(parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)))
            except Exception as e:
                print("Shimmer extraction error:", e)
                shimmer = 0
        else:
            print("PointProcess invalid or too few pulses for jitter/shimmer.")
            jitter = 0
            shimmer = 0
        formant_obj = snd.to_formant_burg()
        time = min(0.5, duration / 2)
        formants = [float(np.nan_to_num(parselmouth.praat.call(formant_obj, "Get value at time", i+1, time, 'Hertz', 'Linear'))) for i in range(4)]
    except Exception as e:
        print("Parselmouth error:", e)
        f0 = 0
        jitter = 0
        shimmer = 0
        formants = [0, 0, 0, 0]

    # --- MFCC, delta-MFCC ---
    try:
        mfcc_feat = mfcc(y, sr)
        mfcc_feat = np.nan_to_num(mfcc_feat)
        delta_mfcc = librosa.feature.delta(mfcc_feat.T).T
        delta_mfcc = np.nan_to_num(delta_mfcc)
    except Exception as e:
        print("MFCC extraction error:", e)
        mfcc_feat = np.zeros((1, 13))
        delta_mfcc = np.zeros((1, 13))

    # --- GTCC, delta-GTCC ---
    try:
        gtcc_feat = compute_gtcc(y, sr)
        gtcc_feat = np.nan_to_num(gtcc_feat)
        delta_gtcc = librosa.feature.delta(gtcc_feat)
        delta_gtcc = np.nan_to_num(delta_gtcc)
    except Exception as e:
        print("GTCC extraction error:", e)
        gtcc_feat = np.zeros((13, 1))
        delta_gtcc = np.zeros((13, 1))

    return {
        'f0': float(np.nan_to_num(f0)),
        'jitter': float(np.nan_to_num(jitter)),
        'shimmer': float(np.nan_to_num(shimmer)),
        'formants': [float(np.nan_to_num(f)) for f in formants],
        'mfcc': mfcc_feat.tolist(),
        'delta_mfcc': delta_mfcc.tolist(),
        'gtcc': gtcc_feat.tolist(),
        'delta_gtcc': delta_gtcc.tolist(),
        'transcription': transcription
    }

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    file = request.files['audio']
    waveform_image_b64 = request.form.get('waveform_image', None)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_in:
        file.save(tmp_in.name)
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        try:
            audio = AudioSegment.from_file(tmp_in.name)
            audio.export(tmp_wav.name, format='wav')
            y, sr = librosa.load(tmp_wav.name, sr=None)
        except Exception as e:
            os.unlink(tmp_in.name)
            os.unlink(tmp_wav.name)
            return jsonify({'error': f'Audio conversion failed: {str(e)}'}), 400

        # --- 1. Analyze raw audio ---
        features_raw = extract_features(y, sr, tmp_wav.name)
        spectrogram_image = generate_spectrogram_image(y, sr)
        # Optionally, you can also generate the waveform image in Python:
        if waveform_image_b64 is None:
            waveform_image_b64 = generate_waveform_image(y, sr)

        # --- 2. Noise reduction ---
        try:
            y_denoised = nr.reduce_noise(y=y, sr=sr)
            tmp_wav_denoised = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(tmp_wav_denoised.name, y_denoised, sr)
            features_denoised = extract_features(y_denoised, sr, tmp_wav_denoised.name)
            os.unlink(tmp_wav_denoised.name)
        except Exception as e:
            print("Noise reduction error:", e)
            features_denoised = {k+'_denoised': 0 for k in features_raw.keys()}

        os.unlink(tmp_in.name)
        os.unlink(tmp_wav.name)

    return jsonify({
        'raw': features_raw,
        'denoised': features_denoised,
        'waveform_image': waveform_image_b64,
        'spectrogram_image': spectrogram_image
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)