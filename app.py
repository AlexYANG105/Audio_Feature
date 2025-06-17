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

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    file = request.files['audio']
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

        # --- Transcription ---
        try:
            transcription_result = whisper_model.transcribe(tmp_wav.name)
            transcription = transcription_result["text"]
        except Exception as e:
            transcription = ""

        # --- Parselmouth features ---
        try:
            snd = parselmouth.Sound(tmp_wav.name)
            # Fundamental frequency (F0)
            pitch = snd.to_pitch()
            f0_values = pitch.selected_array['frequency']
            f0 = float(np.mean(f0_values[f0_values > 0])) if np.any(f0_values > 0) else 0
            # Jitter & Shimmer
            point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
            jitter = parselmouth.praat.call([snd, point_process], "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            # Formants (first 4 at 0.5s)
            formant_obj = snd.to_formant_burg()
            duration = snd.get_total_duration()
            time = min(0.5, duration / 2)
            formants = [parselmouth.praat.call(formant_obj, "Get value at time", i+1, time, 'Hertz', 'Linear') for i in range(4)]
        except Exception as e:
            f0 = 0
            jitter = 0
            shimmer = 0
            formants = [0, 0, 0, 0]

        # --- MFCC, delta-MFCC ---
        mfcc_feat = mfcc(y, sr)
        delta_mfcc = librosa.feature.delta(mfcc_feat.T).T

        # --- GTCC, delta-GTCC ---
        gtcc_feat = compute_gtcc(y, sr)
        delta_gtcc = librosa.feature.delta(gtcc_feat)

        os.unlink(tmp_in.name)
        os.unlink(tmp_wav.name)

    return jsonify({
        'f0': f0,
        'jitter': jitter,
        'shimmer': shimmer,
        'formants': formants,
        'mfcc': mfcc_feat.tolist(),
        'delta_mfcc': delta_mfcc.tolist(),
        'gtcc': gtcc_feat.tolist(),
        'delta_gtcc': delta_gtcc.tolist(),
        'transcription': transcription
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)