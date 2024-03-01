from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from tensorflow.keras.models import load_model
from pytube import YouTube
from moviepy.editor import VideoFileClip
from django.http import HttpResponse, FileResponse

import os
import tempfile
import numpy as np
import json
import csv
import subprocess
import librosa
import vamp
from pydub import AudioSegment
from pydub.silence import detect_silence
import mimetypes

@csrf_exempt
@require_POST
def predict_view(request):
    try:
        model_path = '/home/luqmanaristio/Tugas Akhir/backend-ta/melatec/model/modelfinal2.h5'
        model = load_model(model_path)

        audio_file = request.FILES.get('audio_file')
        audio = AudioSegment.from_file(audio_file, format="wav")

        processed_audio = cut_and_limit_duration(audio, target_duration=120000)
        processed_audio_path = '/home/luqmanaristio/Tugas Akhir/backend-ta/audio_cut.wav'
        processed_audio.export(processed_audio_path, format="wav")

        mfcc_result = extract_mfcc_from_audio(processed_audio_path)
        mfcc_result = {key: float(value) for key, value in mfcc_result.items()}
        mfcc_result_list = list(mfcc_result.values())

        input_model = [mfcc_result_list]
        data_uji = np.expand_dims(input_model, axis=0)

        prediction = model.predict(data_uji)
        prediction_list = prediction.tolist()

        return JsonResponse({'prediction': prediction_list})

    except Exception as e:
        return JsonResponse({'error': str(e)})

def cut_and_limit_duration(audio, target_duration=120000):  
    silence_ranges = detect_silence(audio, silence_thresh=-40, seek_step=1)

    if silence_ranges:
        start_time = silence_ranges[0][1]
        trimmed_audio = audio[start_time:]

        return trimmed_audio[:target_duration]

    else:
        print("Tidak ada bagian hening di awal yang terdeteksi.")
        return audio[:target_duration]

def extract_mfcc_from_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        mfcc_mean, mfcc_var = calculate_mfcc(y, sr)

        mfcc_result = {}
        for i in range(1, 21):
            mfcc_result[f'Mean_MFCC_{i}'] = mfcc_mean[i-1]
            mfcc_result[f'Var_MFCC_{i}'] = mfcc_var[i-1]

        return mfcc_result

    except Exception as e:
        y, sr = librosa.load(audio_path, sr=None)
        return {'error': str(e)}

def calculate_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mean_values = [mfccs[i, :].mean() for i in range(20)]
    variance_values = [mfccs[i, :].var() for i in range(20)]

    return mean_values, variance_values

@csrf_exempt
@require_POST
def youtube_to_melody(request):
    try:
        youtube_url = request.POST.get('youtube_url')
        temp_audio_filename = 'temp_audio.mp3'

        y2mate_command = f'y2mate -f mp3 {youtube_url} -o {temp_audio_filename}'
        subprocess.run(y2mate_command, shell=True, check=True)

        print(" ==> Download audio completed, starting melody extraction")

        audio_file = "/home/luqmanaristio/Tugas Akhir/backend-ta/temp_audio.mp3"
        audio, sr = librosa.load(audio_file, sr=44100, mono=True)

        data = vamp.collect(audio, sr, "mtg-melodia:melodia")
        hop, melody = data['vector']

        timestamps = 8 * 128/44100.0 + np.arange(len(melody)) * (128/44100.0)
        params = {"minfqr": 100.0, "maxfqr": 800.0, "voicing": 0.2, "minpeaksalience": 0.0}

        data = vamp.collect(audio, sr, "mtg-melodia:melodia", parameters=params)
        hop, melody = data['vector']

        csv_file_path = os.path.splitext(temp_audio_filename)[0] + '.csv'
        csv_file_path = os.path.join("/home/luqmanaristio/Tugas Akhir/backend-ta/", csv_file_path)

        print(" ==> Extraction completed, converting to .wav")
        
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(zip(timestamps, map(lambda x: x if x >= 0 else x, melody)))

        melosynth = "/home/luqmanaristio/Tugas Akhir/backend-ta/melatec/melosynth.py"
        command = f'python3 "{melosynth}" "{csv_file_path}"'
        process = subprocess.Popen(command, shell=True)
        process.wait()

        process.communicate()

        print(" ==> Conversion completed, sending response")
        wav_file_path = "/home/luqmanaristio/Tugas Akhir/backend-ta/temp_audio_melosynth.wav"

        if os.path.exists(wav_file_path):
            with open(wav_file_path, 'rb') as wav_file:

                mime_type, _ = mimetypes.guess_type(wav_file_path)
                response = HttpResponse(wav_file.read(), content_type=mime_type)
                response['Content-Disposition'] = f'attachment; filename="{temp_audio_filename}.wav"'
                os.remove(wav_file_path)
                os.remove(csv_file_path)
                os.remove(audio_file)
                return response
        else:
            return JsonResponse({'error': 'WAV file not found'})

    except subprocess.CalledProcessError as e:
        return JsonResponse({'error': f'Error running subprocess: {str(e)}'})

    except Exception as e:
        return JsonResponse({'error': str(e)})

