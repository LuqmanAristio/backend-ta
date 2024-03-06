from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from tensorflow.keras.models import load_model
from pytube import YouTube
from moviepy.editor import VideoFileClip
from django.http import HttpResponse, FileResponse
from y2mate_api import Handler

import pickle
import soundfile as sf
import warnings
import glob
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
        model_path = '/home/aristio170802/backendta/melatec/model/modelfinal2.h5'
        model = load_model(model_path)

        audio_file = request.FILES.get('audio_file')
        audio = AudioSegment.from_file(audio_file, format="wav")

        processed_audio = cut_and_limit_duration(audio, target_duration=120000)
        processed_audio_path = '/home/aristio170802/backendta/audio_cut.wav'
        processed_audio.export(processed_audio_path, format="wav")

        mfcc_result = extract_mfcc_from_audio(processed_audio_path)
        mfcc_result = {key: float(value) for key, value in mfcc_result.items()}
        mfcc_result_list = list(mfcc_result.values())

        normalisasi = '/home/aristio170802/backendta/melatec/model/parameter_normalisasi.pkl'

        with open(normalisasi, 'rb') as file:
            rata_rata_mfcc_loaded, standar_deviasi_mfcc_loaded = pickle.load(file)
        
        mfcc_data_normalized = (mfcc_result_list - rata_rata_mfcc_loaded) / standar_deviasi_mfcc_loaded
        
        input_model = [mfcc_data_normalized]
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
        temp_audio_filename = []

        api = Handler(youtube_url)
        for audio_metadata in api.run(format="mp3"):
            title = audio_metadata['title']
            temp_audio_filename.append(title)
        
        api.auto_save(format="mp3")
        folder_path = "/home/aristio170802/backendta/"

        mp3_files = glob.glob(f"{folder_path}/*.mp3")

        if mp3_files:
            audio_file = mp3_files[0]
            print(f"File MP3 ditemukan: {audio_file}")
        else:
            print("Tidak ada file MP3 yang ditemukan dalam folder.")

        print(" ==> Download audio completed, starting melody extraction")
        xxx = "/home/aristio170802/backendta/Perfect - Ed Sheeran & Beyoncé (Boyce Avenue acoustic cover) on Spotify & Apple 3G8CM-6BZC4_128.mp3"

        audio = AudioSegment.from_file(audio_file, format="mp3")
        audio = audio.set_channels(1).set_frame_rate(44100)
        sample_rate = audio.frame_rate
        audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)

        data = vamp.collect(audio_array, sample_rate, "mtg-melodia:melodia")
        hop, melody = data['vector']

        timestamps = 8 * 128/44100.0 + np.arange(len(melody)) * (128/44100.0)
        params = {"minfqr": 100.0, "maxfqr": 800.0, "voicing": 0.2, "minpeaksalience": 0.0}

        data = vamp.collect(audio_array, sample_rate, "mtg-melodia:melodia", parameters=params)
        hop, melody = data['vector']
        print("check")

        csv_name = 'temp_mel'
        csv_file_path = csv_name + '.csv'
        csv_file_path = os.path.join("/home/aristio170802/backendta", csv_file_path)

        print(" ==> Extraction completed, converting to .wav")
        
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(zip(timestamps, map(lambda x: x if x >= 0 else x, melody)))

        melosynth = "/home/aristio170802/backendta/melatec/melosynth.py"
        command = f'python3 "{melosynth}" "{csv_file_path}"'
        process = subprocess.Popen(command, shell=True)
        process.wait()

        process.communicate()

        print(" ==> Conversion completed, sending response")
        wav_file_path = "/home/aristio170802/backendta/temp_mel_melosynth.wav"

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
