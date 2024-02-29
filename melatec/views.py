from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from tensorflow.keras.models import load_model
from pytube import YouTube
from moviepy.editor import VideoFileClip
import os
import tempfile
import numpy as np
import json

@csrf_exempt
@require_POST
def predict_view(request):
    try:
        model_path = 'melatec\model\modelfinal2.h5'
        model = load_model(model_path)

        input_data = json.loads(request.body.decode('utf-8')).get('input_data')
        data_uji = np.expand_dims(input_data, axis=1)

        prediction = model.predict(data_uji)
        prediction_list = prediction.tolist()

        return JsonResponse({'prediction': prediction_list})

    except Exception as e:
        return JsonResponse({'error': str(e)})

def youtube_to_melody(request):
    try:
        youtube_url = request.POST.get('youtube_url')

        video = YouTube(youtube_url)
        video_stream = video.streams.filter(only_audio=True).first()
        
        temp_file_path = os.path.join(tempfile.gettempdir(), 'temp_audio.mp4')

        video_stream.download(temp_file_path)

        wav_file_path = os.path.join(tempfile.gettempdir(), 'output_audio.wav')

        audio_clip = VideoFileClip(temp_file_path)
        audio_clip.audio.write_audiofile(wav_file_path, codec='pcm_s16le', fps=44100)

        os.remove(temp_file_path)

        return JsonResponse({'wav_file_path': wav_file_path})

    except Exception as e:
        return JsonResponse({'error': str(e)})