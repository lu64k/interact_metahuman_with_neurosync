# utils/local_tts.py
import requests
import dashscope
from dashscope.audio.tts import SpeechSynthesizer
from dashscope.audio.tts_v2 import *
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
import numpy as np
import io
import requests
import soundfile as sf  # 导入 soundfile 库并使用 sf 别名
from flask import Flask, request, jsonify
import requests
from config import LOCAL_TTS_URL
from utils.audio.convert_audio import pcm_to_wav, audio_to_bytes, convert_to_wav, bytes_to_wav, is_valid_audio
from utils.utils import init_api_key
import os
import asyncio


def call_local_tts(text, voice=None): 
    """
    Calls the local TTS Flask endpoint to generate speech for the given (already-cleaned) text.
    Optionally, a voice can be specified.
    Returns the audio bytes if successful, otherwise returns None.
    """
    payload = {"text": text}

    if voice is not None:
        payload["voice"] = voice

    try:
        response = requests.post(LOCAL_TTS_URL, json=payload)
        response.raise_for_status()
        return response.content
    except Exception as e:
        # Optionally log error here
        return None


def synthesize_speech_cosy_cove_streaming(text):

    if 'DASHSCOPE_API_KEY' in os.environ:#DASHSCOPE_API_KEY是你存放key的系统变量名
        dashscope.api_key = os.environ['DASHSCOPE_API_KEY']        
    else:
        dashscope.api_key = '输入你的api'
    
    if text:
        model='cosyvoice-v2'
        voice="cosyvoice-v2-prefix-8165fbaef5b34306bbf6d1edcf3ed705"
        format= AudioFormat.PCM_48000HZ_MONO_16BIT
        try:
            # 生成 TTS 音频数据
            synthesizer=SpeechSynthesizer(model=model, voice=voice, format=format)
            result = synthesizer.streaming_call(text)
            
            if result is None is None:
                print("❌ No audio data returned from TTS API")
                return jsonify({'error': 'Failed to generate audio, no data returned'}), 500

            audio_bytes = result
            audio_file = io.BytesIO(audio_bytes)
            
            wavio = pcm_to_wav(audio_bytes, sample_rate=48000, channels=1, sample_width=2)
            print("got audio bytes")
            
            return wavio
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'error': 'Exception occurred', 'details': str(e)}), 500
    elif text == "llm finished":
        synthesizer=SpeechSynthesizer(model=model, voice=voice, format=format)
        synthesizer.streaming_complete()
    return ("finished")
halsin = "cosyvoice-v2-halsin-cb27968ead8241b7acd9941b86a37d53"
cove = "cosyvoice-v2-prefix-8165fbaef5b34306bbf6d1edcf3ed705"        
from flask import current_app
app = Flask(__name__)  
dashscope.api_key = init_api_key(env_name= 'DASHSCOPE_API_KEY', input_api_key="输入你的api")
async def synthesize_speech(text):
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        # 使用 Dashscope TTS v2 进行语音合成
        model = "cosyvoice-v2"  # 替换为您使用的模型名称
        voice = cove  # 替换为您选择的语音名称
        format = AudioFormat.PCM_22050HZ_MONO_16BIT
        speech_rate = 1.30
        synthesizer = SpeechSynthesizer(model=model, voice=voice, format=format, speech_rate=speech_rate)
        audio = await asyncio.to_thread (synthesizer.call,text)
        print('requestId: ', synthesizer.get_last_request_id())
        #duration = get_audio_length(audio)
        return audio
    except Exception as e:
        print(f"TTS synthesis exception: {e}")
        return jsonify({'error': 'TTS synthesis failed', 'details': str(e)}), 500
        
async def call_TTS(output_text):
    if not output_text:
        return None
    try:
        print(f"tts received {output_text}")
        with app.app_context():
            raw_audio = await synthesize_speech(output_text)  # 已异步，直接await
        if raw_audio:
            print("got audio")
            wav_bytes = await asyncio.to_thread(pcm_to_wav, raw_audio)
            return wav_bytes.getvalue()
        else:
            print("failed to get audio output")
            return None
    except Exception as e:
        print(f"call tts error: {e}")
        return None