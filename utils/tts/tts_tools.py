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
import uuid
from utils.logging_config import get_logger

logger = get_logger(__name__)


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
                logger.error("No audio data returned from TTS API")
                return jsonify({'error': 'Failed to generate audio, no data returned'}), 500

            audio_bytes = result
            audio_file = io.BytesIO(audio_bytes)
            
            wavio = pcm_to_wav(audio_bytes, sample_rate=48000, channels=1, sample_width=2)
            logger.info("Got audio bytes")
            
            return wavio
        except Exception as e:
            logger.exception("TTS streaming error: %s", e)
            return jsonify({'error': 'Exception occurred', 'details': str(e)}), 500
    elif text == "llm finished":
        synthesizer=SpeechSynthesizer(model=model, voice=voice, format=format)
        synthesizer.streaming_complete()
    return ("finished")
halsin = "cosyvoice-v2-halsin-cb27968ead8241b7acd9941b86a37d53"
cove = "cosyvoice-v2-prefix-8165fbaef5b34306bbf6d1edcf3ed705" #推荐语速1.2-1.3
henry = "cosyvoice-v3.5-plus-henry-2c9e9ac9bb7c443a934a46aef36d69ab" 
henry_flash = "cosyvoice-v3.5-flash-henry-8561249abf224db3b195f8399ff69fac"       
henry_v2 = "cosyvoice-v2-henry2-2681c1bf3d51401fa8bb8bd7c2569fdf" 
henry_v3 = "cosyvoice-v3-plus-henry-a833a4c8ff814446bcc82ca432410fb5" 
from flask import current_app
app = Flask(__name__)  
dashscope.api_key = init_api_key(env_name= 'DASHSCOPE_API_KEY', input_api_key="输入你的api")
async def synthesize_speech(text):
    if not text:
        return None, None  # ✅ 返回 (audio, request_id)
    try:
        # 使用 Dashscope TTS v2 进行语音合成
        model = "cosyvoice-v2"  # 替换为您使用的模型名称
        voice = cove # 为您选择的语音名称
        format = AudioFormat.PCM_22050HZ_MONO_16BIT
        speech_rate = 1.2  # 可选：调整语速，默认为 1.0
        synthesizer = SpeechSynthesizer(model=model, voice=voice, format=format, speech_rate=speech_rate)
        audio = await asyncio.to_thread (synthesizer.call,text)
        
        # ✅ 尝试获取 request_id，如果失败则生成 UUID
        request_id = None
        try:
            request_id = synthesizer.get_last_request_id()
        except:
            pass
            
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())
            logger.warning("Could not get request_id from TTS, generated UUID: %s", request_id)
        else:
            logger.info("requestId: %s", request_id)
            
        # ✅ 返回音频和 request_id
        return audio, request_id
    except Exception as e:
        logger.exception("TTS synthesis exception: %s", e)
        return None, None
        
async def call_TTS(output_text):
    if not output_text:
        return None, None  # ✅ 返回 (audio, request_id)
    try:
        logger.info("TTS received text: %s", output_text)
        # ✅ [FIX] 移除 app.app_context()，因为我们不在 Flask 请求上下文中运行
        raw_audio, request_id = await synthesize_speech(output_text)  # ✅ 接收两个返回值
            
        # ✅ 再次确保有 request_id
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())
            
        if raw_audio:
            logger.info("Got audio from TTS provider")
            wav_bytes = await asyncio.to_thread(pcm_to_wav, raw_audio)
            return wav_bytes.getvalue(), request_id  # ✅ 返回音频和 request_id
        else:
            logger.warning("Failed to get audio output")
            return None, None
    except Exception as e:
        logger.exception("call_TTS error: %s", e)
        return None, None