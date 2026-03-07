"""
play_audio.py
-----------------
This module provides functions to play audio using Pygame. It includes
helper functions for initializing the mixer and unified playback loops.
It also supports audio conversion on the fly (e.g. raw PCM to WAV) where needed.
"""

import io
import time
import pygame
from utils.logging_config import get_logger
from utils.audio.convert_audio import convert_to_wav
from pydub import AudioSegment
AudioSegment.ffmpeg = "path/to/ffmpeg"
logger = get_logger(__name__)
# --- Helper Functions ---

def init_pygame_mixer():
    """
    Initialize the Pygame mixer only once.
    """
    if not pygame.mixer.get_init():
        pygame.mixer.init()


def sync_playback_loop():
    """
    A playback loop that synchronizes elapsed time with the music position.
    """
    start_time = time.perf_counter()
    clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
        elapsed_time = time.perf_counter() - start_time
        current_pos = pygame.mixer.music.get_pos() / 1000.0  # convert ms to sec

        # If behind, sleep briefly; if ahead, let it catch up.
        if elapsed_time > current_pos:
            time.sleep(0.01)
        elif elapsed_time < current_pos:
            continue
        clock.tick(10)


def simple_playback_loop(stop_flag_getter=None):
    """
    A simple playback loop that just ticks the clock until playback finishes.
    
    Args:
        stop_flag_getter: Optional callable that returns True if playback should be interrupted
    """
    clock = pygame.time.Clock()
    while pygame.mixer.music.get_busy():
        # ✅ 检查是否需要打断
        if stop_flag_getter and stop_flag_getter():
            logger.warning("[播放中断] 检测到打断信号，立即停止音频")
            pygame.mixer.music.stop()
            break
        clock.tick(10)
    
    # ✅ 确保音频完全停止后再返回
    # 有时候 get_busy() 返回 False 但音频缓冲区还没完全清空
    # 增加一个小延迟确保下一句不会重叠
    time.sleep(0.1)


# --- Playback Functions ---

def play_audio_bytes(audio_bytes, start_event, sync=True):
    """
    Play audio from raw bytes.
    
    Parameters:
      - audio_bytes: audio data as bytes.
      - start_event: threading.Event to wait for before starting playback.
      - sync: if True, uses time-syncing playback loop.
    """
    try:
        init_pygame_mixer()
        audio_file = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(audio_file)
        start_event.wait()  # Wait for the signal to start
        pygame.mixer.music.play()
        if sync:
            sync_playback_loop()
        else:
            simple_playback_loop()
    except pygame.error as e:
        logger.exception("Error in play_audio_bytes: %s", e)


def play_audio_from_memory(audio_data, start_event, sync=False, stop_flag_getter=None):
    """
    Play audio from memory (assumes valid WAV bytes).
    Uses a simple playback loop.
    
    Args:
        audio_data: Audio bytes in WAV format
        start_event: Event to wait for before starting playback
        sync: Whether to use sync playback (default False)
        stop_flag_getter: Optional callable that returns True if playback should be interrupted
    """
    try:
        init_pygame_mixer()
        audio_file = io.BytesIO(audio_data)
        pygame.mixer.music.load(audio_file)
        start_event.wait()
        pygame.mixer.music.play()
        simple_playback_loop(stop_flag_getter)
    except pygame.error as e:
        if "Unknown WAVE format" in str(e):
            logger.warning("Unknown WAVE format encountered. Skipping to the next item in the queue.")
        else:
            logger.exception("Error in play_audio_from_memory: %s", e)
    except Exception as e:
        logger.exception("Error in play_audio_from_memory: %s", e)


def play_audio_from_path(audio_path, start_event, sync=True):
    """
    Play audio from a file path. If the format is unsupported,
    automatically convert it to WAV.
    """
    try:
        init_pygame_mixer()
        try:
            pygame.mixer.music.load(audio_path)
        except pygame.error:
            logger.warning("Unsupported format for %s. Converting to WAV.", audio_path)
            audio_path = convert_to_wav(audio_path)
            pygame.mixer.music.load(audio_path)
        start_event.wait()
        pygame.mixer.music.play()
        if sync:
            sync_playback_loop()
        else:
            simple_playback_loop()
    except pygame.error as e:
        logger.exception("Error in play_audio_from_path: %s", e)


def read_audio_file_as_bytes(file_path):
    """
    Read a WAV audio file from disk as bytes.
    Only WAV files are supported.
    """
    if not file_path.lower().endswith('.wav'):
        logger.warning("Unsupported file format: %s. Only WAV files are supported.", file_path)
        return None
    try:
        with open(file_path, 'rb') as f:
            return f.read()
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        return None
    except Exception as e:
        logger.exception("Error reading audio file: %s", e)
        return None
    
    
def get_audio_length(audio_bytes):
    """获取音频的长度"""
    audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))  # 从字节流加载音频
    duration_seconds = len(audio) / 1000.0  # `len(audio)` 返回音频的时长，单位为秒
    return duration_seconds