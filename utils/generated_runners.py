# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.


from threading import Thread, Event, Lock
import numpy as np
import random
import io
from utils.logging_config import get_logger

from utils.audio.play_audio import play_audio_from_path, play_audio_from_memory
from livelink.send_to_unreal import pre_encode_facial_data, send_pre_encoded_data_to_unreal
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from livelink.connect.livelink_init import initialize_py_face 
from livelink.animations.animation_emotion import determine_highest_emotion,  merge_emotion_data_into_facial_data_wrapper
from livelink.animations.animation_loader import emotion_animations

queue_lock = Lock()
logger = get_logger(__name__)

def run_audio_animation(audio_input, generated_facial_data, py_face, socket_connection, default_animation_thread, stop_flag_getter=None):
    """
    运行音频动画
    
    Args:
        stop_flag_getter: Optional callable that returns True if playback should be interrupted
    """
    #print(f"输入数据检查: generated_facial_data 类型: {type(generated_facial_data)}, 长度: {len(generated_facial_data) if generated_facial_data is not None else 'None'}")
    
    initial_stop_flag = stop_flag_getter() if stop_flag_getter else None
    logger.debug(
        "[R1] run_audio_animation-start stop_flag=%s audio_type=%s facial_frames=%s",
        initial_stop_flag,
        type(audio_input).__name__,
        len(generated_facial_data) if generated_facial_data is not None else 0,
    )

    if (generated_facial_data is not None and 
        len(generated_facial_data) > 0 and 
        len(generated_facial_data[0]) > 61):
        
        #print(f"面部数据第一帧长度: {len(generated_facial_data[0])}")
        if isinstance(generated_facial_data, np.ndarray):
        #    print("将numpy数组转换为列表")
            generated_facial_data = generated_facial_data.tolist()
        
        facial_data_array = np.array(generated_facial_data)
        #print(f"facial_data_array 形状: {facial_data_array.shape}")
        dominant_emotion = determine_highest_emotion(facial_data_array)
        #print(f"Dominant emotion: {dominant_emotion}")

        if dominant_emotion in emotion_animations and len(emotion_animations[dominant_emotion]) > 0:
        #    print(f"emotion_animations['{dominant_emotion}'] 长度: {len(emotion_animations[dominant_emotion])}")
            selected_animation = random.choice(emotion_animations[dominant_emotion])
            #print(f"选择的动画: {selected_animation}")
            generated_facial_data = merge_emotion_data_into_facial_data_wrapper(generated_facial_data, selected_animation)
         #   print(f"融合后generated_facial_data 长度: {len(generated_facial_data)}")
        else:
            logger.warning("未找到有效的情绪动画 for %s", dominant_emotion)

    else:
        logger.warning("面部数据不符合要求，跳过情绪处理")

    encoding_face = initialize_py_face()
    #print("初始化 py_face 完成")
    encoded_facial_data = pre_encode_facial_data(generated_facial_data, encoding_face)
    #print(f"编码后数据长度: {len(encoded_facial_data)}")

    with queue_lock:
        stop_default_animation.set()
        if default_animation_thread and default_animation_thread.is_alive():
            logger.debug("等待默认动画线程结束")
            default_animation_thread.join()

    start_event = Event()

    if isinstance(audio_input, bytes):       
        audio_thread = Thread(target=play_audio_from_memory, args=(audio_input, start_event, False, stop_flag_getter))
        logger.debug("准备 bytes 音频线程")
    elif isinstance(audio_input, io.BytesIO):
        try:
            logger.debug("处理 BytesIO 音频输入")
            files = audio_input.getvalue()
            audio_thread = Thread(target=play_audio_from_memory, args=(files, start_event, False, stop_flag_getter))
        except Exception as e:
            logger.exception("捕获音频时出错: %s", e)
    else:
        logger.debug("处理音频文件路径输入")
        audio_thread = Thread(target=play_audio_from_path, args=(audio_input, start_event))
    #print("成功捕获语音")

    data_thread = Thread(target=send_pre_encoded_data_to_unreal, args=(encoded_facial_data, start_event, 60, socket_connection, stop_flag_getter))
    #print("数据线程已创建")

    audio_thread.start()
    data_thread.start()
    #print("音频和数据线程已启动")

    start_event.set()
    logger.debug("[R1] run_audio_animation-threads-started")

    join_start = __import__("time").perf_counter()
    audio_thread.join()
    logger.debug(
        "[R1] run_audio_animation-audio-joined alive=%s stop_flag_now=%s",
        audio_thread.is_alive(),
        stop_flag_getter() if stop_flag_getter else None,
    )
    data_thread.join()
    logger.debug(
        "[R1] run_audio_animation-data-joined alive=%s stop_flag_now=%s elapsed_ms=%.2f",
        data_thread.is_alive(),
        stop_flag_getter() if stop_flag_getter else None,
        (__import__("time").perf_counter() - join_start) * 1000,
    )
    #print("音频和数据线程已完成")
    
    # ✅ 确保音频完全播放完毕后再返回
    # 虽然 audio_thread.join() 已经等待了，但为了保险起见，可以再检查一下 mixer 状态
    try:
        import pygame
        while pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            # 如果还在播放（可能是因为 data_thread 结束得比 audio_thread 早），继续等待
            # 但要注意 stop_flag
            if stop_flag_getter and stop_flag_getter():
                pygame.mixer.music.stop()
                break
            import time
            time.sleep(0.05)
    except:
        pass

    with queue_lock:
        stop_default_animation.clear()
    logger.debug("[R1] run_audio_animation-end stop_flag=%s", stop_flag_getter() if stop_flag_getter else None)
        #print("默认动画停止标志已清除")
        #default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
        #default_animation_thread.start()

