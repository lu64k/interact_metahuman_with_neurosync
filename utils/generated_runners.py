# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.


from threading import Thread, Event, Lock
import numpy as np
import random
import io

from utils.audio.play_audio import play_audio_from_path, play_audio_from_memory
from livelink.send_to_unreal import pre_encode_facial_data, send_pre_encoded_data_to_unreal
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from livelink.connect.livelink_init import initialize_py_face 
from livelink.animations.animation_emotion import determine_highest_emotion,  merge_emotion_data_into_facial_data_wrapper
from livelink.animations.animation_loader import emotion_animations

queue_lock = Lock()

def run_audio_animation(audio_input, generated_facial_data, py_face, socket_connection, default_animation_thread):
    #print(f"输入数据检查: generated_facial_data 类型: {type(generated_facial_data)}, 长度: {len(generated_facial_data) if generated_facial_data is not None else 'None'}")
    
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
            print(f"未找到有效的情绪动画 for {dominant_emotion}")

    else:
        print("面部数据不符合要求，跳过情绪处理")

    encoding_face = initialize_py_face()
    #print("初始化 py_face 完成")
    encoded_facial_data = pre_encode_facial_data(generated_facial_data, encoding_face)
    #print(f"编码后数据长度: {len(encoded_facial_data)}")

    with queue_lock:
        stop_default_animation.set()
        if default_animation_thread and default_animation_thread.is_alive():
            print("等待默认动画线程结束")
            default_animation_thread.join()

    start_event = Event()

    if isinstance(audio_input, bytes):       
        audio_thread = Thread(target=play_audio_from_memory, args=(audio_input, start_event))
        print("播放完毕")
    elif isinstance(audio_input, io.BytesIO):
        try:
            print("处理BytesIO音频输入")
            files = audio_input.getvalue()
            audio_thread = Thread(target=play_audio_from_memory, args=(files, start_event))
        except Exception as e:
            print(f"捕获音频时出错: {e}")           
    else:
        print("处理音频文件路径输入")
        audio_thread = Thread(target=play_audio_from_path, args=(audio_input, start_event))
    #print("成功捕获语音")

    data_thread = Thread(target=send_pre_encoded_data_to_unreal, args=(encoded_facial_data, start_event, 60, socket_connection))
    #print("数据线程已创建")

    audio_thread.start()
    data_thread.start()
    #print("音频和数据线程已启动")

    start_event.set()

    audio_thread.join()
    data_thread.join()
    #print("音频和数据线程已完成")

    with queue_lock:
        stop_default_animation.clear()
        #print("默认动画停止标志已清除")
        #default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
        #default_animation_thread.start()

