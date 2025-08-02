
from utils.files.file_utils import save_generated_data, initialize_directories
from livelink.connect.livelink_init import create_socket_connection, initialize_py_face
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from utils.tts.tts_tools import *   #导入tts
from utils.llm.llm_send import LLMChat
from utils_local_api.generate_face_shapes import generate_facial_data_from_bytes
from utils_local_api.model.model import load_model
from utils_local_api.config import config
from utils.generated_runners import run_audio_animation
import queue
import random
import torch
import time
import numpy as np
from threading import Thread
import threading
import asyncio

def parse_blendshapes_from_json(json_response):
    blendshapes = json_response.get("blendshapes", [])
    facial_data = []

    for frame in blendshapes:
        frame_data = [float(value) for value in frame]
        facial_data.append(frame_data)

    return facial_data

llm_chat = LLMChat()

class Run_LLM_To_Anim():
    def __init__(self):
    # 全局动画合成资源初始化
        self.ENABLE_EMOTE_CALLS = False
        self.model_path = 'utils_local_api/model/model.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.blendshape_model = load_model(self.model_path, config, self.device)
        self.start_default_animation = False  #默认脸部待机动画开关，如果虚幻没有做脸部待机动画就改成True，不然会导致动画重叠      
        self.seed = random.seed
        self.tts_queue= queue.Queue()        
        self.py_face, self.socket_connection, self.default_animation_thread = self.initialize_resources(self.start_default_animation)
        self.stop = False

    def initialize_resources(self, start_default_animation=True):
        initialize_directories()
        py_face = initialize_py_face()
        socket_connection = create_socket_connection()
        
        default_animation_thread = None
        if start_default_animation:
            default_animation_thread = Thread(target=default_animation_loop, args=(self.py_face,))
            default_animation_thread.start()

        return py_face, socket_connection, default_animation_thread

    def audio_to_blendshapes_route(self, audio_bytes):
        generated_facial_data = generate_facial_data_from_bytes(audio_bytes, self.blendshape_model, self.device, config)
        generated_facial_data_list = generated_facial_data.tolist() if isinstance(generated_facial_data, np.ndarray) else generated_facial_data
        return  generated_facial_data_list


    def process_audio_queue(self):
        while True:
            if self.stop:
                while not self.tts_queue.empty():
                    self.tts_queue.get()
                    self.tts_queue.task_done()
                print("TTS queue cleared due to stop signal")
                break
            if self.tts_queue.empty():
                time.sleep(0.1)  # 同步等待，避免空转CPU
                if self.tts_queue.qsize() == 0 and llm_chat.on_streaming==False:  # 确认TTS流结束且队列空
                    print("TTS完事，队列也空，撤了！")
                    break
            #print("开始搞音频！")
            audio_wav = self.tts_queue.get()
            start_time = time.time()
            print(f"队列还剩: {self.tts_queue.qsize()}段音频")
            with app.app_context():
                if audio_wav:
                    try:
                        # 面部数据处理
                        facial_data_raw = self.audio_to_blendshapes_route(audio_wav)
                        facial_data = parse_blendshapes_from_json({'blendshapes': facial_data_raw})
                        #print(f"面部数据帧数: {len(facial_data)}")
                        if facial_data:
                            # 跑动画
                            run_audio_animation(audio_wav, facial_data, self.py_face, self.socket_connection, self.default_animation_thread)
                            #print("动画跑起来啦！")
                        else:
                            print("面部数据空了，跳过吧")
                    except Exception as e:
                        print(f"翻车了: {e}")
                    finally:
                        self.tts_queue.task_done()
                        #print(f"这段音频搞定，耗时: {time.time() - start_time:.3f}秒")
                else:
                    if self.tts_queue.qsize() == 0:
                        self.tts_queue.task_done()

    async def process_llm_to_tts(self, prompt, llm_model,stop_event: asyncio.Event, chatname:str):
        try:
            result = []
            #if stop_event.is_set():
                #self.tts_queue.task_done()
            async for text in llm_chat.process_streaming_content(prompt,llm_model,stop_event, chatname):
                if text:
                    #print(f"process_llm_to_tts received {text}")
                    if self.stop is True:
                        llm_chat.is_streaming_done()
                        await asyncio.sleep(0.2)
                        print("tts队列被打断清空")
                        return
                    audio = await call_TTS(text)
                if audio:
                    #print(f"🔊 二进制音频长度：{len(audio)}")
                    self.tts_queue.put(audio)
                    result.append(text)
                else:
                    print("⚠️ call_TTS 返回空音频")

            self.tts_queue.put(None)
            return "".join(result)

        except Exception as e:
            print(f"❌ Process LLM to TTS error: {e}")
            # 确保上层知道结束
            try:
                self.tts_queue.put(None)
            except:
                pass
            return ""
            
    async def start_queue_audio(self, prompt, llm_model, stop_event: asyncio.Event, chatname:str):
        try:
            from threading import Thread
            tts_thread = Thread(target=self.process_audio_queue)  # 同步线程跑队列
            tts_thread.start()
            response = await self.process_llm_to_tts(prompt, llm_model, stop_event, chatname)  # 异步TTS
            tts_thread.join()  # 等待线程
            print ("所以队列已经完成")
            return response
        except Exception as e:
            print(f"Start queue audio error: {e}")
            return ""

