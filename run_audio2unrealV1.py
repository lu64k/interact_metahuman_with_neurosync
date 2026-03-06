# %%
#===================================== llm模组 ======================================

from utils.utils import init_api_key, get_timestamp
from utils.llm.text_utils import *
from utils.llm.llm_send import load_history
from utils.llm.chat_list_manager import get_all_chat_names
from utils.stt_llm_tts import Run_LLM_To_Anim
from utils.stt.Ali_voicer_rc import *
audio_que=Run_LLM_To_Anim()

#====================================必要的依赖=======================================
import aiohttp.client_exceptions
from pathlib import Path
import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
import numpy as np
import threading
import json
from threading import Thread
import pyaudio
import warnings
import time
from datetime import datetime
import queue
from queue import Queue
import asyncio
warnings.filterwarnings(
    "ignore", 
    message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work"
)

#=====================================语音识别========================================

#--------------------------------------全局变量---------------------------------------
  
recognition = None
recording_thread = None
sample_rate = 16000
channels = 1
block_size = 3200
SILENT_TIMEOUT = 1
MAX_SENTENCE_DURATION = 5
current_task = None  # 跟踪当前任务
callback_rc = Callback_rc()  # 实例化回调函数
rc_queue = callback_rc.rc_queue

def init_recognition(env_name='DASHSCOPE_API_KEY', input_api_key="输入你的api"):
    """初始化语音识别"""
    global recognition
    dashscope.api_key = init_api_key(env_name, input_api_key)
    recognition = Recognition(
        model='paraformer-realtime-v2',
        format='pcm',
        sample_rate=16000,
        heartbeat=True,
        callback=callback_rc
    )
    recognition.start()
    print("Recording started...")

async def process_rc_queue(llm_model,chatname):
    print("Start processing queue")
    global current_task
    while True:
        if rc_queue.empty():
            await asyncio.sleep(0.1)  # 降低CPU占用
            if rc_queue.qsize() == 0 and not callback_rc.stream:
                print("Queue clean, stopping")
                break
        try:
            prompt = rc_queue.get()
            print(f"新的文本生成：{prompt}")
            if prompt:
                stop_event = asyncio.Event()
                callback_rc.stop_llm_generate = False
                audio_que.stop=False
                current_task = await audio_que.start_queue_audio(prompt, llm_model, stop_event, chatname)
                current_task = None            
                prompt = None

        except Exception as e:
            print(f"Queue processing error: {e}")
        finally:
            rc_queue.task_done()

async def async_recording():
    """异步录音循环，非阻塞读音频"""
    global recognition,current_task
    loop = asyncio.get_event_loop()
    try:
        while recognition and callback_rc.stream:
            try:
                # 用run_in_executor非阻塞读音频
                data = await loop.run_in_executor(None, lambda: callback_rc.stream.read(3200, exception_on_overflow=False))
                #print("读取录音...")
                recognition.send_audio_frame(data)#线程已通，on_event已收取结果且成功排队
                #print("已发送录音到识别")

                if callback_rc.stop_llm_generate == True and not callback_rc.first_run :#线程已通，可以被即时打断
                    audio_que.stop = True
                    print("试图停止执行函数收取新内容循环")
                    await asyncio.sleep(5)
                    audio_que.stop = False
                    current_task = None
                    callback_rc.stop_llm_generate = False
            except aiohttp.client_exceptions.ClientConnectionResetError:
                print("WebSocket connection reset, retrying...")
                recognition.stop()
                init_recognition()
                return "retry"
            except Exception as e:
                print(f"Async recording error: {e}")
                if recognition:
                    recognition.stop()
                return
            #print("小循环中")
    except Exception as e:
        print(f"Async recording loop error: {e}")

def record_and_recognize(if_end: bool, llm_model, chatname):  # 录音线程，推送到队列
    global recognition, recording_thread, current_task
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if not if_end:
                # 清理旧状态
                if recognition:
                    recognition.stop()
                    recognition = None
                if recording_thread:
                    recording_thread.join()
                callback_rc.final_texts.clear()
                callback_rc.new_input = None
                callback_rc.stop_llm_generate = False
                while not callback_rc.rc_queue.empty():
                    callback_rc.rc_queue.get()
                    callback_rc.rc_queue.task_done()
                # 启动新线程
                stt_thread = Thread(target=asyncio.run, args=(process_rc_queue(llm_model, chatname),))
                stt_thread.daemon = True
                stt_thread.start()
                init_recognition()
                recording_thread = Thread(target=asyncio.run, args=(async_recording(),))
                recording_thread.daemon = True
                recording_thread.start()
                return "", [], "Recording started"
                
            else:
                if recognition:
                    recognition.stop()
                    recognition = None
                if recording_thread:
                    recording_thread.join()
                paragraph = ''.join(callback_rc.final_texts)
                callback_rc.final_texts.clear()
                print("Recognition stopped. Paragraph:", paragraph)
                if paragraph:
                    return paragraph, [], "Recording stopped"
                return "", [], "Stopped with no text"
        except aiohttp.client_exceptions.ClientConnectionResetError:
            print(f"Connection reset, retry {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                return "", [], f"Failed after {max_retries} retries"
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            if attempt == max_retries - 1:
                return "", [], f"Failed after {max_retries}: {e}"
            continue


# ✅ 动态从 chat_list.json 加载 - 不再硬编码

#=================================== Gradio ========================================== 
# Gradio 界面#

sys_prompt_path = r"E:\Unreal Projects\lipsync_neurosync\unrealversion\config_files\sys_prompt.txt"
system_prompt=load_text(sys_prompt_path)

async def auto_load_history(chatname, status, chatbot):
    while True:
        path, history = load_history(chatname)
        yield {"status": f"Auto-loaded history from {path}", "chatbot": history}
        await asyncio.sleep(60)  # 每60秒加载一次


import gradio as gr

# ✅ 动态获取对话列表
def get_chat_choices():
    """Get current chat choices from chat_list.json"""
    chats = get_all_chat_names()
    return chats if chats else ["default"]

with gr.Blocks() as demo:
    gr.Markdown("# 语音+LLM 聊天机器人")
    with gr.Row():
        with gr.Column(scale=1):
            record_btn = gr.Button("🎙️ 开始录音")
            stop_btn = gr.Button("⏹️ 停止录音")
        with gr.Column(scale=3):
            text_input = gr.Textbox(label="输入或录音结果", placeholder="说点啥...")
            send_btn = gr.Button("🚀 发送")
            update_btn = gr.Button("更新对话历史")
    # ✅ 使用动态加载的对话列表
    chat_choices = get_chat_choices()
    chats = gr.Dropdown(choices=chat_choices, label="选择对话", value=chat_choices[0] if chat_choices else "default")
    chatbot = gr.Chatbot( type="messages", label="对话")
    status = gr. Textbox(label="unreal请求情况")
    with gr.Accordion("LLM 设置", open=False):
        model = gr.Dropdown(choices=["grok-3", "gpt-4o-mini", "deepseek-chat"], label="模型", value="grok-3")

    record_btn.click(
        record_and_recognize,
        inputs=[gr.State(False), model, chats],
        outputs=[text_input,chatbot,status]
    )
    stop_btn.click(
        record_and_recognize,
        inputs=[gr.State(True), model, chats],
        outputs=[text_input,chatbot,status]
    )

    '''    text_input.submit(
            audio_que.start_queue_audio,
            inputs=[text_input, model, chats],
            outputs=[text_input,chatbot,status]
        )'''

    chats.change(
        load_history,
        inputs=[chats],
        outputs=[status,chatbot]
    )
    update_btn.click(
        load_history,
        inputs=[chats],
        outputs=[status,chatbot]
    )
    demo.load(
        auto_load_history,
        inputs=[chats, status, chatbot],
        outputs=[status, chatbot],
        queue=True
    )
demo.launch(server_port=8999, debug=True)

    


