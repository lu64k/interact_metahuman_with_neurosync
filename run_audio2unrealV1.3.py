# %%
#===================================== llm模组 ======================================

from utils.utils import init_api_key, get_timestamp
from utils.llm.text_utils import *
from utils.llm.llm_send import load_history
from utils.llm.chat_list_manager import get_all_chat_names, add_chat
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
from flask import Flask, request, jsonify, Response, stream_with_context
import signal
import sys

# ✅ 初始化 Flask app
app = Flask(__name__)

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

# ✅ 新增：用于存储流式响应的全局字典
streaming_responses = {}
streaming_lock = threading.Lock()

# ✅ 新增：配置锁和当前配置
config_lock = threading.Lock()
current_config = {
    'model': 'gemini-3-flash-preview',
    'chats': 'default'
}

# ✅ 定义推送句子的回调函数（已禁用，兼容任意调用签名）
def push_sentence_callback(*args, **kwargs):
    """
    原本用于向 UE 推送生成句子。移除该功能以避免 UE 端接收不兼容参数（如 init）。
    该函数现在为兼容性空操作，接受任意位置/关键字参数并忽略它们。
    """
    # 调试时可取消注释下面一行以查看调用详情
    # print(f"🔕 push_sentence_callback called with args={args}, kwargs={kwargs} (disabled)")
    return

# ✅ 定义清空 TTS 队列的回调函数
def clear_tts_queue():
    """
    清空 TTS 队列（在检测到新语音输入时调用）
    """
    print("🛑 [实时打断] 清空 TTS 队列并停止 LLM 生成...")
    
    # ✅ 【关键1】设置停止标志
    audio_que.stop = True
    
    # ✅ 【关键2】设置 stop_event 来中断 LLM 流式生成
    if audio_que.current_stop_event:
        try:
            audio_que.current_stop_event.set()
            print("  ✅ 已设置 stop_event，LLM 流式生成将停止")
        except Exception as e:
            print(f"  ⚠️ 设置 stop_event 失败: {e}")
    
    # ✅ 【关键3】清空 TTS 队列
    cleared_count = 0
    while not audio_que.tts_queue.empty():
        try:
            audio_que.tts_queue.get_nowait()
            audio_que.tts_queue.task_done()
            cleared_count += 1
        except:
            break
    if cleared_count > 0:
        print(f"  ✅ 已清空 {cleared_count} 个 TTS 队列项")
    
    print("✅ [实时打断] 打断完成")
    # 注意：不重置 stop 标志，让 process_audio_queue 自己重置

# ✅ 注册回调
callback_rc.clear_tts_callback = clear_tts_queue

def init_recognition(env_name='DASHSCOPE_API_KEY', input_api_key="输入你的api"):
    """初始化语音识别"""
    global recognition
    
    try:
        print("🔧 正在初始化 Dashscope API...")
        dashscope.api_key = init_api_key(env_name, input_api_key)
        
        print("🔧 正在创建 Recognition 对象...")
        recognition = Recognition(
            model='paraformer-realtime-v2',
            format='pcm',
            sample_rate=16000,
            heartbeat=True,
            callback=callback_rc
        )
        
        print("🔧 正在启动 Recognition...")
        recognition.start()
        
        print("✅ Recording started successfully")
        return True
        
    except Exception as e:
        print(f"❌ init_recognition 失败: {e}")
        import traceback
        traceback.print_exc()
        recognition = None
        raise  # 重新抛出异常,让上层处理

async def process_rc_queue(llm_model, chatname, session_id=None):
    """
    处理语音识别队列
    
    ✅ 现在支持动态读取配置：每次处理时从 current_config 获取最新的模型和对话设置
    """
    print(f"Start processing queue (session: {session_id})")
    global current_task, current_config
    
    # ✅ 如果 session_id 为空，生成一个新的（兜底）
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
        print(f"⚠️ [队列处理] 初始 session_id 为空，生成新的: {session_id}")
    
    while True:
        if rc_queue.empty():
            await asyncio.sleep(0.1)  # 降低CPU占用
            if rc_queue.qsize() == 0 and not callback_rc.stream:
                print("Queue clean, stopping")
                break
            continue  # ✅ 队列为空时直接继续，不执行后续逻辑
        
        try:
            # ✅ 【关键修复】只在队列不为空时才重置标志
            print("🔄 [新请求] 重置打断标志（在获取队列项之前）...")
            callback_rc.stop_llm_generate = False
            audio_que.stop = False
            
            # ✅ 【重要】延迟时间要足够让监控线程的边缘触发逻辑重置
            # 监控线程每 50ms 检查一次，这里等待至少 2-3 个检查周期
            await asyncio.sleep(0.2)
            
            prompt = rc_queue.get()
            print(f"新的文本生成：{prompt}")
            
            # ✅ 【关键修复】检查 prompt 是否为空
            if not prompt or (isinstance(prompt, str) and prompt.strip() == ""):
                print("⚠️ [队列处理] 收到空 prompt，跳过处理")
                rc_queue.task_done()
                continue
            
            if prompt:
                stop_event = asyncio.Event()
                
                # ✅ 保存 stop_event 引用，用于打断
                audio_que.current_stop_event = stop_event
                
                # ✅ 【动态配置】从 current_config 读取最新的模型和对话设置
                with config_lock:
                    active_model = current_config.get('model', llm_model)
                    active_chats = current_config.get('chats', chatname)
                
                print(f"📊 [队列处理] 使用配置: model={active_model}, chats={active_chats}")
                
                # ✅ 传入 session_id 和回调函数
                # 在播放音频时会自动推送句子到 UE5
                streaming_callback = push_sentence_callback
                
                # ✅ 如果 session_id 为空，生成一个新的（虽然通常应该由上层传入）
                if not session_id:
                    import uuid
                    session_id = str(uuid.uuid4())
                    print(f"⚠️ [队列处理] 未收到 session_id，生成新的: {session_id}")

                # ✅ 使用最新的配置
                result = await audio_que.start_queue_audio(
                    prompt, active_model, stop_event, active_chats, session_id, streaming_callback
                )
                
                # ✅ 解包返回值
                if result:
                    response_text, generated_session_id = result
                    if generated_session_id:
                        print(f"🆔 [队列处理] 生成的 Session ID: {generated_session_id}")
                
                current_task = None
                audio_que.current_stop_event = None  # ✅ 清理引用
                prompt = None

        except Exception as e:
            print(f"Queue processing error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            rc_queue.task_done()

async def async_recording():
    """异步录音循环，非阻塞读音频"""
    global recognition, current_task
    loop = asyncio.get_event_loop()
    
    print("📹 async_recording 开始")
    
    try:
        # ✅ 等待 stream 准备好
        retry_count = 0
        while not callback_rc.stream and retry_count < 10:
            print(f"  ⏳ 等待音频流准备... ({retry_count+1}/10)")
            await asyncio.sleep(0.5)
            retry_count += 1
        
        if not callback_rc.stream:
            print("❌ 音频流未能在5秒内准备好,退出录音")
            return "stream_not_ready"
        
        print("✅ 音频流已准备好,开始录音循环")
        
        while recognition and callback_rc.stream:
            try:
                # 检查 stream 是否仍然有效
                if not callback_rc.stream.is_active():
                    print("⚠️ 音频流已停止")
                    break
                
                # 用run_in_executor非阻塞读音频
                data = await loop.run_in_executor(
                    None, 
                    lambda: callback_rc.stream.read(3200, exception_on_overflow=False)
                )
                
                if recognition:
                    recognition.send_audio_frame(data)

                # ✅ 检测到新语音输入，快速打断
                if callback_rc.stop_llm_generate == True and not callback_rc.first_run:
                    print("🛑 [实时打断] 检测到新语音，准备清空队列...")
                    # 注意：TTS 队列已经在 callback 中清空了
                    # 这里只需等待一小段时间让队列处理完成
                    await asyncio.sleep(0.5)  # ✅ 改为 0.5 秒，更快响应
                    current_task = None
                    callback_rc.stop_llm_generate = False
                    print("✅ [实时打断] 准备接收新对话")
                    
            except aiohttp.client_exceptions.ClientConnectionResetError:
                print("⚠️ WebSocket connection reset")
                if recognition:
                    try:
                        recognition.stop()
                    except:
                        pass
                    recognition = None
                return "connection_reset"
                
            except OSError as e:
                print(f"⚠️ 音频设备错误: {e}")
                if recognition:
                    try:
                        recognition.stop()
                    except:
                        pass
                    recognition = None
                return "audio_device_error"
                
            except Exception as e:
                print(f"⚠️ Async recording error: {e}")
                import traceback
                traceback.print_exc()
                if recognition:
                    try:
                        recognition.stop()
                    except:
                        pass
                    recognition = None
                return "error"
                
    except Exception as e:
        print(f"❌ Async recording loop error: {e}")
        import traceback
        traceback.print_exc()
        if recognition:
            try:
                recognition.stop()
            except:
                pass
            recognition = None
    
    print("📹 async_recording 结束")
    return "completed"

def record_and_recognize(if_end: bool, llm_model, chatname, session_id=None):  # 录音线程，推送到队列
    global recognition, recording_thread, current_task
    
    # ✅ 更新全局配置
    with config_lock:
        current_config['model'] = llm_model
        current_config['chats'] = chatname
    
    try:
        if not if_end:
            # ✅ 强制清理旧状态 - 开始录音
            print(f"🎙️ 开始录音 (session: {session_id})...")

            # 停止旧的 recognition
            if recognition:
                try:
                    recognition.stop()
                except:
                    pass
                recognition = None
            
            # 等待旧线程结束
            if recording_thread:
                try:
                    recording_thread.join(timeout=2)
                except:
                    pass
                recording_thread = None
            
            # ✅ 关闭旧的音频流
            if callback_rc.stream:
                try:
                    callback_rc.stream.stop_stream()
                    callback_rc.stream.close()
                except:
                    pass
                callback_rc.stream = None
            
            if callback_rc.mic:
                try:
                    callback_rc.mic.terminate()
                except:
                    pass
                callback_rc.mic = None
            
            # ✅ 重置所有回调状态
            callback_rc.final_texts.clear()
            callback_rc.new_input = None
            callback_rc.new_sentence = None
            callback_rc.stop_llm_generate = False
            callback_rc.first_run = False
            callback_rc.first_sentence_time = None
            callback_rc.last_sentence_time = None
            
            # ✅ 清空语音识别队列
            print("🧹 清空语音识别队列...")
            while not callback_rc.rc_queue.empty():
                try:
                    callback_rc.rc_queue.get_nowait()
                    callback_rc.rc_queue.task_done()
                except:
                    break
            
            # ✅ 清空 TTS 队列（重要！）
            print("🧹 清空 TTS 队列...")
            audio_que.stop = True  # 设置停止标志
            cleared_tts = 0
            while not audio_que.tts_queue.empty():
                try:
                    audio_que.tts_queue.get_nowait()
                    audio_que.tts_queue.task_done()
                    cleared_tts += 1
                except:
                    break
            if cleared_tts > 0:
                print(f"✅ 已清空 {cleared_tts} 个 TTS 队列项")
            # 重置停止标志将在新线程启动时自动完成
            
            # 短暂延迟，确保资源释放
            time.sleep(0.5)
            
            # 启动新线程
            stt_thread = Thread(target=asyncio.run, args=(process_rc_queue(llm_model, chatname, session_id),))
            stt_thread.daemon = True
            stt_thread.start()
            
            # ✅ 重新初始化识别
            try:
                init_recognition()
            except Exception as e:
                print(f"❌ 初始化识别失败: {e}")
                return "", [], f"Init failed: {e}"
            
            # 启动录音线程
            recording_thread = Thread(target=asyncio.run, args=(async_recording(),))
            recording_thread.daemon = True
            recording_thread.start()
            
            print("✅ 录音已启动")
            return "", [], "Recording started"
            
        else:
            # ✅ 停止录音 - 不手动拼接，等待 time_judger 完成
            print("⏹️ 停止录音...")
            
            if recognition:
                try:
                    recognition.stop()
                except:
                    pass
                recognition = None
            
            if recording_thread:
                try:
                    recording_thread.join(timeout=3)
                except:
                    pass
                recording_thread = None
            
            # ✅ 等待最后一个 time_judger 完成（最多2秒）
            wait_start = time.time()
            while callback_rc.new_sentence is not None and time.time() - wait_start < 2:
                time.sleep(0.1)
            
            print("✅ 录音已停止")
            return "", [], "Recording stopped"
            
    except Exception as e:
        print(f"❌ record_and_recognize 错误: {e}")
        import traceback
        traceback.print_exc()
        
        # ✅ 发生错误时，彻底清理状态
        if recognition:
            try:
                recognition.stop()
            except:
                pass
            recognition = None
        
        if callback_rc.stream:
            try:
                callback_rc.stream.stop_stream()
                callback_rc.stream.close()
            except:
                pass
            callback_rc.stream = None
        
        if callback_rc.mic:
            try:
                callback_rc.mic.terminate()
            except:
                pass
            callback_rc.mic = None
        
        return "", [], f"Error: {str(e)}"

def record_and_recognize_with_stream(if_end: bool, llm_model, chatname, session_id=None):
    """
    带流式响应的录音识别函数
    
    Args:
        if_end: 是否结束录音
        llm_model: 模型名称
        chatname: 对话名称
        session_id: 会话ID,用于流式推送
    """
    global recognition, recording_thread, current_task
    
    try:
        if not if_end:
            # ✅ 强制清理旧状态 - 开始录音
            print(f"🎙️ 开始录音 (session: {session_id})...")
            
            # 停止旧的 recognition
            if recognition:
                try:
                    recognition.stop()
                except:
                    pass
                recognition = None
            
            if recording_thread:
                try:
                    recording_thread.join(timeout=2)
                except:
                    pass
                recording_thread = None
            
            if callback_rc.stream:
                try:
                    callback_rc.stream.stop_stream()
                    callback_rc.stream.close()
                except:
                    pass
                callback_rc.stream = None
            
            if callback_rc.mic:
                try:
                    callback_rc.mic.terminate()
                except:
                    pass
                callback_rc.mic = None
            
            callback_rc.final_texts.clear()
            callback_rc.new_input = None
            callback_rc.new_sentence = None
            callback_rc.stop_llm_generate = False
            callback_rc.first_run = False
            callback_rc.first_sentence_time = None
            callback_rc.last_sentence_time = None
            
            # ✅ 清空语音识别队列
            print("🧹 清空语音识别队列...")
            while not callback_rc.rc_queue.empty():
                try:
                    callback_rc.rc_queue.get_nowait()
                    callback_rc.rc_queue.task_done()
                except:
                    break
            
            # ✅ 清空 TTS 队列（重要！）
            print("🧹 清空 TTS 队列...")
            audio_que.stop = True
            cleared_tts = 0
            while not audio_que.tts_queue.empty():
                try:
                    audio_que.tts_queue.get_nowait()
                    audio_que.tts_queue.task_done()
                    cleared_tts += 1
                except:
                    break
            if cleared_tts > 0:
                print(f"✅ 已清空 {cleared_tts} 个 TTS 队列项")
            
            time.sleep(0.5)
            
            # ✅ 启动新线程,传入 session_id
            stt_thread = Thread(
                target=asyncio.run, 
                args=(process_rc_queue(llm_model, chatname, session_id),)
            )
            stt_thread.daemon = True
            stt_thread.start()
            
            try:
                init_recognition()
            except Exception as e:
                print(f"❌ 初始化识别失败: {e}")
                return "", [], f"Init failed: {e}"
            
            recording_thread = Thread(target=asyncio.run, args=(async_recording(),))
            recording_thread.daemon = True
            recording_thread.start()
            
            print(f"✅ 录音已启动 (session: {session_id})")
            return "", [], "Recording started"
            
        else:
            # ✅ 停止录音 - 不手动拼接，等待 time_judger 完成
            print("⏹️ 停止录音...")
            
            if recognition:
                try:
                    recognition.stop()
                except:
                    pass
                recognition = None
            
            if recording_thread:
                try:
                    recording_thread.join(timeout=3)
                except:
                    pass
                recording_thread = None
            
            # ✅ 等待最后一个 time_judger 完成（最多2秒）
            wait_start = time.time()
            while callback_rc.new_sentence is not None and time.time() - wait_start < 2:
                time.sleep(0.1)
            
            print("✅ 录音已停止")
            return "", [], "Recording stopped"
            
    except Exception as e:
        print(f"❌ record_and_recognize_with_stream 错误: {e}")
        import traceback
        traceback.print_exc()
        
        # 清理资源
        if recognition:
            try:
                recognition.stop()
            except:
                pass
            recognition = None
        
        if callback_rc.stream:
            try:
                callback_rc.stream.stop_stream()
                callback_rc.stream.close()
            except:
                pass
            callback_rc.stream = None
        
        if callback_rc.mic:
            try:
                callback_rc.mic.terminate()
            except:
                pass
            callback_rc.mic = None
        
        return "", [], f"Error: {str(e)}"

# ✅ 动态从 chat_list.json 加载 - 不再硬编码

# ✅ 新增：创建新对话的函数
def create_new_chat(display_name, file_name):
    """
    创建新对话（Gradio回调函数）
    
    Args:
        display_name: 新对话的显示名称
        file_name: 新对话的文件名（用于存储对话历史）
    
    Returns:
        tuple: (status_message, updated_chats_dropdown)
    """
    if not display_name or not file_name:
        return "❌ 请输入对话名称和文件名", gr.Dropdown.update()
    
    try:
        # Check if chat already exists
        if display_name in get_all_chat_names():
            return f"❌ 对话 '{display_name}' 已存在", gr.Dropdown.update(choices=get_all_chat_names())
        
        # Create empty dialogue history file
        import os
        chat_path = os.path.join("dialogue_histories", f"{file_name}.txt")
        path = resolve_path(chat_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create empty JSON array file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        
        # Add to chat_list.json
        success = add_chat(display_name, file_name)
        
        if success:
            new_choices = get_all_chat_names()
            return f"✅ 成功创建对话: {display_name}", gr.Dropdown.update(choices=new_choices, value=display_name)
        else:
            return "❌ 添加到 chat_list.json 失败", gr.Dropdown.update(choices=get_all_chat_names())
            
    except Exception as e:
        print(f"❌ create_new_chat error: {e}")
        return f"❌ 创建失败: {e}", gr.Dropdown.update(choices=get_all_chat_names())

@app.route('/create_chat', methods=['POST'])
def create_chat_api():
    """
    API端点：创建新对话
    
    POST JSON data:
    {
        "display_name": "新对话名称",
        "file_name": "文件名"
    }
    
    Returns:
        {
            "success": bool,
            "message": str,
            "chat_list": list of all chat names
        }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "message": "No JSON data provided",
                "chat_list": get_all_chat_names()
            }), 400
        
        display_name = data.get('display_name', '').strip()
        file_name = data.get('file_name', '').strip()
        
        if not display_name or not file_name:
            return jsonify({
                "success": False,
                "message": "display_name and file_name are required",
                "chat_list": get_all_chat_names()
            }), 400
        
        # Check if chat already exists
        if display_name in get_all_chat_names():
            return jsonify({
                "success": False,
                "message": f"Chat '{display_name}' already exists",
                "chat_list": get_all_chat_names()
            }), 400
        
        # Create empty dialogue history file
        import os
        chat_path = os.path.join("dialogue_histories", f"{file_name}.txt")
        path = resolve_path(chat_path)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create empty JSON array file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        
        # Add to chat_list.json
        success = add_chat(display_name, file_name)
        if success:
            return jsonify({
                "success": True,
                "message": f"Successfully created chat: {display_name}",
                "chat_list": get_all_chat_names()
            }), 200
        else:
            return jsonify({
                "success": False,
                "message": "Failed to add chat to chat_list.json",
                "chat_list": get_all_chat_names()
            }), 500
    
    except Exception as e:
        print(f"❌ create_chat_api error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}",
            "chat_list": get_all_chat_names()
        }), 500


@app.route('/get_chats', methods=['GET'])
def get_chats_api():
    """
    API端点：获取所有对话列表
    
    Returns:
        {
            "chats": list of chat display names
        }
    """
    try:
        chats = get_all_chat_names()
        return jsonify({
            "success": True,
            "chats": chats
        }), 200
    except Exception as e:
        print(f"❌ get_chats_api error: {e}")
        return jsonify({
            "success": False,
            "message": str(e),
            "chats": []
        }), 500

@app.route('/unreal_mh', methods=['POST'])
def unreal_mh_api():
    """
    Unreal Engine 交互端点
    支持:
    1. type='start' 或 On Recording=True: 开始录音
    2. type='stop' 或 On Recording=False: 停止录音
    3. type='poll': 获取最新生成的句子 (流式/轮询)
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No JSON data"}), 400
            
        # ✅ 兼容虚幻引擎的字段名 (On Recording, Model, Chats)
        on_recording = data.get('on_recording')
        model = data.get('model') or data.get('model', 'gemini-3-flash-preview')
        chatname = data.get('chats') or data.get('chatname', 'default')
        req_type = data.get('type')
        session_id = data.get('session_id')
        
        if not session_id:
            # 如果没有 session_id，生成一个
            import uuid
            session_id = str(uuid.uuid4())
            
        # ✅ 逻辑判断：支持新旧两种格式
        is_start = (req_type == 'start') or (on_recording is True) or (on_recording == 1)
        is_stop = (req_type == 'stop') or (on_recording is False) or (on_recording == 0)
        is_poll = (req_type == 'poll')
        # ✅ 兼容旧版/自定义字段，支持更新配置的请求
        is_update_info = (on_recording == 'keep')

        if is_start:
            # 启动录音 (带流式支持)
            msg, _, status = record_and_recognize_with_stream(False, model, chatname, session_id)
            return jsonify({
                "success": True, 
                "message": status,
                "session_id": session_id
            })
            
        elif is_stop:
            # 停止录音
            msg, _, status = record_and_recognize_with_stream(True, "", "", session_id)
            return jsonify({
                "success": True, 
                "message": status,
                "session_id": session_id
            })
        

        elif is_update_info:
            # ✅ 更新全局配置
            with config_lock:
                current_config['model'] = model
                current_config['chats'] = chatname
            print(f"🔄 [unreal_mh] Updated config: model={model}, chats={chatname}")
            return jsonify({
                "success": True,
                "message": f"Config updated: model={model}, chats={chatname}",
                "session_id": session_id
            })
            
        elif is_poll:
            # 获取最新的句子
            with streaming_lock:
                if session_id in streaming_responses:
                    resp_data = streaming_responses[session_id]
                    sentences = resp_data["sentences"]
                    last_index = resp_data["last_index"]
                    
                    # 获取新句子
                    new_sentences = sentences[last_index:]
                    
                    # 更新索引
                    resp_data["last_index"] = len(sentences)
                    
                    return jsonify({
                        "success": True,
                        "sentences": new_sentences,
                        "session_id": session_id
                    })
                else:
                    return jsonify({
                        "success": True,
                        "sentences": [],
                        "session_id": session_id,
                        "message": "No session found"
                    })
                    
        else:
            # 如果既不是 start/stop 也不是 poll，返回错误
            return jsonify({"success": False, "message": "Invalid request parameters (missing type or On Recording)"}), 400
            
    except Exception as e:
        print(f"❌ unreal_mh_api error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500

def run_flask():
    app.run(host='0.0.0.0', port=8999, debug=True, use_reloader=False)
def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)    
#=================================== Gradio ========================================== 
# Gradio 界面#

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
        # ✅ 新增：创建新对话的输入框和按钮
        with gr.Row():
            new_chat_name = gr.Textbox(label="新对话名称", placeholder="输入新对话的名称")
            new_chat_filename = gr.Textbox(label="文件名", placeholder="输入对话的文件名（无扩展名）")
            create_chat_btn = gr.Button("➕ 创建新对话")

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

    # ✅ 新增：创建新对话的回调
    create_chat_btn.click(
        create_new_chat,
        inputs=[new_chat_name, new_chat_filename],
        outputs=[status, chats]
    )

    # ✅ [FIX] 确保 chats 下拉框更新时，正确加载历史记录
    # 注意：load_history 返回 (path, history)，但 chatbot 需要 history
    # 我们需要一个包装函数来只返回 history
    def load_history_for_gradio(chatname):
        path, history = load_history(chatname)
        return f"Loaded from {path}", history

    chats.change(
        load_history_for_gradio,
        inputs=[chats],
        outputs=[status,chatbot]
    )
    update_btn.click(
        load_history_for_gradio,
        inputs=[chats],
        outputs=[status,chatbot]
    )
    # ✅ [FIX] auto_load_history 已经是生成器，可以直接用
    # 但需要确保它在后台运行，不阻塞 UI
    # demo.load 已经支持 queue=True
    demo.load(
        auto_load_history,
        inputs=[chats, status, chatbot],
        outputs=[status, chatbot],
        queue=True
    )


if __name__ == '__main__':
    # ✅ 添加全局异常处理
    import sys
    import traceback
    
    def global_exception_handler(exctype, value, tb):
        """全局异常处理器,防止程序闪退"""
        print("\n" + "="*60)
        print("❌ 捕获到未处理的异常:")
        print("="*60)
        traceback.print_exception(exctype, value, tb)
        print("="*60)
        print("程序将继续运行...")
        print("="*60 + "\n")
    
    sys.excepthook = global_exception_handler
    
    # 启动 Flask
    flask_thread = Thread(target=run_flask)
    flask_thread.daemon = True  # ✅ 设为守护线程
    flask_thread.start()

    # 启动 Gradio
    try:
        demo.launch(server_port=8989, debug=True)
    except Exception as e:
        print(f"❌ Gradio 启动失败: {e}")
        traceback.print_exc()




