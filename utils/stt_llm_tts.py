from utils.files.file_utils import save_generated_data, initialize_directories
from livelink.connect.livelink_init import create_socket_connection, initialize_py_face
from livelink.animations.default_animation import default_animation_loop, stop_default_animation
from utils.tts.tts_tools import *   #导入tts
from utils.llm.llm_send import LLMChat, load_history
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
from utils.logging_config import get_logger

logger = get_logger(__name__)

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
        # ✅ 新增：用于传递文本和音频的队列项
        self.current_session_id = None  # 当前 session_id
        self.streaming_callback = None  # 回调函数，用于推送句子
        self.current_stop_event = None  # ✅ 当前的 stop_event 引用
        self.interrupt_monitor_running = False  # ✅ 监控线程运行标志
        self.interrupt_monitor_thread = None  # ✅ 监控线程
        self.queue_processor_thread = None  # ✅ 队列处理线程
        self.queue_processor_running = False  # ✅ 队列处理线程运行标志
        self.interrupt_seq = 0  # R1 debug: monotonic interrupt sequence id

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

    def interrupt_monitor(self):
        """
        ✅ 监控线程：实时检查打断信号并立即停止音频
        独立线程运行，不受 process_audio_queue 阻塞影响
        """
        logger.info("[监控线程] 启动打断监控")
        self.interrupt_monitor_running = True
        last_stop_state = False  # ✅ 记录上一次的 stop 状态
        
        while self.interrupt_monitor_running:
            try:
                # ✅ 只在 stop 从 False 变为 True 时才执行打断（边缘触发）
                if self.stop and not last_stop_state:
                    logger.warning("[监控线程] 检测到打断信号，立即执行强制中断")
                    
                    # ✅ 1. 立即停止音频播放
                    try:
                        import pygame
                        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                            pygame.mixer.music.stop()
                            logger.info("[监控线程] 已强制停止音频播放")
                    except Exception as e:
                        logger.warning("[监控线程] 停止音频失败: %s", e)
                    
                    # ✅ 2. 设置 stop_event 停止 LLM 生成
                    if self.current_stop_event:
                        try:
                            self.current_stop_event.set()
                            logger.info("[监控线程] 已设置 stop_event")
                        except Exception as e:
                            logger.warning("[监控线程] 设置 stop_event 失败: %s", e)
                    
                    logger.info("[监控线程] 强制中断完成，等待新请求")
                
                # ✅ 更新上一次状态
                last_stop_state = self.stop
                
                time.sleep(0.05)  # 每 50ms 检查一次
                
            except Exception as e:
                logger.exception("[监控线程] 错误: %s", e)
                time.sleep(0.1)
        
        logger.info("[监控线程] 停止监控")

    def process_audio_queue(self):
        """
        持久运行的队列处理线程
        不会因为打断而退出，持续监听队列
        """
        logger.info("[队列线程] 启动音频队列处理线程")
        self.queue_processor_running = True
        stop_handled = False  # 防抖：同一轮 stop=True 只处理一次
        
        try:
            while self.queue_processor_running:
                try:
                    # ✅ 检查是否需要停止当前播放
                    if self.stop:
                        if stop_handled:
                            # 已处理过本轮 stop，等待新请求复位 stop=False
                            time.sleep(0.05)
                            continue

                        interrupt_seq = self.interrupt_seq
                        stop_branch_ts = time.perf_counter()
                        logger.warning("[队列线程] 收到停止信号，立即停止音频播放并清空 TTS 队列")
                        logger.debug(
                            "[R1] stop-branch-enter seq=%s stop=%s queue_size=%s",
                            interrupt_seq,
                            self.stop,
                            self.tts_queue.qsize(),
                        )
                        # ✅ 【关键】立即停止正在播放的音频
                        mixer_busy = False
                        try:
                            import pygame
                            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                                pygame.mixer.music.stop()
                                logger.info("[队列线程] 已停止正在播放的音频")
                                mixer_busy = True
                        except Exception as e:
                            logger.warning("[队列线程] 停止音频播放失败: %s", e)
                        
                        # ✅ 清空队列
                        cleared_count = 0
                        while not self.tts_queue.empty():
                            try:
                                self.tts_queue.get_nowait()
                                self.tts_queue.task_done()
                                cleared_count += 1
                            except:
                                break
                        logger.info("[队列线程] TTS 队列已清空 (%s 项)", cleared_count)
                        # ✅ 关键：不要在队列线程里立即复位 stop。
                        # 复位交给新请求入口（ASRManager.process_rc_queue）统一执行，
                        # 避免旧 LLM/TTS 在打断后回流入队。
                        logger.debug(
                            "[R1] stop-reset-before seq=%s stop=%s mixer_busy_was=%s queue_size=%s elapsed_ms=%.2f",
                            interrupt_seq,
                            self.stop,
                            mixer_busy,
                            self.tts_queue.qsize(),
                            (time.perf_counter() - stop_branch_ts) * 1000,
                        )
                        logger.info("[队列线程] 打断处理完成，等待新请求复位 stop")
                        logger.debug(
                            "[R1] stop-reset-after seq=%s stop=%s queue_size=%s",
                            interrupt_seq,
                            self.stop,
                            self.tts_queue.qsize(),
                        )
                        stop_handled = True
                        # ❌ 不退出，继续循环等待新队列项
                        continue

                    # stop=False，允许下一轮 stop 再次触发一次性处理
                    if stop_handled:
                        logger.debug("[R1] stop latch released by new request")
                        stop_handled = False
                    
                    if self.tts_queue.empty():
                        time.sleep(0.1)  # 同步等待，避免空转CPU
                        # ✅ 不再退出，持续等待新的队列项
                        continue  # ✅ 队列空时继续等待
                    
                    # ✅ 从队列获取项目前再次检查打断
                    if self.stop:
                        continue  # 回到循环开始处理打断
                    
                    # ✅ 从队列获取项目
                    try:
                        queue_item = self.tts_queue.get(timeout=0.5)
                    except:
                        continue
                    
                    logger.debug("[队列线程] 从队列获取项目，剩余: %s 段音频", self.tts_queue.qsize())
                    
                    # ✅ None 是队列结束标记（但不退出线程，继续等待下一轮）
                    if queue_item is None:
                        logger.info("[队列线程] 收到队列结束标记，本轮播放完成")
                        self.tts_queue.task_done()
                        # ❌ 不退出线程，继续等待新的队列项
                        continue  # ✅ 改为 continue
                    
                    # ✅ 处理会话结束标记
                    if isinstance(queue_item, dict) and queue_item.get("type") == "end":
                        logger.info("[队列线程] 收到会话结束标记")
                        # session_id = queue_item.get("session_id")
                        # # ✅ 优先使用 item 中的 callback
                        # callback = queue_item.get("callback", self.streaming_callback)
                        
                        # if session_id and callback:
                        #     try:
                        #         callback(None, session_id, completed=True)
                        #         print(f"✅ [队列线程] 标记 Session {session_id} 完成")
                        #     except Exception as e:
                        #         print(f"⚠️ [队列线程] 标记 Session 完成失败: {e}")
                        self.tts_queue.task_done()
                        continue

                    # ✅ 处理前再次检查打断
                    if self.stop:
                        self.tts_queue.task_done()
                        continue
                    
                    # ✅ 处理音频项
                    # with app.app_context(): # ❌ 移除 app.app_context()，因为 app 在这里不可见
                    try:
                        # 提取文本和音频
                        sentence_text = queue_item.get("text", "")
                        audio_wav = queue_item.get("audio", None)
                        # ✅ 获取 session_id (优先使用 item 中的)
                        item_session_id = queue_item.get("session_id", self.current_session_id)
                        
                        # ✅ 优先使用 item 中的 callback
                        callback = queue_item.get("callback", self.streaming_callback)

                        # ✅ 【关键】在播放音频前，推送文本到 UE5
                        # 这样可以确保字幕和音频同步
                        # if sentence_text and callback:
                        #     try:
                        #         # ✅ [FIX] 修正 callback 调用参数
                        #         # push_sentence_callback(session_id, sentence)
                        #         callback(item_session_id, sentence_text)
                        #         print(f"📤 [队列线程] 推送句子到 UE5: {sentence_text}")
                        #     except Exception as e:
                        #         print(f"⚠️ [队列线程] 推送句子失败: {e}")
                        
                        # ✅ 播放前最后一次检查打断
                        if self.stop:
                            logger.warning("[队列线程] 播放前检测到打断，跳过该音频")
                            self.tts_queue.task_done()
                            continue
                        
                        if audio_wav:
                            logger.info("[队列线程] 开始播放音频: %s...", sentence_text[:30])
                            # 面部数据处理
                            facial_data_raw = self.audio_to_blendshapes_route(audio_wav)
                            facial_data = parse_blendshapes_from_json({'blendshapes': facial_data_raw})
                            if facial_data:
                                # ✅ 跑动画，传入打断检查函数
                                run_audio_animation(
                                    audio_wav, 
                                    facial_data, 
                                    self.py_face, 
                                    self.socket_connection, 
                                    self.default_animation_thread,
                                    stop_flag_getter=lambda: self.stop  # 传入打断检查
                                )
                                logger.info("[队列线程] 音频播放完成: %s...", sentence_text[:30])
                            else:
                                logger.warning("[队列线程] 面部数据空了，跳过")
                        else:
                            logger.warning("[队列线程] 音频数据为空")
                        
                    except Exception as e:
                        logger.exception("[队列线程] 处理音频队列项失败: %s", e)
                    finally:
                        self.tts_queue.task_done()
                
                except Exception as e:
                    # ✅ 捕获内层循环的所有异常，防止线程意外退出
                    logger.exception("[队列线程] 内层循环异常: %s", e)
                    time.sleep(0.1)  # 防止异常导致的快速循环
        
        except Exception as e:
            # ✅ 捕获外层异常（理论上不应该发生）
            logger.exception("[队列线程] 外层循环异常: %s", e)
        finally:
            # ✅ 线程退出清理
            logger.info("[队列线程] 音频队列处理线程退出")
            self.queue_processor_running = False

    async def process_llm_to_tts(self, prompt, llm_model,stop_event: asyncio.Event, chatname:str):
        try:
            result = []
            first_request_id = None  # ✅ 保存第一个 TTS request_id
            
            async for text in llm_chat.process_streaming_content(prompt,llm_model,stop_event, chatname):
                # ✅ 检查是否被打断
                if self.stop is True:
                    logger.warning("检测到停止信号，停止 LLM-to-TTS 流程")
                    llm_chat.is_streaming_done()
                    await asyncio.sleep(0.2)
                    # ✅ 不再放入结束标记，避免与清空逻辑冲突
                    return "".join(result), first_request_id
                
                if text:
                    # 生成 TTS 音频
                    tts_result = await call_TTS(text)
                    
                    # ✅ [FIX] 检查返回值是否有效，防止解包错误
                    if not tts_result or len(tts_result) != 2:
                        logger.error("TTS 返回值异常: %s", tts_result)
                        continue
                        
                    audio, request_id = tts_result  # ✅ 安全解包
                    
                    # ✅ 保存第一个 request_id 作为 session_id
                    # 如果外部传入了 session_id (self.current_session_id)，则优先使用它
                    if first_request_id is None:
                        first_request_id = self.current_session_id if self.current_session_id else request_id
                        logger.info("Session ID (Effective): %s", first_request_id)
                        
                        # ✅ [FIX] 如果之前没有 session_id，现在有了，立即初始化
                        if self.current_session_id is None:
                            self.current_session_id = first_request_id
                            # if self.streaming_callback:
                            #     try:
                            #         self.streaming_callback(None, self.current_session_id, init=True)
                            #         print(f"✅ [LLM] 延迟初始化 Session: {self.current_session_id}")
                            #     except Exception as e:
                            #         print(f"⚠️ [LLM] 延迟初始化 Session 失败: {e}")
                    
                    # ✅ 再次检查是否被打断
                    if self.stop is True:
                        logger.warning("TTS 完成后检测到停止信号，丢弃该音频")
                        llm_chat.is_streaming_done()
                        return "".join(result), first_request_id
                    
                    if audio:
                        # ✅ [新增] 立即推送句子到 UE5，而不是等待播放时
                        # 这样 UE5 可以立刻收到文本，无需等待音频队列
                        # if self.streaming_callback:
                        #     try:
                        #         self.streaming_callback(text, first_request_id)
                        #         print(f"📤 [LLM] 立即推送句子到 UE5: {text[:20]}...")
                        #     except Exception as e:
                        #         print(f"⚠️ [LLM] 推送句子失败: {e}")

                        # ✅ 将文本和音频一起放入队列
                        # 附带 session_id，确保队列处理时能正确关联
                        self.tts_queue.put({
                            "text": text, 
                            "audio": audio,
                            "session_id": first_request_id,
                            "callback": self.streaming_callback  # ✅ 传递 callback
                        })
                        result.append(text)
                        logger.info("已将句子放入 TTS 队列: %s...", text[:20])
                    else:
                        logger.warning("call_TTS 返回空音频")

            # ✅ 正常结束，放入结束标记
            if not self.stop:
                # 放入特殊的结束标记，携带 session_id
                self.tts_queue.put({
                    "type": "end",
                    "session_id": first_request_id,
                    "callback": self.streaming_callback  # ✅ 传递 callback
                })
                logger.info("已放入队列结束标记")
            
            return "".join(result), first_request_id

        except Exception as e:
            logger.exception("Process LLM to TTS error: %s", e)
            # 确保上层知道结束
            try:
                if not self.stop:
                    self.tts_queue.put(None)
            except:
                pass
            return ""
            
    async def start_queue_audio(self, prompt, llm_model, stop_event: asyncio.Event, chatname:str, session_id=None, streaming_callback=None):
        """
        启动音频队列处理
        
        Args:
            session_id: 如果提供,则在播放时推送句子到 UE5
            streaming_callback: 回调函数 callback(sentence, session_id)
        """
        try:
            from threading import Thread
            
            # ✅ 保存 session_id 和回调,用于推送
            self.current_session_id = session_id
            self.streaming_callback = streaming_callback
            
            # ✅ 如果有回调,初始化流式响应数据
            # if session_id and streaming_callback:
            #     streaming_callback(None, session_id, init=True)  # 初始化信号
            
            # ✅ 【关键】启动监控线程（检查线程是否存活）
            if not self.interrupt_monitor_running or (self.interrupt_monitor_thread and not self.interrupt_monitor_thread.is_alive()):
                if self.interrupt_monitor_thread and not self.interrupt_monitor_thread.is_alive():
                    logger.warning("[启动] 监控线程已死亡，重新启动")
                    self.interrupt_monitor_running = False
                
                self.interrupt_monitor_thread = Thread(target=self.interrupt_monitor, daemon=True)
                self.interrupt_monitor_thread.start()
                logger.info("[启动] 打断监控线程已启动")
            else:
                logger.info("[启动] 打断监控线程已运行中")
            
            # ✅ 【关键】启动队列处理线程（检查线程是否存活）
            if not self.queue_processor_running or (self.queue_processor_thread and not self.queue_processor_thread.is_alive()):
                if self.queue_processor_thread and not self.queue_processor_thread.is_alive():
                    logger.warning("[启动] 队列处理线程已死亡，重新启动")
                    self.queue_processor_running = False
                
                self.queue_processor_thread = Thread(target=self.process_audio_queue, daemon=True)
                self.queue_processor_thread.start()
                logger.info("[启动] 队列处理线程已启动")
            else:
                logger.info("[启动] 队列处理线程已运行中")
                logger.info("[启动] 当前队列状态: %s 个待处理项", self.tts_queue.qsize())
            
            # ✅ 生成 TTS（异步）
            response, generated_session_id = await self.process_llm_to_tts(prompt, llm_model, stop_event, chatname)
            logger.info("LLM 生成完成")
            
            # ✅ 对话完成后，自动生成记忆片段
            try:
                logger.info("[记忆] 开始为对话生成记忆片段")
                # 加载当前对话历史
                from utils.llm.llm_utils import load_memory, add_memory, get_system_prompt_with_memory
                history_path, history = load_history(chatname)

                if history and len(history) >= 30:  # 至少有30轮对话
                    # 生成记忆摘要（summarize_memory 会自动处理）
                    from utils.llm.llm_utils import summarize_memory
                    memory_summary, success = summarize_memory(chatname, history)
                    
                    if success:
                        # 添加到记忆库
                        add_memory(chatname, memory_summary)
                        logger.info("[记忆] 成功生成并保存记忆片段: %s...", memory_summary[:50])
                    else:
                        logger.warning("[记忆] 生成记忆摘要失败")
                else:
                    logger.warning("%s条历史，[记忆] 对话历史不足，跳过记忆生成", len(history))
            except Exception as e:
                logger.exception("[记忆] 自动生成记忆失败: %s", e)
            
            # ✅ 标记完成
            # 如果调用时没有传入 session_id，则使用生成的 session_id
            final_session_id = session_id if session_id else generated_session_id
            
            # ❌ 不要在主线程标记完成，而是让队列线程在播放完所有音频后标记
            # if final_session_id and streaming_callback:
            #     streaming_callback(None, final_session_id, completed=True)  # 完成信号
            #     print(f"✅ Session {final_session_id} 完成 (Generation)")
            
            # ✅ 清理
            self.current_session_id = None
            self.streaming_callback = None
            
            # ✅ 返回响应和 session_id
            return response, generated_session_id
        except Exception as e:
            logger.exception("Start queue audio error: %s", e)
            # 清理
            self.current_session_id = None
            self.streaming_callback = None
            # if session_id and streaming_callback:
            #     try:
            #         streaming_callback(None, session_id, completed=True, error=str(e))
            #     except:
            #         pass
            return ""
