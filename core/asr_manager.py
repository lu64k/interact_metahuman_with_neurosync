import asyncio
import threading
import time
import dashscope
from dashscope.audio.asr import Recognition
import aiohttp.client_exceptions
from threading import Thread
from utils.utils import init_api_key
from utils.logging_config import get_logger

logger = get_logger(__name__)

class ASRManager:
    def __init__(self, callback_rc, audio_que, push_sentence_callback=None):
        self.callback_rc = callback_rc
        self.audio_que = audio_que
        self.push_sentence_callback = push_sentence_callback
        self.recognition = None
        self.recording_thread = None
        self.rc_queue = callback_rc.rc_queue
        
        # Current config to be managed
        self.current_config = {
            'model': 'gemini-3-flash-preview',
            'chats': 'default'
        }
        self.config_lock = threading.Lock()

    def clear_tts_queue(self):
        """Clear TTS queue and stop current streaming generation."""
        self.audio_que.interrupt_seq += 1
        interrupt_seq = self.audio_que.interrupt_seq
        queue_size_before = self.audio_que.tts_queue.qsize()
        logger.info("[实时打断] 清空 TTS 队列并停止 LLM 生成")
        logger.debug(
            "[R1] interrupt-trigger seq=%s stop_before=%s queue_size_before=%s has_stop_event=%s",
            interrupt_seq,
            self.audio_que.stop,
            queue_size_before,
            bool(self.audio_que.current_stop_event),
        )

        self.audio_que.stop = True

        if self.audio_que.current_stop_event:
            try:
                self.audio_que.current_stop_event.set()
                logger.info("已设置 stop_event，LLM 流式生成将停止")
            except Exception as e:
                logger.warning("设置 stop_event 失败: %s", e)

        cleared_count = 0
        while not self.audio_que.tts_queue.empty():
            try:
                self.audio_que.tts_queue.get_nowait()
                self.audio_que.tts_queue.task_done()
                cleared_count += 1
            except Exception:
                break
        if cleared_count > 0:
            logger.info("已清空 %s 个 TTS 队列项", cleared_count)

        logger.info("[实时打断] 打断完成")

    def init_recognition(self, env_name='DASHSCOPE_API_KEY', input_api_key="输入你的api"):
        """初始化语音识别"""
        try:
            logger.info("正在初始化 Dashscope API (from ASRManager)")
            dashscope.api_key = init_api_key(env_name, input_api_key)
            
            logger.info("正在创建 Recognition 对象")
            self.recognition = Recognition(
                model='paraformer-realtime-v2',
                format='pcm',
                sample_rate=16000,
                heartbeat=True,
                callback=self.callback_rc
            )
            
            logger.info("正在启动 Recognition")
            self.recognition.start()
            
            logger.info("Recording started successfully")
            return True
            
        except Exception as e:
            logger.exception("init_recognition 失败: %s", e)
            self.recognition = None
            raise

    async def process_rc_queue(self, llm_model, chatname, session_id=None):
        """处理语音识别队列"""
        logger.info("Start processing queue (session: %s)", session_id)
        
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
            logger.warning("[队列处理] 初始 session_id 为空，生成新的: %s", session_id)
        
        while True:
            if self.rc_queue.empty():
                await asyncio.sleep(0.1)
                if self.rc_queue.qsize() == 0 and not self.callback_rc.stream:
                    logger.info("Queue clean, stopping")
                    break
                continue
            
            try:
                logger.info("[新请求] 重置打断标志")
                self.callback_rc.stop_llm_generate = False
                self.audio_que.stop = False
                
                await asyncio.sleep(0.2)
                
                prompt = self.rc_queue.get()
                logger.info("新的文本生成：%s", prompt)
                
                if not prompt or (isinstance(prompt, str) and prompt.strip() == ""):
                    logger.warning("[队列处理] 收到空 prompt，跳过处理")
                    self.rc_queue.task_done()
                    continue
                
                stop_event = asyncio.Event()
                self.audio_que.current_stop_event = stop_event
                
                with self.config_lock:
                    active_model = self.current_config.get('model', llm_model)
                    active_chats = self.current_config.get('chats', chatname)
                
                logger.info("[队列处理] 使用配置: model=%s, chats=%s", active_model, active_chats)
                
                # ✅ 使用传参进来的回调函数
                callback = self.push_sentence_callback
                
                result = await self.audio_que.start_queue_audio(
                    prompt, active_model, stop_event, active_chats, session_id, callback
                )
                
                if result:
                    _, generated_session_id = result
                    if generated_session_id:
                        logger.info("[队列处理] 生成的 Session ID: %s", generated_session_id)
                
                self.audio_que.current_stop_event = None

            except Exception as e:
                logger.exception("Queue processing error: %s", e)
            finally:
                self.rc_queue.task_done()

    async def async_recording(self):
        """异步录音循环"""
        loop = asyncio.get_event_loop()
        logger.info("async_recording 开始")
        
        try:
            retry_count = 0
            while not self.callback_rc.stream and retry_count < 10:
                logger.info("等待音频流准备... (%s/10)", retry_count + 1)
                await asyncio.sleep(0.5)
                retry_count += 1
            
            if not self.callback_rc.stream:
                logger.error("音频流未能在5秒内准备好,退出录音")
                return "stream_not_ready"
            
            while self.recognition and self.callback_rc.stream:
                try:
                    if not self.callback_rc.stream.is_active():
                        logger.warning("音频流已停止")
                        break
                    
                    data = await loop.run_in_executor(
                        None, 
                        lambda: self.callback_rc.stream.read(3200, exception_on_overflow=False)
                    )
                    
                    if self.recognition:
                        self.recognition.send_audio_frame(data)

                    if self.callback_rc.stop_llm_generate == True and not self.callback_rc.first_run:
                        logger.info("[实时打断] 检测到新语音，准备清空队列")
                        await asyncio.sleep(0.5)
                        self.callback_rc.stop_llm_generate = False
                        logger.info("[实时打断] 准备接收新对话")
                        
                except aiohttp.client_exceptions.ClientConnectionResetError:
                    logger.warning("WebSocket connection reset")
                    self._cleanup_recognition()
                    return "connection_reset"
                except Exception as e:
                    logger.exception("Async recording error: %s", e)
                    self._cleanup_recognition()
                    return "error"
        except Exception as e:
            logger.exception("Async recording loop error: %s", e)
            self._cleanup_recognition()
        
        logger.info("async_recording 结束")
        return "completed"

    def record_and_recognize_with_stream(self, if_end, llm_model, chatname, session_id=None):
        """带流式支持的录音控制"""
        try:
            if not if_end:
                logger.info("开始录音 (session: %s)", session_id)
                self._cleanup_resources()
                
                with self.config_lock:
                    self.current_config['model'] = llm_model
                    self.current_config['chats'] = chatname
                
                # Reset Callback state
                self._reset_callback_state()
                
                time.sleep(0.5)
                
                # Start processing thread
                stt_thread = Thread(target=asyncio.run, args=(self.process_rc_queue(llm_model, chatname, session_id),))
                stt_thread.daemon = True
                stt_thread.start()
                
                self.init_recognition()
                
                # Start recording thread
                self.recording_thread = Thread(target=asyncio.run, args=(self.async_recording(),))
                self.recording_thread.daemon = True
                self.recording_thread.start()
                
                return "", [], "Recording started"
            else:
                logger.info("停止录音")
                self._cleanup_recognition()
                
                if self.recording_thread:
                    self.recording_thread.join(timeout=3)
                    self.recording_thread = None
                
                wait_start = time.time()
                while self.callback_rc.new_sentence is not None and time.time() - wait_start < 2:
                    time.sleep(0.1)
                
                return "", [], "Recording stopped"
        except Exception as e:
            logger.exception("record_and_recognize_with_stream 错误: %s", e)
            self._cleanup_resources()
            return "", [], f"Error: {str(e)}"

    def record_and_recognize(self, if_end, llm_model, chatname, session_id=None):
        """兼容旧版的录音控制"""
        return self.record_and_recognize_with_stream(if_end, llm_model, chatname, session_id)

    def _cleanup_recognition(self):
        if self.recognition:
            try:
                self.recognition.stop()
            except:
                pass
            self.recognition = None

    def _cleanup_resources(self):
        self._cleanup_recognition()
        
        if self.callback_rc.stream:
            try:
                self.callback_rc.stream.stop_stream()
                self.callback_rc.stream.close()
            except:
                pass
            self.callback_rc.stream = None
        
        if self.callback_rc.mic:
            try:
                self.callback_rc.mic.terminate()
            except:
                pass
            self.callback_rc.mic = None

    def _reset_callback_state(self):
        self.callback_rc.final_texts.clear()
        self.callback_rc.new_input = None
        self.callback_rc.new_sentence = None
        self.callback_rc.stop_llm_generate = False
        self.callback_rc.first_run = False
        self.callback_rc.first_sentence_time = None
        self.callback_rc.last_sentence_time = None
        
        while not self.rc_queue.empty():
            try:
                self.rc_queue.get_nowait()
                self.rc_queue.task_done()
            except:
                break
        
        self.audio_que.stop = True
        while not self.audio_que.tts_queue.empty():
            try:
                self.audio_que.tts_queue.get_nowait()
                self.audio_que.tts_queue.task_done()
            except:
                break
