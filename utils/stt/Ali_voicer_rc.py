import pyaudio
from pathlib import Path
import time
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
import asyncio
from queue import Queue
import queue
from threading import Thread
from utils.logging_config import get_logger

logger = get_logger(__name__)

class Callback_rc(RecognitionCallback):
    def __init__(self):
        self.stop_llm_generate = False
        self.stop_event = None

        self.first_sentence_time = None
        self.last_sentence_time = None
        self.final_texts = []
        self.new_sentence = None
        self.pending_paragraph = None
        self.mic = None
        self.stream = None
        # ✅ 修改：增加容差时间，允许正常停顿
        self.SILENT_TIMEOUT = 1  # 静音超时（快速连续说话的阈值）
        self.MAX_SENTENCE_DURATION = 5  # 最大句子间隔（新对话判定）
        self.WAIT_FOR_CONTINUATION = 3  # ⭐ 新增：等待后续句子的时间（允许正常停顿）
        self.rc_queue = queue.Queue()
        self.new_input = None
        self.first_run = False
        # ✅ 新增：用于清空 TTS 队列的回调
        self.clear_tts_callback = None

    def set_stop_event(self, stop_event: asyncio.Event):
        if self.stop_event is None:
            self.stop_event = asyncio.Event()
        self.stop_event = stop_event

    def stop_llm(self):
        if self.new_input:
            self.stop_llm_generate = True
            self.new_input = None
        return self.stop_llm_generate

    def on_open(self) -> None:
        logger.info('RecognitionCallback opening')
        try:
            # ✅ 如果已有旧的流，先关闭
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
                self.stream = None
            
            if self.mic:
                try:
                    self.mic.terminate()
                except:
                    pass
                self.mic = None
            
            # 创建新的音频对象
            logger.info('正在初始化 PyAudio')
            self.mic = pyaudio.PyAudio()
            
            logger.info('正在打开音频流')
            self.stream = self.mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=3200  # ✅ 添加缓冲区大小
            )
            logger.info('RecognitionCallback opened successfully')
            
        except Exception as e:
            logger.exception('RecognitionCallback on_open 失败: %s', e)
            # 清理失败的资源
            if self.stream:
                try:
                    self.stream.close()
                except:
                    pass
                self.stream = None
            if self.mic:
                try:
                    self.mic.terminate()
                except:
                    pass
                self.mic = None
            raise  # 重新抛出,让上层知道失败了

    def on_close(self) -> None:
        logger.info('RecognitionCallback closing')
        try:
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                    logger.info('Stream closed')
                except Exception as e:
                    logger.warning('Stream close error: %s', e)
                finally:
                    self.stream = None
            
            if self.mic:
                try:
                    self.mic.terminate()
                    logger.info('Mic terminated')
                except Exception as e:
                    logger.warning('Mic terminate error: %s', e)
                finally:
                    self.mic = None
            
            logger.info('RecognitionCallback closed')
        except Exception as e:
            logger.exception('RecognitionCallback on_close 错误: %s', e)

    def on_complete(self) -> None:
        logger.info('RecognitionCallback completed')

    def on_error(self, message) -> None:
        logger.error('RecognitionCallback error: %s', message.message)

    def on_event(self, result: RecognitionResult, *args, **kwargs) -> None:
        self.new_input = result.get_sentence()  #用于储存实时输入的变量，为不完整句子
        
        if 'text' in self.new_input and RecognitionResult.is_sentence_end(self.new_input):
            self.first_sentence_time = time.time()  #马上记录收取句子时间
            self.stop_llm_generate = False
            self.new_sentence = self.new_input.get("text") #只有完整句子时才会被赋值
            logger.info("收到完整句子: %s", self.new_sentence)
            self.new_input = None
            judger_thread = Thread(target=asyncio.run, args=(self.time_judger(),))
            judger_thread.daemon = True  # ✅ 设置为守护线程，避免阻塞
            judger_thread.start()

    async def time_judger(self):
        """
        句子拼接判断逻辑
        
        ✅ 优化后的时间容差：
        - WAIT_FOR_CONTINUATION (3秒): 等待后续句子的时间，允许正常停顿
        - SILENT_TIMEOUT (1秒): 快速连续说话的阈值
        - MAX_SENTENCE_DURATION (5秒): 新对话判定阈值
        """
        logger.info("句子判断启动")
        if self.new_input:  # 防止句子不完整就进入判断
            logger.warning("句子不完整，跳过判断")
            return
        
        # ✅ 获取完整句子并立即清空，避免重复处理
        sentence = self.new_sentence if isinstance(self.new_sentence, str) else None
        self.new_sentence = None
        
        if not sentence:
            logger.warning("无有效句子，跳过")
            return
        
        current_time = time.time()
        self.stop_event = asyncio.Event()
        send = False
        
        # ✅ 初始化（第一句话）
        if self.last_sentence_time is None:
            self.first_run = True
            self.last_sentence_time = self.first_sentence_time - 0.5
            self.final_texts.clear()
            logger.info("首句，重置状态")
        else:
            self.first_run = False
        
        # ✅ 添加当前句子到缓冲区
        self.final_texts.append(sentence)
        
        # ✅ 计算时间间隔
        time_diff = self.first_sentence_time - self.last_sentence_time if self.first_sentence_time else 0
        logger.info("时间间隔: %.2f秒 | 累积句子数: %s", time_diff, len(self.final_texts))
        
        # ✅ 判断逻辑
        if len(self.final_texts) == 1:
            # 首句：无论间隔多久，都先打断旧对话（如果有的话）
            logger.info("首句（间隔 %.2f秒），先打断旧对话", time_diff)
            self.stop_llm_generate = True
            if self.clear_tts_callback:
                self.clear_tts_callback()  # 清空 TTS 队列和停止 LLM
            
            # 然后等待看是否有后续句子
            if time_diff <= self.MAX_SENTENCE_DURATION:
                # 间隔不大，可能是连续说话，等待拼接
                logger.info("等待 %s 秒（允许正常停顿）", self.WAIT_FOR_CONTINUATION)
                await asyncio.sleep(self.WAIT_FOR_CONTINUATION)
                if self.new_input:  # 有新输入，返回等待拼接
                    logger.debug("检测到新输入，等待拼接")
                    return
            send = True
            
        elif time_diff < self.SILENT_TIMEOUT:
            # <1秒：快速连续说话，继续累积
            logger.info("间隔<1秒，快速连续说话，继续累积")
            await asyncio.sleep(self.WAIT_FOR_CONTINUATION)  # ✅ 使用新的等待时间
            if self.new_input:
                logger.debug("检测到新输入，等待拼接")
                return
            # ✅ 打断当前对话并清空 TTS 队列
            logger.warning("触发打断机制：停止 LLM 并清空 TTS 队列")
            self.stop_llm_generate = True
            if self.clear_tts_callback:
                self.clear_tts_callback()  # 清空 TTS 队列
            send = True
            
        elif time_diff < self.MAX_SENTENCE_DURATION:
            # 1-5秒：正常拼接（允许停顿思考）
            logger.info("间隔 %.2f秒（正常停顿），等待 %s 秒后拼接", time_diff, self.WAIT_FOR_CONTINUATION)
            self.stop_event.set()
            await asyncio.sleep(self.WAIT_FOR_CONTINUATION)  # ✅ 使用新的等待时间
            if self.new_input:
                logger.debug("检测到新输入，等待拼接")
                return
            # ✅ 打断当前对话并清空 TTS 队列
            logger.warning("触发打断机制：停止 LLM 并清空 TTS 队列")
            self.stop_llm_generate = True
            if self.clear_tts_callback:
                self.clear_tts_callback()
            send = True
            
        else:
            # >5秒：新对话，清空旧内容
            logger.info("间隔 %.2f秒（>5秒），视为新对话", time_diff)
            self.stop_event.set()
            await asyncio.sleep(self.WAIT_FOR_CONTINUATION)  # ✅ 使用新的等待时间
            if self.new_input:
                logger.debug("检测到新输入，等待拼接")
                return
            # ✅ 打断当前对话并清空 TTS 队列
            logger.warning("触发打断机制：停止 LLM 并清空 TTS 队列")
            self.stop_llm_generate = True
            if self.clear_tts_callback:
                self.clear_tts_callback()
            # ✅ 清空旧句子，只保留当前句子
            self.final_texts.clear()
            self.final_texts.append(sentence)
            self.last_sentence_time = None
            send = True
        
        # ✅ 发送到队列
        if send:
            paragraph = ''.join(self.final_texts)
            
            # ✅ 【关键修复】如果拼接后是空字符串，跳过发送
            if not paragraph or paragraph.strip() == "":
                logger.warning("拼接后为空，跳过发送")
                return
            
            # 清空队列
            while not self.rc_queue.empty():
                self.rc_queue.get()
                self.rc_queue.task_done()
            
            self.rc_queue.put(paragraph)
            logger.info("已加入队列: [%s]", paragraph)
            
            # ✅ 发送后清空缓冲区
            self.final_texts.clear()
            self.last_sentence_time = self.first_sentence_time
        
        logger.info("句子判断结束")
        return


    def on_event44(self, result: RecognitionResult, *args, **kwargs) -> None:
        self.new_input = result.get_sentence()
        if 'text' in self.new_input and RecognitionResult.is_sentence_end(self.new_input):
            self.final_texts.append(self.new_input['text'])
            paragraph = ''.join(self.final_texts)
            while not self.rc_queue.empty():  # 清空队列
                self.rc_queue.get()
                self.rc_queue.task_done()    
                logger.info("已清空录音队列")
            self.stop_llm_generate = True
            self.rc_queue.put(paragraph)  # 塞进队列
            logger.info("请求已加入队列：%s", paragraph)
            self.final_texts.clear()
