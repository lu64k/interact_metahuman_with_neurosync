import pyaudio
from pathlib import Path
import time
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
import asyncio
from queue import Queue
import queue
from threading import Thread

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
        self.SILENT_TIMEOUT = 1
        self.MAX_SENTENCE_DURATION = 5
        self.rc_queue = queue.Queue()
        self.new_input = None
        self.first_run = False

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
        print('RecognitionCallback open.')
        self.mic = pyaudio.PyAudio()
        self.stream = self.mic.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True
        )

    def on_close(self) -> None:
        print('RecognitionCallback close.')
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.mic:
            self.mic.terminate()
        self.stream = None
        self.mic = None

    def on_complete(self) -> None:
        print('RecognitionCallback completed.')

    def on_error(self, message) -> None:
        print(f'RecognitionCallback error: {message.message}')

    def on_event(self, result: RecognitionResult, *args, **kwargs) -> None:
        self.new_input = result.get_sentence()  #用于储存实时输入的变量，为不完整句子
        
        if 'text' in self.new_input and RecognitionResult.is_sentence_end(self.new_input):
            self.first_sentence_time = time.time()  #马上记录收取句子时间
            self.stop_llm_generate = False
            self.new_sentence = self.new_input.get("text") #只有完整句子时才会被赋值
            print(self.new_sentence)
            self.new_input = None
            judger_thread = Thread(target=asyncio.run, args=(self.time_judger(),))
            #judger_thread.daemon = True
            judger_thread.start()

    async def time_judger(self):
        print("句子判断启动")
        if self.new_input: #防止句子不完整就进入判断
            return        
        sentence = self.new_sentence if isinstance(self.new_sentence, str) else None#timejudger拿到完整句子
        self.new_input = None #拿取后马上清空
         #拿取后马上清空
        current_time = time.time()
        self.stop_event = asyncio.Event()

        send = False #发送归位 可考虑删除该值
        if sentence: #确认已经有新完整句子开始判断
            if self.last_sentence_time is None:#如果是第一句的话首句和末尾句子都还没有赋值
                self.first_run = True
                self.last_sentence_time = self.first_sentence_time - 0.5 #假设一个历史句子
                self.final_texts.clear()
                print("句子复位")
            else:
                self.first_run = False
            self.final_texts.append(sentence) #拼接句子
            

            time_diff = self.first_sentence_time - self.last_sentence_time if self.first_sentence_time else 0
            print(time_diff)
            #时间间隔就是最新识别的句子减去上一个句子结束的时间

            if len(self.final_texts) == 1:  # 首句直接放队列
                self.last_sentence_time = self.first_sentence_time#把最后一个句子的时间设置为刚刚收取句子的时间 
                self.first_sentence_time = None
                await asyncio.sleep(1)
                if self.new_input:
                    return
                self.new_sentence = None
                send = True
                

            else:#如果不是第一句
                if time_diff < self.SILENT_TIMEOUT:  # 小于1秒，拼接不发，不打断
                    self.last_sentence_time = self.first_sentence_time#更新历史句子时间
                    print("收到句子与上句间隔小于  1秒")
                    await asyncio.sleep(1)
                    if self.new_input:
                        return
                    self.stop_llm_generate = True  
                    send = True
                    
                elif time_diff < self.MAX_SENTENCE_DURATION:  # 1-5秒，拼接回初始逻辑
                    self.stop_event.set()
                    self.last_sentence_time = self.first_sentence_time#更新历史句子时间
                    print("收到句子与上句间隔小于  5秒")
                    await asyncio.sleep(1)#等一秒是否有新句子输入
                    if self.new_input:
                        return                    
                    self.stop_llm_generate = True
                    send = True

                else:  # 超5秒，新任务
                    self.final_texts.clear()
                    self.stop_event.set()                    
                    print("超五秒新请求")
                    await asyncio.sleep(1)
                    if self.new_input:
                        return
                    self.stop_llm_generate = True
                    self.final_texts.append(sentence)          
                    send = True
                
            if send:
                paragraph = ''.join(self.final_texts)
                while not self.rc_queue.empty():  # 清空队列
                    self.rc_queue.get()
                    self.rc_queue.task_done()
                    print(f"已清空录音队列")
                
                self.rc_queue.put(paragraph)  # 塞进队列
                print(f"请求已加入队列：{paragraph}")
                send = False
                #此时不能清空历史消息，因为有5秒内拼接规则，所以只有新输入大于五秒才会被彻底清理
                self.last_sentence_time = self.first_sentence_time#更新历史句子时间
        print("句子判断结束")
        return


    def on_event44(self, result: RecognitionResult, *args, **kwargs) -> None:
        self.new_input = result.get_sentence()
        if 'text' in self.new_input and RecognitionResult.is_sentence_end(self.new_input):
            self.final_texts.append(self.new_input['text'])
            paragraph = ''.join(self.final_texts)
            while not self.rc_queue.empty():  # 清空队列
                self.rc_queue.get()
                self.rc_queue.task_done()    
                print(f"已清空录音队列")
            self.stop_llm_generate = True
            self.rc_queue.put(paragraph)  # 塞进队列
            print(f"请求已加入队列：{paragraph}")
            self.final_texts.clear()
                