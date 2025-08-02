import requests
from threading import Thread
from queue import Queue
from openai import OpenAI
from utils.llm.llm_utils import *
from utils.llm.text_utils import *
from utils.utils import init_api_key
import os
import io
import asyncio
import time



class LLMChat():
    def __init__(self):
        self.history = []
        self.on_streaming = True
        self.system_prompt=load_text(r"E:\Unreal Projects\lipsync_neurosync\unrealversion\config_files\sys_prompt.txt")
        self.llm_url = "https://api.tu-zi.com/v1"
        self.llm_key = init_api_key(env_name="tuzi_api_key", input_api_key="your llm api key")
        self.seed = 42
        self.temperature =0.7
        self.buffer_text =""

    def is_streaming_done(self):
        self.on_streaming=False
        print("llm停止事件设置成功")
        return not self.on_streaming

        
    async def stream_llm_request(self, prompt, llm_model, stop_event: asyncio.Event, chatname:str):
        """llm请求函数, 需要句子处理时不调用，直接调用句子处理函数"""
        path, local_history = load_history(chatname)
        self.history = local_history
        self.history.append({'role': 'user', 'content': prompt})
        messages = [{'role': 'system', 'content': self.system_prompt}]
        messages.extend(self.history)
        last_chunk = time.time()
        # 模拟远程调用，流式返回数据
        client = OpenAI(api_key=self.llm_key, base_url=self.llm_url)
        completion = client.chat.completions.create(
            model=llm_model,
            seed=self.seed,
            temperature=self.temperature,
            messages=messages,
            stream=True
        )
        full_content = ""
        # 假设返回的每个 chunk 代表流式数据的一部分
        for event in completion:
            if not self.on_streaming:
                completion = []
                client.close()
                self.buffer_text = ""
                print("Stop signal received, LLM client closed")
                self.on_streaming=True
                break
            content = event.choices[0].delta.content or ""
            chunk_dur =time.time()-last_chunk
            yield content  # 返回流式内容
            full_content += content
            #print(f"本段收取用时 {chunk_dur}")
            last_chunk = time.time()
        self.history.append({'role': 'assistant', 'content': full_content})
        save_history(chatname, self.history)

    async def process_streaming_content(self, prompt,llm_model, stop_event, chatname):
        self.on_streaming=True
        #请求llm并异步处理句子，每个句段收取后就进行检测#   
        result_list = []  # 用来保存符合条件的文段
        self.buffer_text = ""  # 用来拼接不符合条件的文本
        last_check = time.time()           
        async for text in self.stream_llm_request(prompt, llm_model, stop_event, chatname):
            if not text:
                continue
            else:
                self.buffer_text += text

            if stop_event.is_set():
                self.buffer_text = ""
                print("文本缓存清理成功")

            if time.time() - last_check >= 0:
                cleaned_buffer = re.sub(r'\s+', '', self.buffer_text)
                if len(cleaned_buffer) > 8 and re.search(r'[，。！？；]', self.buffer_text):
                    #print(f"收取完整句子 {cleaned_buffer}, 长度为 {len(cleaned_buffer)}")
                    match = re.search(r'[，。！？；]', self.buffer_text[::-1])
                    if match:
                        end_pos = len(self.buffer_text) - match.start()
                        sentence = self.buffer_text[:end_pos]
                        if len(re.sub(r'\s+', '', sentence)) > 6:  # 前半句去空格后>8
                            yield sentence
                            result_list.append(sentence)
                            self.buffer_text = self.buffer_text[end_pos:].strip()
                        #else:
                            #print(f"前半句长度不够8, 继续拼接")
                    #else:
                        #print("无标点，继续拼接下一个内容")
                    last_check = time.time()
        clean_list = process_for_tts(result_list)
        self.on_streaming=False
        #print("最终处理列表：", result_list)
        #return result_list

        # 这个类将替代您的原始 LLMChat 类

chat_list = {
    "defult" :"defult",
    "吉他教室": "guitar",
    "语言教室": "japanese",
    "职业规划": "career",
    "xxxooo": "xxxooo",
    "闲聊": "small_talk",  
}

def load_history(chatname):
    file_name = chat_list.get(chatname)
    chat_path = os.path.join("dialogue_histories", f"{file_name}.txt")
    path = resolve_path(chat_path)
    print(path)
    history = load_text(r"{}".format(path))
    # 解析 JSON 字符串
    history = json.loads(history)
    return path, history

def save_history(chatname, history):
    file_name = chat_list.get(chatname)
    chat_path = os.path.join("dialogue_histories", f"{file_name}.txt")
    path = resolve_path(chat_path)
    # 将 history 转为 JSON 字符串并写入文件
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
    print(f"历史记录保存成功：{path}")
    return "saved history"

def process_for_tts(text_list):
    # 定义允许的标点符号：中文和英文的标点
    allowed_punctuation = r'[，。！？：；“”""\'\'\n]'

    # 过滤掉不需要发音的特殊字符和空格
    cleaned_text = []
    
    for sentence in text_list:
        # 去除不需要的特殊字符（例如 *, #, @ 等）和空格
        sentence = re.sub(r'[^a-zA-Z0-9' + allowed_punctuation + r']|\s+', '', sentence)
        cleaned_text.append(sentence)   
    return cleaned_text