import requests
from threading import Thread
from queue import Queue
from openai import OpenAI
from utils.llm.llm_utils import *
from utils.llm.text_utils import *
from utils.llm.chat_list_manager import get_file_name, get_all_chat_names
from utils.utils import init_api_key
import os
import io
import asyncio
import time
import json
import re
from utils.logging_config import get_logger
from utils.llm.chat_utils import create_new_chat

logger = get_logger(__name__)



class LLMChat():
    def __init__(self):
        self.history = []
        self.on_streaming = True
        self.last_stream_aborted = False
        # ✅ 动态加载系统提示词：从项目根目录的 config_files/sys_prompt.txt
        root_dir = resolve_path(os.path.join(os.path.dirname(__file__), "..", ".."))
        sys_prompt_path = os.path.join(root_dir, "config_files", "sys_prompt.txt")
        self.system_prompt = load_text(sys_prompt_path) if os.path.exists(sys_prompt_path) else ""
        if not self.system_prompt:
            logger.warning("无法从 %s 加载系统提示词，使用空字符串", sys_prompt_path)
        self.llm_url = "https://api.tu-zi.com/v1"
        self.llm_key = init_api_key(env_name="tuzi_api_key", input_api_key="your llm api key")
        self.seed = 42
        self.temperature =0.7
        self.buffer_text =""

    def is_streaming_done(self):
        self.on_streaming=False
        logger.info("LLM 停止事件设置成功")
        return not self.on_streaming

        
    async def stream_llm_request(self, prompt, llm_model, stop_event: asyncio.Event, chatname:str, skip_history_on_abort: bool = True):
        """llm请求函数, 需要句子处理时不调用，直接调用句子处理函数"""
        path, local_history = load_history(chatname)
        self.history = local_history
        self.history.append({'role': 'user', 'content': prompt})
        
        # ✅ 生成带有记忆片段的系统提示词
        system_prompt_with_memory = get_system_prompt_with_memory(chatname, self.system_prompt)
        
        messages = [{'role': 'system', 'content': system_prompt_with_memory}]
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
        aborted = False
        # 假设返回的每个 chunk 代表流式数据的一部分
        try:
            for event in completion:
                if stop_event.is_set() or not self.on_streaming:
                    completion = []
                    client.close()
                    self.buffer_text = ""
                    aborted = True
                    logger.info("Stop signal received, LLM client closed")
                    self.on_streaming=True
                    break
                
                # ✅ [FIX] 安全检查：确保 choices 存在且不为空
                if not event.choices:
                    continue
                    
                content = event.choices[0].delta.content or ""
                chunk_dur =time.time()-last_chunk
                full_content += content
                yield content  # 返回流式内容
                #print(f"本段收取用时 {chunk_dur}")
                last_chunk = time.time()
        except GeneratorExit:
            # 上层放弃了 generator（如 TTS 中断后 process_llm_to_tts return）
            # LLM 可能已经完成，full_content 可能是完整的
            logger.info("[历史] Generator 被上层关闭，进入 finally 保存历史")
            raise
        finally:
            # 无论正常完成、break 中断、还是 GeneratorExit，都保存历史
            try:
                self.last_stream_aborted = aborted
                if not aborted:
                    self.history.append({'role': 'assistant', 'content': full_content})
                    save_history(chatname, self.history)
                else:
                    if not skip_history_on_abort:
                        if full_content:
                            self.history.append({'role': 'assistant', 'content': full_content})
                            logger.info("[历史] 请求中断但保留部分历史: chat=%s", chatname)
                        else:
                            logger.info("[历史] 请求中断无返回文本，保留用户输入: chat=%s", chatname)
                        save_history(chatname, self.history)
                    else:
                        logger.info("[历史] 请求中断，跳过保存历史: chat=%s", chatname)
            except Exception as e:
                logger.warning("[历史] finally 保存历史时异常: %s", e)

    async def process_streaming_content(self, prompt,llm_model, stop_event, chatname, skip_history_on_abort: bool = True):
        self.on_streaming=True
        #请求llm并异步处理句子，每个句段收取后就进行检测#   
        result_list = []  # 用来保存符合条件的文段
        self.buffer_text = ""  # 用来拼接不符合条件的文本
        last_check = time.time()           
        async for text in self.stream_llm_request(
            prompt,
            llm_model,
            stop_event,
            chatname,
            skip_history_on_abort=skip_history_on_abort,
        ):
            if not text:
                continue
            else:
                self.buffer_text += text

            if stop_event.is_set():
                self.buffer_text = ""
                logger.info("文本缓存清理成功")

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


def _sanitize_file_name(name: str) -> str:
    """Create a filesystem-safe file stem for chat history files."""
    safe = re.sub(r'[^\w\-\u4e00-\u9fff]+', '_', (name or '').strip(), flags=re.UNICODE)
    safe = safe.strip('_')
    return safe or 'default'


def ensure_chat_context(chatname: str) -> str:
    """Ensure chat mapping, history file, and memory files exist for the chat."""
    display_name = (chatname or 'default').strip() or 'default'

    file_name = get_file_name(display_name)
    if not file_name:
        file_name = _sanitize_file_name(display_name)
        ok, message = create_new_chat(display_name, file_name)
        if ok:
            logger.info("自动创建聊天上下文成功: %s -> %s", display_name, file_name)
        else:
            logger.warning("自动创建聊天上下文返回: %s", message)

        # Re-fetch from source of truth after create attempt.
        file_name = get_file_name(display_name) or file_name

    # Ensure history file exists.
    chat_path = os.path.join("dialogue_histories", f"{file_name}.txt")
    history_path = resolve_path(chat_path)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    if not os.path.exists(history_path):
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=4)
        logger.info("自动创建历史文件: %s", history_path)

    # Ensure new memory file exists (fragment list format).
    try:
        memory_path = get_memory_file_path(display_name)
        if not os.path.exists(memory_path):
            with open(memory_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)
            logger.info("自动创建记忆文件: %s", memory_path)
    except Exception as e:
        logger.warning("自动创建记忆文件失败: %s", e)

    # Ensure legacy memory file exists for compatibility path.
    legacy_memory_path = resolve_path(os.path.join("dialogue_histories", f"{file_name}_memory.json"))
    if not os.path.exists(legacy_memory_path):
        with open(legacy_memory_path, 'w', encoding='utf-8') as f:
            json.dump({"summary": "", "updated_at": get_timestamp()}, f, ensure_ascii=False, indent=4)
        logger.info("自动创建兼容记忆文件: %s", legacy_memory_path)

    return file_name


def load_history(chatname):
    """
    Load dialogue history for a given chat name
    
    Args:
        chatname: The display name of the chat
    
    Returns:
        tuple: (path, history_list)
    """
    # ✅ 动态从 chat_list.json 获取文件名（缺失时自动创建）
    file_name = ensure_chat_context(chatname)
    
    chat_path = os.path.join("dialogue_histories", f"{file_name}.txt")
    path = resolve_path(chat_path)
    logger.info("加载历史路径: %s", path)
    
    try:
        history_text = load_text(r"{}".format(path))
        
        # 处理空文件或空字符串
        if not history_text or history_text.strip() == "":
            logger.warning("历史文件为空或不存在: %s", path)
            return path, []
        
        # 解析 JSON 字符串
        history = json.loads(history_text)
        
        # 确保返回的是列表
        if not isinstance(history, list):
            logger.warning("历史记录不是列表格式，重置为空列表")
            return path, []
        
        return path, history
        
    except json.JSONDecodeError as e:
        logger.warning("JSON 解析失败: %s，初始化为空列表", e)
        return path, []
    except Exception as e:
        logger.exception("加载历史记录失败: %s，初始化为空列表", e)
        return path, []


def save_history(chatname, history):
    """
    Save dialogue history for a given chat name
    
    Args:
        chatname: The display name of the chat
        history: The history list to save
    
    Returns:
        str: Success message
    """
    # ✅ 动态从 chat_list.json 获取文件名（缺失时自动创建）
    file_name = ensure_chat_context(chatname)
    
    chat_path = os.path.join("dialogue_histories", f"{file_name}.txt")
    path = resolve_path(chat_path)
    
    # 将 history 转为 JSON 字符串并写入文件
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=4)
    logger.info("历史记录保存成功：%s", path)
    return "saved history"

def process_for_tts(text_list):
    """
    处理文本列表以供 TTS 使用。
    对每个句子进行清理，移除 Markdown、推理标签等格式化符号。
    """
    cleaned_text = []
    
    for sentence in text_list:
        # 使用 clean_for_tts 进行全面清理
        cleaned_sentence = clean_for_tts(sentence)
        if cleaned_sentence:  # 只添加非空的句子
            cleaned_text.append(cleaned_sentence)
    
    return cleaned_text