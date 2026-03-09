# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.

import requests
from threading import Thread
from queue import Queue
from openai import OpenAI
import json
import os
import shutil
from utils.utils import init_api_key
from utils.llm.sentence_builder import SentenceBuilder
from utils.llm.text_utils import resolve_path, load_json_file, load_text
from utils.utils import get_timestamp




def warm_up_llm_connection(config):
    """
    Perform a lightweight dummy request to warm up the LLM connection.
    This avoids the initial delay when the user sends the first real request.
    """
    if config["USE_LOCAL_LLM"]:
        try:
            # For local LLM, use a dummy ping request with a short timeout.
            requests.post(config["LLM_STREAM_URL"], json={"dummy": "ping"}, timeout=1)
            print("Local LLM connection warmed up.")
        except Exception as e:
            print("Local LLM connection warm-up failed:", e)
    else:
        try:
            # For OpenAI API, send a lightweight ping message.
            client = OpenAI(api_key=config["OPENAI_API_KEY"])
            client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "ping"}],
                max_tokens=1,
                stream=False
            )
            print("OpenAI API connection warmed up.")
        except Exception as e:
            print("OpenAI API connection warm-up failed:", e)


def update_ui(token: str):
    """
    Immediately update the UI with the token.
    This version checks for newline characters and prints them so that
    line breaks and paragraphs are preserved.
    """
    # Replace Windows-style newlines with Unix-style
    token = token.replace('\r\n', '\n')
    # If the token contains newline(s), split and print accordingly.
    if '\n' in token:
        parts = token.split('\n')
        for i, part in enumerate(parts):
            print(part, end='', flush=True)
            if i < len(parts) - 1:
                print()
    else:
        print(token, end='', flush=True)


def build_llm_payload(user_input, chat_history, config):
    """
    Build the conversation messages and payload from the user input,
    chat history, and configuration.

    Returns:
        dict: The payload containing the messages and generation parameters.
    """
    system_message = config.get(
        "system_message",
        "You are Mai, speak naturally and like a human might with humour and dryness."
    )
    messages = [{"role": "system", "content": system_message}]
    for entry in chat_history:
        messages.append({"role": "user", "content": entry["input"]})
        messages.append({"role": "assistant", "content": entry["response"]})
    messages.append({"role": "user", "content": user_input})
    
    payload = {
        "messages": messages,
        "max_new_tokens": 4000,
        "temperature": 1,
        "top_p": 0.9
    }
    return payload


def local_llm_streaming(user_input, chat_history, chunk_queue, config):
    """
    Streams tokens from a local LLM using streaming.
    """
    payload = build_llm_payload(user_input, chat_history, config)
    full_response = ""
    max_chunk_length = config.get("max_chunk_length", 500)
    flush_token_count = config.get("flush_token_count", 10)
    
    # Create the SentenceBuilder and a dedicated token_queue.
    sentence_builder = SentenceBuilder(chunk_queue, max_chunk_length, flush_token_count)
    token_queue = Queue()  
    sb_thread = Thread(target=sentence_builder.run, args=(token_queue,))
    sb_thread.start()
    
    try:
        session = requests.Session()
        with session.post(config["LLM_STREAM_URL"], json=payload, stream=True) as response:
            response.raise_for_status()
            print("\n\nAssistant Response (streaming - local):\n", flush=True)
            for token in response.iter_content(chunk_size=1, decode_unicode=True):
                if not token:
                    continue
                full_response += token
                update_ui(token)
                token_queue.put(token)
        session.close()
        
        token_queue.put(None)
        sb_thread.join()
        return full_response.strip()
    
    except Exception as e:
        print(f"\nError during streaming local LLM call: {e}")
        return "Error: Streaming LLM call failed."

def local_llm_non_streaming(user_input, chat_history, chunk_queue, config):
    """
    Calls a local LLM non-streaming endpoint and processes the entire response.
    """
    payload = build_llm_payload(user_input, chat_history, config)
    full_response = ""
    max_chunk_length = config.get("max_chunk_length", 500)
    flush_token_count = config.get("flush_token_count", 10)
    
    # Set up the SentenceBuilder.
    sentence_builder = SentenceBuilder(chunk_queue, max_chunk_length, flush_token_count)
    token_queue = Queue()
    sb_thread = Thread(target=sentence_builder.run, args=(token_queue,))
    sb_thread.start()
    
    try:
        session = requests.Session()
        response = session.post(config["LLM_API_URL"], json=payload)
        session.close()
        if response.ok:
            result = response.json()
            text = result.get('assistant', {}).get('content', "Error: No response.")
            print("Assistant Response (non-streaming - local):\n", flush=True)
            tokens = text.split(' ')
            for token in tokens:
                token_with_space = token + " "
                full_response += token_with_space
                update_ui(token_with_space)
                token_queue.put(token_with_space)
            
            token_queue.put(None)
            sb_thread.join()
            return full_response.strip()
        else:
            print(f"LLM call failed: HTTP {response.status_code}")
            return "Error: LLM call failed."
    
    except Exception as e:
        print(f"Error calling local LLM: {e}")
        return "Error: Exception occurred."
class LLMChat:
    def __init__(self):
        self.history = []
    def chat_streaming(self,prompt, model, api_url, api_key, system_prompt, temperature=0.7, use_meta_prompt=False, seed=42, stop_generate=False):

        self.history.append({'role': 'user', 'content': prompt})
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(self.history)
        
        client = OpenAI(api_key=api_key, base_url=api_url)
        completion = client.chat.completions.create(
            model=model,
            seed=seed,
            temperature=temperature,
            messages=messages,
            stream=True
        )

        full_content = ""

        for event in completion:
            content = event.choices[0].delta.content
#            print(content)
            if content and stop_generate==False:
                full_content += content
                yield content
            else:
                yield "llm finished"
                break
        self.history.append({'role': 'assistant', 'content': full_content})
        
    def chat(self, prompt, model, api_url, api_key, system_prompt, temperature=0.7, use_meta_prompt=False, seed=42):
        META_PROMPT = "You are a helpful assistant." # Default meta prompt if not defined
        self.history.append({'role': 'user', 'content': prompt})
        messages = [{'role': 'system', 'content': system_prompt if not use_meta_prompt else META_PROMPT}]
        messages.extend(self.history)
        
        client = OpenAI(api_key=api_key, base_url=api_url)
        completion = client.chat.completions.create(
            model=model,
            seed=seed,
            temperature=temperature,
            messages=messages,
            stream=True
        )
        
        response = completion.choices[0].message.content
        self.history.append({'role': 'assistant', 'content': response})
        return response
    

def openai_llm_streaming(user_input, chat_history, chunk_queue, config):
    """
    Streams tokens from the OpenAI API.
    """
    payload = build_llm_payload(user_input, chat_history, config)
    full_response = ""
    max_chunk_length = config.get("max_chunk_length", 500)
    flush_token_count = config.get("flush_token_count", 10)
    
    # Set up the SentenceBuilder.
    sentence_builder = SentenceBuilder(chunk_queue, max_chunk_length, flush_token_count)
    token_queue = Queue()
    sb_thread = Thread(target=sentence_builder.run, args=(token_queue,))
    sb_thread.start()
    
    try:
        client = OpenAI(api_key=config["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=payload["messages"],
            max_tokens=4000,
            temperature=1,
            top_p=0.9,
            stream=True
        )
        print("Assistant Response (streaming - OpenAI):\n", flush=True)
        for chunk in response:
            token = chunk.choices[0].delta.content if chunk.choices[0].delta else ""
            if not token:
                continue
            full_response += token
            update_ui(token)
            token_queue.put(token)
        
        token_queue.put(None)
        sb_thread.join()
        return full_response.strip()
    
    except Exception as e:
        print(f"Error calling OpenAI API (streaming): {e}")
        return "Error: OpenAI API call failed."



def openai_llm_non_streaming(user_input, chat_history, chunk_queue, config):
    """
    Calls the OpenAI API without streaming.
    """
    payload = build_llm_payload(user_input, chat_history, config)
    full_response = ""
    max_chunk_length = config.get("max_chunk_length", 500)
    flush_token_count = config.get("flush_token_count", 10)
    
    # Set up the SentenceBuilder.
    sentence_builder = SentenceBuilder(chunk_queue, max_chunk_length, flush_token_count)
    token_queue = Queue()
    sb_thread = Thread(target=sentence_builder.run, args=(token_queue,))
    sb_thread.start()
    
    try:
        client = OpenAI(api_key=config["OPENAI_API_KEY"])
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=payload["messages"],
            max_tokens=4000,
            temperature=1,
            top_p=0.9
        )
        text = response.choices[0].message.content
        
        print("Assistant Response (non-streaming - OpenAI):\n", flush=True)
        tokens = text.split(' ')
        for token in tokens:
            token_with_space = token + " "
            full_response += token_with_space
            update_ui(token_with_space)
            token_queue.put(token_with_space)
        
        token_queue.put(None)
        sb_thread.join()
        return full_response.strip()
    
    except Exception as e:
        print(f"Error calling OpenAI API (non-streaming): {e}")
        return "Error: OpenAI API call failed."


def stream_llm_chunks(user_input, chat_history, chunk_queue, config):
    """
    Dispatches the LLM call to the proper variant based on the configuration.
    """
    USE_LOCAL_LLM = config["USE_LOCAL_LLM"]
    USE_STREAMING = config["USE_STREAMING"]
    
    if USE_LOCAL_LLM:
        if USE_STREAMING:
            return local_llm_streaming(user_input, chat_history, chunk_queue, config)
        else:
            return local_llm_non_streaming(user_input, chat_history, chunk_queue, config)
    else:
        if USE_STREAMING:
            return openai_llm_streaming(user_input, chat_history, chunk_queue, config)
        else:
            return openai_llm_non_streaming(user_input, chat_history, chunk_queue, config)

# 记忆配置
# ✅ 使用相对于当前文件的路径
MEMORY_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "memory_config.json")
# 修正：dialogue_histories 在 utils/llm/ 下，所以不需要回退两层
# 再次修正：dialogue_histories 实际上是在 utils/llm/ 下面，所以应该是当前目录下的 dialogue_histories
MEMORY_DIR = os.path.join(os.path.dirname(__file__), "dialogue_histories", "memories")

def load_memory_config():
    """加载记忆配置"""
    config_path = MEMORY_CONFIG_FILE
    if not os.path.exists(config_path):
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        # 默认配置
        default_config = {
            "summary_model": "gpt-4o-mini",
            "summary_base_url": "https://api.tu-zi.com/v1",
            "summary_api_key_env": "tuzi_api_key",
            "summary_system_prompt": "你是一个专业的对话记忆助手。你的任务是阅读一段对话历史和之前的记忆摘要，然后生成一个新的、更简洁的记忆摘要。请保留关键信息，如用户的名字、喜好、重要的事件和状态。忽略琐碎的闲聊。请直接输出摘要内容，不要包含其他解释。",
            "max_history_length": 30,
            "retain_history_length": 5
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=4)
        return default_config
    
    return load_json_file(MEMORY_CONFIG_FILE)

def get_memory_file_path(chatname):
    """获取记忆文件路径"""
    memory_dir = MEMORY_DIR
    os.makedirs(memory_dir, exist_ok=True)
    return os.path.join(memory_dir, f"{chatname}_memory.json")

def load_memory(chatname):
    """加载特定聊天的记忆列表"""
    memory_path = get_memory_file_path(chatname)
    if not os.path.exists(memory_path):
        return []
    
    try:
        with open(memory_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 兼容旧格式：如果读取到的是字典且有 summary 字段，转换为列表格式
            if isinstance(data, dict) and "summary" in data:
                print("⚠️ 检测到旧版记忆格式，正在转换...")
                return [{
                    "timestamp": data.get("updated_at", get_timestamp()),
                    "summary": data["summary"],
                    "importance": 5 # 默认重要度
                }]
            # 正常情况返回列表
            if isinstance(data, list):
                return data
            return []
    except Exception as e:
        print(f"⚠️ 加载记忆失败: {e}")
        return []

def save_memory(chatname, memory_list):
    """保存记忆列表"""
    memory_path = get_memory_file_path(chatname)
    try:
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(memory_list, f, ensure_ascii=False, indent=4)
        print(f"✅ 记忆库已更新: {chatname} (当前片段数: {len(memory_list)})")
    except Exception as e:
        print(f"❌ 保存记忆失败: {e}")

def cleanup_memories(memories, max_fragments=30):
    """
    清理记忆片段，保留高价值记忆。
    基于重要度和时间衰减进行评分。
    """
    if len(memories) <= max_fragments:
        return memories
        
    print(f"🧹 记忆片段过多 ({len(memories)} > {max_fragments})，开始清理...")
    
    # 优先保留重要度高的，其次保留时间新的
    # importance: 5 > 1
    # timestamp: "2025..." > "2024..."
    sorted_memories = sorted(memories, key=lambda x: (x.get("importance", 3), x.get("timestamp", "")), reverse=True)
    retained = sorted_memories[:max_fragments]
    
    # 重新按时间排序（恢复原始顺序的相对关系，或者直接按 timestamp 排序）
    retained.sort(key=lambda x: x.get("timestamp", ""))
    
    return retained

def add_memory(chatname, summary, importance=3):
    """添加一条新记忆"""
    memories = load_memory(chatname)
    
    new_fragment = {
        "timestamp": get_timestamp(),
        "summary": summary,
        "importance": importance
    }
    
    memories.append(new_fragment)
    
    # 清理旧记忆
    config = load_memory_config()
    max_fragments = config.get("max_memory_fragments", 30)
    memories = cleanup_memories(memories, max_fragments)
    
    save_memory(chatname, memories)

def summarize_memory(chatname, history, system_prompt=None):
    """
    对过长的历史记录进行概括，生成新的记忆片段。
    """
    config = load_memory_config()
    max_history = config.get("max_history_length", 30)
    
    # 1. 只有聊天满记录比如default.txt里满20轮（一问一答为一轮）才对20轮进行总结
    # 注意：history 列表通常包含 user 和 assistant 的消息，所以 20 轮对话大约是 40 条消息
    # 这里我们使用 max_history 作为阈值（假设 max_history 是消息条数，或者我们需要将其解释为轮数）
    # 如果 max_history 是轮数，那么消息数应该是 max_history * 2
    # 假设 max_history_length 配置的是消息条数（通常做法），如果配置的是 20，那就是 10 轮。
    # 如果用户说“20轮”，那我们假设阈值是 40 条消息。
    # 为了保持最小改动，我们沿用 max_history 变量，但逻辑上改为“只有超过阈值才总结”
    
    if len(history) < max_history:
        print(f"⚠️ 历史记录未达到阈值 ({len(history)}/{max_history})，跳过概括。")
        return history, False
        
    print(f"🧹 开始概括对话历史 (当前条数: {len(history)})...")
    
    # 提取需要概括的消息
    preserve_count = config.get("retain_history_length", 5)
    if len(history) > preserve_count:
        # 2. 已经触发被总结的20条应该从default.txt里被移除
        # to_summarize 是要被总结并移除的部分
        # recent_history 是要保留的部分
        # 逻辑：总结 to_summarize -> 生成记忆 -> 返回 recent_history (作为新的 history)
        to_summarize = history[:-preserve_count]
        recent_history = history[-preserve_count:]
    else:
        # 这种情况理论上不会发生，因为上面已经判断了 len(history) < max_history
        # 且通常 max_history > retain_history_length
        to_summarize = history
        recent_history = []
    
    # 将对话转换为文本格式
    conversation_text = ""
    for msg in to_summarize:
        role = "用户" if msg['role'] == 'user' else "助手"
        conversation_text += f"{role}: {msg['content']}\n"
        
    # 构建提示词：生成独立摘要并评分
    instruction = (
        "你是一个对话记忆提取助手，请根据人设将与用户的对话进行概括为一段记忆， 主语是我。\n"
        "请阅读以下对话，提取出一个独立的记忆片段， 注意不是摘抄原文是概括成一段话。\n"
        "要求：\n"
        "1. 总结对话中的关键事件、用户偏好、重要决策或状态变更。\n"
        "2. 不要过度概括，保留用户原话中的独特表达或具体名词和重要细节。\n"
        "3. 忽略琐碎的闲聊（如问候、简单确认）。\n"
        "4. 评估该记忆片段的重要性（1-5分，5分最重要）。\n"
        "5. 输出格式必须为 JSON：{\"summary\": \"摘要内容\", \"importance\": 整数分数}\n"
        "6. 如果对话内容完全是无意义的闲聊，summary 返回空字符串。\n"
    )
    
    if system_prompt:
        instruction += f"\n参考人设背景：\n'''{system_prompt}'''\n"

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"对话内容：\n{conversation_text}"}
    ]
    
    try:
        # API 调用逻辑
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        
        if not api_key:
            env_var_name = config.get("summary_api_key_env", "DEEPSEEK_API_KEY")
            api_key = os.environ.get(env_var_name)
            
        if not api_key:
            # 尝试从 utils.utils init_api_key 获取
            api_key = init_api_key(env_name="tuzi_api_key", input_api_key="")

        if not api_key:
            raise ValueError("未找到 API Key。")
        
        if "deepseek.com" in base_url and not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
            
        client = OpenAI(api_key=api_key, base_url=base_url)
        model = config.get("summary_model", "deepseek-chat")
        
        print(f"📡 调用 {model} 提取记忆片段...")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=500,
            response_format={ "type": "json_object" } # 强制 JSON 输出
        )
        
        content = response.choices[0].message.content.strip()
        # 解析 JSON
        try:
            result = json.loads(content)
            summary = result.get("summary", "")
            importance = result.get("importance", 3)
        except json.JSONDecodeError:
            # 降级处理：如果没返回 JSON，直接当做摘要
            print("⚠️ 模型未返回标准 JSON，尝试直接使用文本。")
            summary = content
            importance = 3
            
        if summary:
            print(f"📝 新增记忆片段 (重要度 {importance}): {summary[:50]}...")
            add_memory(chatname, summary, importance)
            
            # ✅ 关键修复：强制保存截断后的历史记录到文件
            # 这样无论外部调用者是否处理，文件都会被更新
            from utils.llm.llm_send import save_history
            save_history(chatname, recent_history)
            print(f"✅ 已自动更新历史记录文件 (剩余 {len(recent_history)} 条)")
            
            return recent_history, True
        else:
            print("⚠️ 生成的摘要为空 (可能是纯闲聊)。")
            # 即使没有生成摘要，也认为处理成功（因为已经看过了），返回 True 以便循环继续
            return recent_history, True
        
    except Exception as e:
        print(f"❌ 概括历史失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return history, False

def get_system_prompt_with_memory(chatname, base_system_prompt):
    """
    Get system prompt with memory injection
    
    Args:
        chatname: The chat name
        base_system_prompt: The base system prompt
    
    Returns:
        str: The system prompt with memory injected
    """
    # Try to load memory summary
    try:
        from utils.llm.chat_list_manager import get_file_name
        file_name = get_file_name(chatname)
        if not file_name:
            return base_system_prompt
            
        memory_path = os.path.join("dialogue_histories", f"{file_name}_memory.json")
        path = resolve_path(memory_path)
        
        if os.path.exists(path):
            memory_data = load_json_file(path)
            if memory_data and isinstance(memory_data, dict):
                summary = memory_data.get("summary", "")
                if summary:
                    return f"{base_system_prompt}\n\n[Previous Conversation Memory]\n{summary}"
    except Exception as e:
        print(f"⚠️ Failed to load memory for {chatname}: {e}")
        
    return base_system_prompt

def get_memory_string(chatname):
    """
    获取纯文本格式的记忆字符串，用于注入到 Prompt 中。
    """
    memories = load_memory(chatname)
    if not memories:
        return ""
        
    memory_text = ""
    for mem in memories:
        timestamp = mem.get("timestamp", "未知时间")
        summary = mem.get("summary", "")
        memory_text += f"- [{timestamp}] {summary}\n"
        
    return memory_text

