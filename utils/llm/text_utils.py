# This software is licensed under a **dual-license model**
# For individuals and businesses earning **under $1M per year**, this software is licensed under the **MIT License**
# Businesses or organizations with **annual revenue of $1,000,000 or more** must obtain permission to use this software commercially.
import requests
from threading import Thread
from queue import Queue
from openai import OpenAI
import os
import chardet
import re
import json

def resolve_path(rel_path: str) -> str:
    """兼容脚本/Notebook 的相对路径解析，全程用原始字符串防乱。"""
    if os.path.isabs(rel_path):
        return rel_path
    try:
        base_dir = os.path.dirname(os.path.abspath(r"{}".format(__file__)))
    except NameError:
        base_dir = r"{}".format(os.getcwd())
    f_path = os.path.join(r"{}".format(base_dir), r"{}".format(rel_path))
    return f_path

def load_json_file(file_path: str):
    """
    1) 解析相对路径
    2) 读取二进制，优先按 UTF-8 解码；失败后用 chardet 探测
    3) 返回 Python 对象
    """
    # 解析相对路径（复用你已有的 resolve_path）
    path = resolve_path(file_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        open(path, "w", encoding="utf-8").write("{}")

    raw = open(path, "rb").read()
    text = None

    # 优先尝试 UTF-8（含 BOM）
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # 回落：让 chardet 帮忙
        detect = chardet.detect(raw)
        enc = detect.get("encoding") or "utf-8"
        text = raw.decode(enc, errors="replace")
        print(f"[WARN] 用 {enc} 解码 JSON（置信度 {detect.get('confidence',0):.2f}）")

    return json.loads(text)


def load_text(file_path: str) -> str:
    """读任意编码的文本文件，若不存在则创建空文件。"""
    path = resolve_path(file_path)
    print(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        open(path, "w", encoding="utf-8").close()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_text(text: str) -> str:   
    # 2. 去除多余的空格
    text = re.sub(r'\s+', ' ', text)  # 将多个空格替换为一个空格
    
    # 3. 去除文本两端的空白字符
    text = text.strip()  # 去掉首尾的空格或换行符
    
    # 4. 替换换行符或回车符为空格（如果需要）
    text = re.sub(r'[\r\n]+', ' ', text)
    
    # 5. 可选：移除一些可能的特殊符号（例如：非标准标点符号）
    text = re.sub(r'[^\w\s,.!?\'"-]', '', text)  # 只保留字母、数字、空格和常见标点符号
    
    return text

def strip_thinking_tags(text: str) -> str:
    """
    移除 Google Gemini 等模型的推理标签 <think>...</think>
    """
    if not text:
        return ""
    # 移除 <think>...</think> 及其内容，包括跨行
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    return text

def clean_for_tts(text: str) -> str:
    """
    清理文本以供 TTS 使用。
    移除：
    1. 推理标签（Google Gemini）
    2. Markdown 格式化符号（*, **, #, 等）
    3. 链接格式 [text](url)
    4. 多余空白字符
    5. 特殊符号
    
    Args:
        text: 原始文本
        
    Returns:
        TTS 友好的清理文本
    """
    if text is None:
        return ""
    
    # Step 1: 移除推理标签及其内容
    text = strip_thinking_tags(text)
    
    # Step 2: 移除 Markdown 加粗 (**text** 或 __text__)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    
    # Step 3: 移除 Markdown 斜体 (*text* 或 _text_)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Step 4: 移除 Markdown 标题 (# ## ### 等)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Step 5: 移除 Markdown 列表 (* - + 开头的行)
    text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    
    # Step 6: 移除 Markdown 链接 [text](url)，保留文本部分
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Step 7: 移除代码块标记 (``` 或 ~~~)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'~~~[\s\S]*?~~~', '', text)
    
    # Step 8: 移除行内代码 (`code`)，保留代码内容
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Step 9: 移除引用标记 (> )
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    
    # Step 10: 移除特殊符号但保留中文标点
    # 保留：中文标点（，。！？；：""''）、英文标点（,.!?;:'"）、数字、字母、中文、日文
    # [FIX] 修复正则表达式，避免范围错误
    text = re.sub(r'[^\w\s\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff，。！？；：""''\'"\-]', '', text)
    
    # Step 11: 清理多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
