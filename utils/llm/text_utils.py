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
