# TTS 文本清理和历史记录加载修复总结

## 问题描述

### 1. JSON 解析错误
```
json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```
**原因**：`load_history` 函数在读取空的或无效的 JSON 文件时，没有进行异常处理。

### 2. 拆包错误
```
Start queue audio error: not enough values to unpack (expected 2, got 0)
```
**原因**：`load_history` 返回空值或返回值格式不正确。

## 解决方案

### 1. 增强 `load_history` 函数（`utils/llm/llm_send.py`）

```python
def load_history(chatname):
    file_name = chat_list.get(chatname)
    if not file_name:
        print(f"⚠️ 未找到聊天名称: {chatname}")
        return "", []  # ✅ 返回 (路径, 空列表)
    
    chat_path = os.path.join("dialogue_histories", f"{file_name}.txt")
    path = resolve_path(chat_path)
    
    try:
        history_text = load_text(r"{}".format(path))
        
        # 处理空文件或空字符串
        if not history_text or history_text.strip() == "":
            print(f"⚠️ 历史文件为空或不存在: {path}")
            return path, []  # ✅ 返回 (路径, 空列表)
        
        # 解析 JSON 字符串
        history = json.loads(history_text)
        
        # 确保返回的是列表
        if not isinstance(history, list):
            print(f"⚠️ 历史记录不是列表格式，重置为空列表")
            return path, []  # ✅ 返回 (路径, 空列表)
        
        return path, history
        
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 解析失败: {e}，初始化为空列表")
        return path, []  # ✅ 返回 (路径, 空列表)
    except Exception as e:
        print(f"❌ 加载历史记录失败: {e}，初始化为空列表")
        return path, []  # ✅ 返回 (路径, 空列表)
```

**关键改进**：
- ✅ 总是返回 2 元组 `(path, history)`，避免拆包错误
- ✅ 处理空文件的情况
- ✅ 处理 JSON 解析异常
- ✅ 处理格式不是列表的情况
- ✅ 提供详细的调试日志

### 2. 增强 TTS 文本清理（`utils/llm/text_utils.py`）

新增两个函数：

#### `strip_thinking_tags(text)`
```python
def strip_thinking_tags(text: str) -> str:
    """
    移除 Google Gemini 推理标签及其内容（</thought> 或 </think>）。
    保留标签之后的实际回复文本。
    """
    if text is None:
        return None
    
    match = re.search(r'(?:</thought>|</think>)([\s\S]*)', text)
    if match:
        return match.group(1).strip()
    else:
        return text
```

#### `clean_for_tts(text)`
```python
def clean_for_tts(text: str) -> str:
    """
    清理文本以供 TTS 使用。
    移除：
    1. 推理标签（Google Gemini）
    2. Markdown 格式化符号（*, **, #, 等）
    3. 链接格式 [text](url)
    4. 多余空白字符
    5. 特殊符号
    """
    # Step 1: 移除推理标签
    text = strip_thinking_tags(text)
    
    # Step 2-9: 移除各种 Markdown 标记
    # - 加粗 (**text**, __text__)
    # - 斜体 (*text*, _text_)
    # - 标题 (#, ##, ###)
    # - 列表 (*, -, +)
    # - 链接 ([text](url))
    # - 代码块 (```)
    # - 行内代码 (`code`)
    # - 引用 (>)
    
    # Step 10: 移除特殊符号但保留中文标点
    # 保留：中文标点、英文标点、数字、字母、中日文字符
    
    # Step 11: 清理多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

### 3. 更新 `process_for_tts` 函数（`utils/llm/llm_send.py`）

```python
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
```

## 测试方法

### 测试 `load_history` 修复
```bash
python test_load_history.py
```

### 测试 `clean_for_tts` 功能
```bash
python test_clean_for_tts.py
```

## 文件变更

1. **utils/llm/llm_send.py**
   - ✅ 修复 `load_history` 函数，增加异常处理和边界情况处理
   - ✅ 更新 `process_for_tts` 函数，使用新的 `clean_for_tts` 函数

2. **utils/llm/text_utils.py**
   - ✅ 新增 `strip_thinking_tags` 函数
   - ✅ 新增 `clean_for_tts` 函数
   - 保留原有的 `clean_text` 函数（向后兼容）

## 预期效果

### 对话流程
1. ✅ 无论历史文件是否存在或为空，`load_history` 都能正确返回
2. ✅ TTS 文本会自动移除 Markdown 格式和推理标签
3. ✅ 用户听到的是清晰的、可读的文本，不会包含 `*`, `**`, `#` 等符号

### 错误处理
- ✅ 空历史文件不再崩溃
- ✅ 无效 JSON 自动重置为空列表
- ✅ 所有异常都有日志输出用于调试

## 注意事项

1. **性能**：`clean_for_tts` 包含多个正则替换操作，但由于通常只处理几百字符的句子，性能影响可忽略

2. **编码**：文本处理使用 UTF-8，支持中日文字符和各种标点符号

3. **兼容性**：如果需要保留特定的符号格式，可修改 `clean_for_tts` 中的正则表达式

## 使用示例

```python
# 清理包含推理标签和 Markdown 的文本
from utils.llm.text_utils import clean_for_tts

raw_text = """silent thinking...

</think>

这是 **重要** 的 *信息*"""

cleaned = clean_for_tts(raw_text)
# 输出: "这是重要的信息"
```
