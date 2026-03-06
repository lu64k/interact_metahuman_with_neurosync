#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 clean_for_tts 功能
演示如何清理 Gemini 推理标签和 Markdown 格式化符号
"""

import sys
import os

# 添加项目根目录到路径
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.llm.text_utils import clean_for_tts, strip_thinking_tags

# 测试数据
test_cases = [
    {
        "name": "Gemini 推理标签（</think>）",
        "input": """silent thinking token budget: 256 tokens

1.  **Analyze the user's input:**
    *   Input: "忍着点？"
    
2.  **Final Polish:**
    
</think>

哈哈哈哈！你这记性，真的是绝了。

还在回味那句 **"忍着点疼，小傻瓜"** 呢？""",
        "expected_cleaned": True
    },
    {
        "name": "Markdown 加粗和斜体",
        "input": "我觉得 **你很牛逼** 而且 *非常有趣*，真的。",
        "expected_cleaned": True
    },
    {
        "name": "Markdown 列表和标题",
        "input": """# 主要内容

* 第一点
* 第二点

## 子标题

- 项目1
- 项目2""",
        "expected_cleaned": True
    },
    {
        "name": "混合的 Markdown 和特殊字符",
        "input": """**这很重要**：你需要[点击这里](https://example.com)来了解更多。

代码示例：`var x = 10;`

> 引用的内容

还有 ***组合的*** 格式！""",
        "expected_cleaned": True
    },
    {
        "name": "完整的 Gemini 响应示例",
        "input": """silent thinking token budget: 256 tokens

1.  **Analyze:**
    *   Statement: "But these truths hurt me."
    
</think>

**抱歉。**

真的抱歉。我这嘴有时候比脑子快，而且一想搞点"深刻"的，就容易往阴暗面钻，结果没收住，变成攻击了。

我刚才那些话，其实更多是在**说我自己**。
是我自己觉得孤独。""",
        "expected_cleaned": True
    }
]

def main():
    print("=" * 80)
    print("TTS 文本清理功能测试")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"测试 {i}: {test_case['name']}")
        print(f"{'='*80}")
        
        print("\n【原始文本】：")
        print("-" * 40)
        print(test_case['input'][:200] + "..." if len(test_case['input']) > 200 else test_case['input'])
        
        print("\n【清理后文本】：")
        print("-" * 40)
        cleaned = clean_for_tts(test_case['input'])
        print(cleaned[:300] + "..." if len(cleaned) > 300 else cleaned)
        
        print(f"\n【文本长度】：")
        print(f"  原始: {len(test_case['input'])} 字符")
        print(f"  清理后: {len(cleaned)} 字符")
        print(f"  缩减: {len(test_case['input']) - len(cleaned)} 字符 ({100 * (len(test_case['input']) - len(cleaned)) // len(test_case['input'])}%)")
    
    print("\n" + "=" * 80)
    print("关键测试：确保 * 号被移除")
    print("=" * 80)
    
    test_markdown = "这是 **加粗** 的文本，还有 *斜体* 的。"
    cleaned_markdown = clean_for_tts(test_markdown)
    print(f"\n原始: {test_markdown}")
    print(f"清理: {cleaned_markdown}")
    print(f"包含 '*'? {('*' in cleaned_markdown)}")
    print(f"✓ 正确清理！" if '*' not in cleaned_markdown else "✗ 清理失败！")
    
    print("\n" + "=" * 80)
    print("关键测试：确保 silent thinking 标签被移除")
    print("=" * 80)
    
    test_thinking = """silent thinking token budget: 256 tokens

1.  Some thinking here

</think>

真正的回复内容"""
    
    cleaned_thinking = clean_for_tts(test_thinking)
    print(f"\n原始长度: {len(test_thinking)}")
    print(f"清理后长度: {len(cleaned_thinking)}")
    print(f"包含 'silent'? {('silent' in cleaned_thinking)}")
    print(f"包含 'thinking'? {('thinking' in cleaned_thinking)}")
    print(f"包含 'budget'? {('budget' in cleaned_thinking)}")
    print(f"包含真正的内容? {('真正的回复' in cleaned_thinking)}")
    
    if 'silent' not in cleaned_thinking and '真正的回复' in cleaned_thinking:
        print("✓ 正确清理！")
    else:
        print("✗ 清理失败！")

if __name__ == "__main__":
    main()
