#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 load_history 修复
"""
import sys
import os
import json

root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.llm.llm_send import load_history
from utils.llm.text_utils import resolve_path

# 测试数据
test_chats = ["guitar", "xo", "default", "japanese", "career"]

print("=" * 80)
print("测试 load_history 修复")
print("=" * 80)

for chatname in test_chats:
    print(f"\n测试 chatname: {chatname}")
    print("-" * 40)
    
    try:
        path, history = load_history(chatname)
        print(f"✅ 加载成功")
        print(f"  路径: {path}")
        print(f"  历史记录条数: {len(history)}")
        print(f"  历史类型: {type(history)}")
        
        if history and len(history) > 0:
            print(f"  第一条记录: {str(history[0])[:80]}...")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
