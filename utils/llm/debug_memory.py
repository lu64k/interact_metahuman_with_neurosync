
import os
import sys
import json
import time

# Add root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(root_dir)

from utils.llm.llm_utils import summarize_memory, load_memory, get_memory_file_path, load_memory_config
from utils.llm.llm_send import load_history
from utils.llm.text_utils import load_text

def debug_process_guitar():
    chatname = "guitar"
    print(f"Debugging memory process for: {chatname}")
    
    # 1. Check history
    try:
        path, history = load_history(chatname)
        print(f"Loaded history from {path}, count: {len(history)}")
    except Exception as e:
        print(f"Failed to load history: {e}")
        return

    # 2. Load system prompt
    try:
        sys_prompt_path = os.path.join(root_dir, "config_files", "sys_prompt.txt")
        system_prompt = load_text(sys_prompt_path)
        print("System prompt loaded.")
    except:
        system_prompt = "You are a helpful assistant."

    # 3. Check existing memory
    mem_path = get_memory_file_path(chatname)
    print(f"Memory file path: {mem_path}")
    memories = load_memory(chatname)
    print(f"Existing memories: {len(memories)}")

    # 4. Try to summarize if history is long enough
    config = load_memory_config()
    max_history = config.get("max_history_length", 20)
    
    if len(history) > max_history:
        print(f"History ({len(history)}) > Max ({max_history}), attempting summary...")
        chunk = history[:max_history]
        recent, success = summarize_memory(chatname, chunk, system_prompt)
        if success:
            print("Summary successful!")
        else:
            print("Summary failed or returned False.")
    else:
        print("History not long enough to trigger summary.")

    # 5. Check memory again
    memories_after = load_memory(chatname)
    print(f"Memories after: {len(memories_after)}")
    if memories_after:
        print(f"Latest memory: {memories_after[-1]}")

if __name__ == "__main__":
    debug_process_guitar()
