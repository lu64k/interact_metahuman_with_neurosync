"""
Chat List Manager Module

This module provides a single source of truth for chat mappings.
All operations are dynamically loaded from chat_list.json at runtime.
Thread-safe operations for concurrent access.
"""

import os
import json
import threading
from utils.llm.text_utils import resolve_path, load_json_file

# Thread lock for safe concurrent access
_chat_list_lock = threading.RLock()

# Cache to avoid frequent file reads
_chat_list_cache = None
_cache_valid = False


def _get_chat_list_path():
    """
    Get the absolute path to chat_list.json
    
    Returns:
        str: Absolute path to chat_list.json
    """
    # Resolve relative to the dialogue_histories directory
    chat_list_path = os.path.join("dialogue_histories", "chat_list.json")
    return resolve_path(chat_list_path)


def _load_chat_list_from_file():
    """
    Load chat_list.json from disk
    
    Returns:
        dict: Chat list mapping (name -> file_name)
    """
    chat_list_path = _get_chat_list_path()
    
    try:
        if not os.path.exists(chat_list_path):
            print(f"⚠️ chat_list.json not found at {chat_list_path}, returning empty dict")
            return {}
        
        chat_list = load_json_file(chat_list_path)
        if not isinstance(chat_list, dict):
            print(f"⚠️ chat_list.json is not a dict, returning empty dict")
            return {}
        
        return chat_list
    
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse chat_list.json: {e}, returning empty dict")
        return {}
    except Exception as e:
        print(f"❌ Failed to load chat_list.json: {e}, returning empty dict")
        return {}


def get_chat_list():
    """
    Get the current chat list with caching
    
    Returns:
        dict: Chat list mapping (name -> file_name)
    """
    global _chat_list_cache, _cache_valid
    
    with _chat_list_lock:
        # For each call, reload from file to ensure freshness
        # (especially when new chats are created from frontend)
        _chat_list_cache = _load_chat_list_from_file()
        _cache_valid = True
        return _chat_list_cache.copy()


def invalidate_cache():
    """
    Invalidate the cache (call when external changes happen)
    """
    global _cache_valid
    with _chat_list_lock:
        _cache_valid = False


def add_chat(display_name, file_name):
    """
    Add a new chat to chat_list.json
    
    Args:
        display_name: The display name (shown in UI)
        file_name: The file name (used for dialogue history)
    
    Returns:
        bool: True if successful, False otherwise
    """
    with _chat_list_lock:
        try:
            # Load current chat_list
            chat_list = _load_chat_list_from_file()
            
            # Check if chat already exists
            if display_name in chat_list:
                print(f"⚠️ Chat '{display_name}' already exists")
                return False
            
            # Add new chat
            chat_list[display_name] = file_name
            
            # Save back to file
            chat_list_path = _get_chat_list_path()
            with open(chat_list_path, 'w', encoding='utf-8') as f:
                json.dump(chat_list, f, ensure_ascii=False, indent=4)
            
            print(f"✅ Added new chat: '{display_name}' -> '{file_name}'")
            invalidate_cache()
            return True
        
        except Exception as e:
            print(f"❌ Failed to add chat: {e}")
            return False


def remove_chat(display_name):
    """
    Remove a chat from chat_list.json
    
    Args:
        display_name: The display name to remove
    
    Returns:
        bool: True if successful, False otherwise
    """
    with _chat_list_lock:
        try:
            # Load current chat_list
            chat_list = _load_chat_list_from_file()
            
            # Check if chat exists
            if display_name not in chat_list:
                print(f"⚠️ Chat '{display_name}' not found")
                return False
            
            # Remove chat
            del chat_list[display_name]
            
            # Save back to file
            chat_list_path = _get_chat_list_path()
            with open(chat_list_path, 'w', encoding='utf-8') as f:
                json.dump(chat_list, f, ensure_ascii=False, indent=4)
            
            print(f"✅ Removed chat: '{display_name}'")
            invalidate_cache()
            return True
        
        except Exception as e:
            print(f"❌ Failed to remove chat: {e}")
            return False


def get_file_name(display_name):
    """
    Get the file name for a given display name
    
    Args:
        display_name: The display name to look up
    
    Returns:
        str: The file name, or None if not found
    """
    chat_list = get_chat_list()
    return chat_list.get(display_name)


def chat_exists(display_name):
    """
    Check if a chat exists
    
    Args:
        display_name: The display name to check
    
    Returns:
        bool: True if chat exists, False otherwise
    """
    chat_list = get_chat_list()
    return display_name in chat_list


def get_all_chat_names():
    """
    Get list of all chat display names
    
    Returns:
        list: List of chat display names
    """
    chat_list = get_chat_list()
    return list(chat_list.keys())


def get_all_chats():
    """
    Get the complete chat list
    
    Returns:
        dict: Chat list mapping (name -> file_name)
    """
    return get_chat_list()
