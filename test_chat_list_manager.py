"""
Test script for chat_list_manager module
验证动态加载和更新chat_list.json的功能
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.llm.chat_list_manager import (
    get_chat_list,
    get_all_chat_names,
    get_file_name,
    chat_exists,
    add_chat,
    remove_chat,
    get_all_chats
)

def test_chat_list_manager():
    """Test all chat_list_manager functions"""
    
    print("\n" + "="*60)
    print("Testing chat_list_manager module")
    print("="*60)
    
    # Test 1: Get all chats
    print("\n✅ Test 1: Get all chats")
    all_chats = get_all_chats()
    print(f"   Found {len(all_chats)} chats:")
    for name, filename in all_chats.items():
        print(f"   - {name} -> {filename}")
    
    # Test 2: Get all chat names
    print("\n✅ Test 2: Get all chat names")
    names = get_all_chat_names()
    print(f"   Chat names: {names}")
    
    # Test 3: Check if specific chat exists
    print("\n✅ Test 3: Check if chats exist")
    for test_name in ["default", "guitar", "nonexistent"]:
        exists = chat_exists(test_name)
        print(f"   Chat '{test_name}' exists: {exists}")
    
    # Test 4: Get file name for a chat
    print("\n✅ Test 4: Get file names")
    for test_name in ["default", "guitar"]:
        filename = get_file_name(test_name)
        print(f"   Chat '{test_name}' -> file '{filename}'")
    
    # Test 5: Add a new chat (if not exists)
    print("\n✅ Test 5: Add new chat")
    test_chat_name = "_test_chat_12345"
    if chat_exists(test_chat_name):
        print(f"   Chat '{test_chat_name}' already exists, removing first...")
        remove_chat(test_chat_name)
    
    success = add_chat(test_chat_name, "_test_chat_file")
    if success:
        print(f"   ✅ Successfully added chat '{test_chat_name}'")
        # Verify it was added
        if chat_exists(test_chat_name):
            print(f"   ✅ Verified chat '{test_chat_name}' exists in chat list")
    else:
        print(f"   ❌ Failed to add chat")
    
    # Test 6: Remove the test chat
    print("\n✅ Test 6: Remove test chat")
    if chat_exists(test_chat_name):
        success = remove_chat(test_chat_name)
        if success:
            print(f"   ✅ Successfully removed chat '{test_chat_name}'")
            if not chat_exists(test_chat_name):
                print(f"   ✅ Verified chat '{test_chat_name}' is no longer in chat list")
        else:
            print(f"   ❌ Failed to remove chat")
    
    # Test 7: Verify load from file
    print("\n✅ Test 7: Verify chat list is loaded from file")
    chat_list = get_chat_list()
    print(f"   Loaded {len(chat_list)} chats from chat_list.json")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        test_chat_list_manager()
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
