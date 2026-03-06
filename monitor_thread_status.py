"""
🔍 实时线程状态监控工具

用于调试队列线程问题，实时显示线程状态
"""

import sys
import time
import threading
import requests

BASE_URL = "http://localhost:5000"

def monitor_threads():
    """监控所有活动线程"""
    print("\n" + "=" * 80)
    print("🔍 当前活动线程：")
    print("=" * 80)
    
    for thread in threading.enumerate():
        print(f"  📌 {thread.name} - 存活: {thread.is_alive()} - Daemon: {thread.daemon}")
    
    print("=" * 80)

def test_with_monitoring():
    """带监控的测试"""
    print("🚀 开始监控测试...")
    print("⚠️ 请确保服务器已启动并开始录音\n")
    
    # 初始线程状态
    print("📊 初始状态：")
    monitor_threads()
    
    input("\n按回车发送第一次请求...")
    
    # 第一次请求
    print("\n📝 发送第一次请求：'这是第一段测试'")
    try:
        requests.post(
            f"{BASE_URL}/api/recognition_complete",
            json={"text": "这是第一段测试"},
            timeout=1
        )
    except:
        pass
    
    time.sleep(1)
    print("\n📊 第一次请求后线程状态：")
    monitor_threads()
    
    input("\n按回车发送第二次请求（打断）...")
    
    # 第二次请求（打断）
    print("\n📝 发送第二次请求（打断）：'这是第二段测试'")
    try:
        requests.post(
            f"{BASE_URL}/api/recognition_complete",
            json={"text": "这是第二段测试"},
            timeout=1
        )
    except:
        pass
    
    time.sleep(1)
    print("\n📊 第二次请求（打断）后线程状态：")
    monitor_threads()
    
    input("\n按回车发送第三次请求（再次打断）...")
    
    # 第三次请求（再次打断）
    print("\n📝 发送第三次请求（再次打断）：'这是第三段测试'")
    try:
        requests.post(
            f"{BASE_URL}/api/recognition_complete",
            json={"text": "这是第三段测试"},
            timeout=1
        )
    except:
        pass
    
    time.sleep(1)
    print("\n📊 第三次请求（再次打断）后线程状态：")
    monitor_threads()
    
    print("\n" + "=" * 80)
    print("✅ 监控测试完成！")
    print("=" * 80)
    print("\n💡 提示：")
    print("  - 检查是否有 'process_audio_queue' 或类似名称的线程")
    print("  - 确认线程在多次打断后仍然存活")
    print("  - 如果线程消失，说明线程意外退出")

def continuous_monitor():
    """持续监控线程状态"""
    print("🔄 启动持续监控（每 2 秒刷新一次，Ctrl+C 停止）...\n")
    
    try:
        while True:
            # 清屏（Windows）
            import os
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print("🔍 实时线程监控")
            print(f"⏰ 时间: {time.strftime('%H:%M:%S')}")
            monitor_threads()
            
            print("\n💡 关键线程检查：")
            thread_names = [t.name for t in threading.enumerate()]
            
            if any('process_audio_queue' in name.lower() for name in thread_names):
                print("  ✅ 队列处理线程：运行中")
            else:
                print("  ❌ 队列处理线程：未找到")
            
            if any('interrupt_monitor' in name.lower() for name in thread_names):
                print("  ✅ 打断监控线程：运行中")
            else:
                print("  ❌ 打断监控线程：未找到")
            
            print("\n按 Ctrl+C 停止监控...")
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\n✅ 监控已停止")

if __name__ == "__main__":
    print("🔧 线程状态监控工具")
    print("=" * 80)
    print("选择模式：")
    print("  1. 交互式测试（手动控制每一步）")
    print("  2. 持续监控（实时显示线程状态）")
    print("=" * 80)
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        test_with_monitoring()
    elif choice == "2":
        continuous_monitor()
    else:
        print("❌ 无效选择")
