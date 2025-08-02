from datetime import datetime
import dashscope
import os

def init_api_key(env_name = 'DASHSCOPE_API_KEY', input_api_key = "输入你的api"):
    if env_name in os.environ:
        api_key = os.environ[env_name]        
    else:
        api_key = input_api_key   
    return api_key
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

