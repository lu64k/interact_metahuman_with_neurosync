import asyncio
import gradio as gr
from utils.llm.llm_send import load_history
from utils.llm.chat_list_manager import get_all_chat_names
from utils.llm.chat_utils import create_new_chat


def get_chat_choices():
    chats = get_all_chat_names()
    return chats if chats else ['default']


async def auto_load_history(chatname, status, chatbot):
    while True:
        path, history = load_history(chatname)
        yield {'status': f'Auto-loaded history from {path}', 'chatbot': history}
        await asyncio.sleep(60)


def load_history_for_gradio(chatname):
    path, history = load_history(chatname)
    return f'Loaded from {path}', history


def create_new_chat_for_gradio(display_name, file_name):
    if not display_name or not file_name:
        return '❌ 请输入对话名称和文件名', gr.Dropdown.update()

    ok, message = create_new_chat(display_name, file_name)
    choices = get_chat_choices()
    if ok:
        return message, gr.Dropdown.update(choices=choices, value=display_name)
    return message, gr.Dropdown.update(choices=choices)


def build_gradio_app(asr_manager):
    with gr.Blocks() as demo:
        gr.Markdown('# 语音+LLM 聊天机器人')
        with gr.Row():
            with gr.Column(scale=1):
                record_btn = gr.Button('🎙️ 开始录音')
                stop_btn = gr.Button('⏹️ 停止录音')
            with gr.Column(scale=3):
                text_input = gr.Textbox(label='输入或录音结果')
                send_btn = gr.Button('🚀 发送')
                update_btn = gr.Button('更新对话历史')
                chat_choices = get_chat_choices()
                chats = gr.Dropdown(choices=chat_choices, label='选择对话', value=chat_choices[0] if chat_choices else 'default')

        chatbot = gr.Chatbot(type='messages', label='对话')
        status = gr.Textbox(label='unreal请求情况')

        with gr.Accordion('LLM 设置', open=False):
            with gr.Row():
                model_presets = gr.Dropdown(
                    choices=['grok-3', 'gpt-4o-mini', 'deepseek-chat', 'gpt-4o', 'custom'], 
                    label='预设模型', 
                    value='grok-3'
                )
                model_custom = gr.Textbox(
                    label='自定义模型名称', 
                    placeholder='如果左侧选择 custom，请在此输入',
                    visible=False
                )
            
            # 内部逻辑判断：如果选择 custom 则显示文本框，否则隐藏并同步选择
            model_final = gr.Textbox(value='grok-3', visible=False) # 隐藏的最终输入值

            def update_model_logic(choice):
                if choice == 'custom':
                    return gr.update(visible=True), gr.update(value="")
                else:
                    return gr.update(visible=False, value=""), gr.update(value=choice)

            model_presets.change(
                update_model_logic, 
                inputs=[model_presets], 
                outputs=[model_custom, model_final]
            )
            # 自定义输入框修改时，更新最终模型
            model_custom.input(
                lambda x: x, 
                inputs=[model_custom], 
                outputs=[model_final]
            )

            with gr.Row():
                new_chat_name = gr.Textbox(label='新对话名称')
                new_chat_filename = gr.Textbox(label='文件名')
                create_chat_btn = gr.Button('➕ 创建新对话')

        record_btn.click(
            asr_manager.record_and_recognize,
            inputs=[gr.State(False), model_final, chats],
            outputs=[text_input, chatbot, status]
        )
        stop_btn.click(
            asr_manager.record_and_recognize,
            inputs=[gr.State(True), model_final, chats],
            outputs=[text_input, chatbot, status]
        )
        create_chat_btn.click(
            create_new_chat_for_gradio,
            inputs=[new_chat_name, new_chat_filename],
            outputs=[status, chats]
        )

        chats.change(load_history_for_gradio, inputs=[chats], outputs=[status, chatbot])
        update_btn.click(load_history_for_gradio, inputs=[chats], outputs=[status, chatbot])
        demo.load(auto_load_history, inputs=[chats, status, chatbot], outputs=[status, chatbot], queue=True)

    return demo
