import threading
import uuid
from flask import Flask, request, jsonify
from utils.llm.chat_list_manager import get_all_chat_names
from utils.llm.chat_utils import create_new_chat


def create_flask_app(asr_manager, streaming_responses, streaming_lock):
    app = Flask(__name__)

    @app.route('/create_chat', methods=['POST'])
    def create_chat_api():
        try:
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'message': 'No JSON data provided',
                    'chat_list': get_all_chat_names()
                }), 400

            display_name = data.get('display_name', '').strip()
            file_name = data.get('file_name', '').strip()

            ok, message = create_new_chat(display_name, file_name)
            return jsonify({
                'success': ok,
                'message': message,
                'chat_list': get_all_chat_names()
            }), 200 if ok else 400
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error: {str(e)}',
                'chat_list': get_all_chat_names()
            }), 500

    @app.route('/get_chats', methods=['GET'])
    def get_chats_api():
        try:
            chats = get_all_chat_names()
            return jsonify({'success': True, 'chats': chats}), 200
        except Exception as e:
            return jsonify({'success': False, 'message': str(e), 'chats': []}), 500

    @app.route('/unreal_mh', methods=['POST'])
    def unreal_mh_api():
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'message': 'No JSON data'}), 400

            on_recording = data.get('on_recording')
            model = data.get('model') or 'gemini-3-flash-preview'
            chatname = data.get('chats') or data.get('chatname', 'default')
            req_type = data.get('type')
            session_id = data.get('session_id') or str(uuid.uuid4())

            is_start = (req_type == 'start') or (on_recording is True) or (on_recording == 1)
            is_stop = (req_type == 'stop') or (on_recording is False) or (on_recording == 0)
            is_poll = req_type == 'poll'
            is_update_info = on_recording == 'keep'

            if is_start:
                _, _, status = asr_manager.record_and_recognize_with_stream(False, model, chatname, session_id)
                return jsonify({'success': True, 'message': status, 'session_id': session_id})

            if is_stop:
                _, _, status = asr_manager.record_and_recognize_with_stream(True, '', '', session_id)
                return jsonify({'success': True, 'message': status, 'session_id': session_id})

            if is_update_info:
                with asr_manager.config_lock:
                    asr_manager.current_config['model'] = model
                    asr_manager.current_config['chats'] = chatname
                return jsonify({
                    'success': True,
                    'message': f'Config updated: model={model}, chats={chatname}',
                    'session_id': session_id
                })

            if is_poll:
                with streaming_lock:
                    if session_id in streaming_responses:
                        resp_data = streaming_responses[session_id]
                        sentences = resp_data['sentences']
                        last_index = resp_data['last_index']
                        new_sentences = sentences[last_index:]
                        resp_data['last_index'] = len(sentences)
                        return jsonify({'success': True, 'sentences': new_sentences, 'session_id': session_id})
                    return jsonify({'success': True, 'sentences': [], 'session_id': session_id, 'message': 'No session found'})

            return jsonify({'success': False, 'message': 'Invalid request parameters'}), 400
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

    return app


def run_flask(app, host='0.0.0.0', port=8999):
    app.run(host=host, port=port, debug=True, use_reloader=False)


def start_flask_thread(app, host='0.0.0.0', port=8999):
    thread = threading.Thread(target=run_flask, args=(app, host, port), daemon=True)
    thread.start()
    return thread
