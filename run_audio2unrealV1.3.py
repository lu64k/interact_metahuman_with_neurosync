# %%
import signal
import sys
import threading
import logging

from utils.stt_llm_tts import Run_LLM_To_Anim
from utils.stt.Ali_voicer_rc import Callback_rc
from core.asr_manager import ASRManager
from api.flask_app import create_flask_app, start_flask_thread
from api.gradio_ui import build_gradio_app
from utils.logging_config import setup_logging, parse_log_level, get_logger

setup_logging(level=parse_log_level(logging.INFO))
logger = get_logger(__name__)

audio_que = Run_LLM_To_Anim()
callback_rc = Callback_rc()


def push_sentence_callback(*args, **kwargs):
    """Compatibility no-op callback for sentence streaming hooks."""
    return

asr_manager = ASRManager(callback_rc, audio_que, push_sentence_callback)
callback_rc.clear_tts_callback = asr_manager.clear_tts_queue

streaming_responses = {}
streaming_lock = threading.Lock()

app = create_flask_app(asr_manager, streaming_responses, streaming_lock)
demo = build_gradio_app(asr_manager)


def signal_handler(sig, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


if __name__ == '__main__':
    def global_exception_handler(exctype, value, tb):
        logger.exception("Unhandled exception captured", exc_info=(exctype, value, tb))

    sys.excepthook = global_exception_handler
    start_flask_thread(app, host='0.0.0.0', port=8999)

    try:
        demo.launch(server_port=8989, debug=True)
    except Exception as e:
        logger.exception("Gradio launch failed: %s", e)
