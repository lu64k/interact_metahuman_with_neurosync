import pyaudio
from pathlib import Path
import time
import json
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
import asyncio
from queue import Queue
import queue
from threading import Thread
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _load_interact_config():
    """Load interaction parameters from root config_files/interact_config.json."""
    defaults = {
        "wait_for_continuation_sec": 3.0,
        "max_sentence_duration_sec": 5.0,
        "llm_first_emit_chars": 8,
    }

    config_path = Path(__file__).resolve().parents[2] / "config_files" / "interact_config.json"
    if not config_path.exists():
        logger.info("[配置] 未找到 %s，使用默认交互参数", config_path)
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            logger.warning("[配置] interact_config.json 不是对象，回退默认值")
            return defaults

        cfg = dict(defaults)
        if "wait_for_continuation_sec" in raw:
            cfg["wait_for_continuation_sec"] = float(raw["wait_for_continuation_sec"])
        if "max_sentence_duration_sec" in raw:
            cfg["max_sentence_duration_sec"] = float(raw["max_sentence_duration_sec"])
        if "llm_first_emit_chars" in raw:
            cfg["llm_first_emit_chars"] = int(raw["llm_first_emit_chars"])
        return cfg
    except Exception as e:
        logger.warning("[配置] 读取 interact_config.json 失败，使用默认值: %s", e)
        return defaults

class Callback_rc(RecognitionCallback):
    def __init__(self):
        self.stop_llm_generate = False
        self.stop_event = None

        self.first_sentence_time = None
        self.last_sentence_time = None
        self.last_received_end_ts = None
        self.last_committed_end_ts = None
        self.last_commit_wall_ts = None
        self._last_time_source = None
        self._timestamp_probe_logged = False
        self.final_texts = []
        self.new_sentence = None
        self.pending_paragraph = None
        self.mic = None
        self.stream = None

        interact_cfg = _load_interact_config()
        # ✅ 修改：增加容差时间，允许正常停顿
        self.SILENT_TIMEOUT = 1  # 静音超时（快速连续说话的阈值）
        self.MAX_SENTENCE_DURATION = interact_cfg["max_sentence_duration_sec"]  # 最大句子间隔（新对话判定）
        self.WAIT_FOR_CONTINUATION = interact_cfg["wait_for_continuation_sec"]  # 等待后续句子的时间（允许正常停顿）
        # 仅预留配置位，当前 LLM 切句原逻辑保持不变。
        self.LLM_FIRST_EMIT_CHARS = interact_cfg["llm_first_emit_chars"]
        logger.info(
            "[配置] 交互参数: wait_for_continuation=%.2fs, max_sentence_duration=%.2fs, llm_first_emit_chars=%s",
            self.WAIT_FOR_CONTINUATION,
            self.MAX_SENTENCE_DURATION,
            self.LLM_FIRST_EMIT_CHARS,
        )
        self.rc_queue = queue.Queue()
        self.new_input = None
        self.first_run = False
        # ✅ 新增：用于清空 TTS 队列的回调
        self.clear_tts_callback = None

        # 句子聚合器：单线程串行处理，避免多线程竞争导致漏句
        self._sentence_event_queue = queue.Queue()
        self._aggregator_thread = None
        self._aggregator_running = False
        self._event_counter = 0
        self._batch_counter = 0
        self._batch_event_ids = []
        self._batch_deadline_ts = None
        self._batch_last_end_ts = None

    def set_stop_event(self, stop_event: asyncio.Event):
        if self.stop_event is None:
            self.stop_event = asyncio.Event()
        self.stop_event = stop_event

    def stop_llm(self):
        if self.new_input:
            self.stop_llm_generate = True
            self.new_input = None
        return self.stop_llm_generate

    def on_open(self) -> None:
        logger.info('RecognitionCallback opening')
        try:
            # ✅ 如果已有旧的流，先关闭
            self._stop_aggregator()
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except:
                    pass
                self.stream = None
            
            if self.mic:
                try:
                    self.mic.terminate()
                except:
                    pass
                self.mic = None
            
            # 创建新的音频对象
            logger.info('正在初始化 PyAudio')
            self.mic = pyaudio.PyAudio()
            
            logger.info('正在打开音频流')
            self.stream = self.mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=3200  # ✅ 添加缓冲区大小
            )
            self._reset_aggregation_state()
            self._start_aggregator()
            logger.info('RecognitionCallback opened successfully')
            
        except Exception as e:
            logger.exception('RecognitionCallback on_open 失败: %s', e)
            # 清理失败的资源
            if self.stream:
                try:
                    self.stream.close()
                except:
                    pass
                self.stream = None
            if self.mic:
                try:
                    self.mic.terminate()
                except:
                    pass
                self.mic = None
            raise  # 重新抛出,让上层知道失败了

    def on_close(self) -> None:
        logger.info('RecognitionCallback closing')
        try:
            self._stop_aggregator()
            if self.stream:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                    logger.info('Stream closed')
                except Exception as e:
                    logger.warning('Stream close error: %s', e)
                finally:
                    self.stream = None
            
            if self.mic:
                try:
                    self.mic.terminate()
                    logger.info('Mic terminated')
                except Exception as e:
                    logger.warning('Mic terminate error: %s', e)
                finally:
                    self.mic = None
            
            logger.info('RecognitionCallback closed')
        except Exception as e:
            logger.exception('RecognitionCallback on_close 错误: %s', e)

    def on_complete(self) -> None:
        logger.info('RecognitionCallback completed')

    def on_error(self, message) -> None:
        logger.error('RecognitionCallback error: %s', message.message)

    def on_event(self, result: RecognitionResult, *args, **kwargs) -> None:
        self.new_input = result.get_sentence()  #用于储存实时输入的变量，为不完整句子
        
        if 'text' in self.new_input and RecognitionResult.is_sentence_end(self.new_input):
            received_wall_ts = time.time()
            sentence_begin_ts, sentence_end_ts, time_source = self._resolve_sentence_times(self.new_input, received_wall_ts)
            self.first_sentence_time = sentence_end_ts
            self.stop_llm_generate = False
            self.new_sentence = self.new_input.get("text") #只有完整句子时才会被赋值
            logger.info("收到完整句子: %s", self.new_sentence)
            self.new_input = None
            self._event_counter += 1
            event = {
                "event_id": self._event_counter,
                "text": self.new_sentence,
                "begin_ts": sentence_begin_ts,
                "end_ts": sentence_end_ts,
                "time_source": time_source,
            }
            self._sentence_event_queue.put(event)

    def _to_seconds(self, value):
        try:
            ts = float(value)
        except (TypeError, ValueError):
            return None

        # Normalize common timestamp forms (epoch ms, relative ms, epoch s, relative s).
        if ts > 1e11:
            return ts / 1000.0
        if ts > 1e9:
            return ts
        if ts > 1e6:
            return ts / 1000.0
        return ts

    def _extract_asr_ts(self, sentence_payload, kind="end"):
        if not isinstance(sentence_payload, dict):
            return None

        end_keys = [
            "end_time", "endTime", "end_ms", "endMs", "end", "sentence_end_time",
            "sentenceEndTime", "sentence_end", "sentenceEnd", "end_timestamp",
            "timestamp_end", "final_end_time", "finalEndTime",
        ]
        begin_keys = [
            "begin_time", "beginTime", "start_time", "startTime", "start_ms", "startMs",
            "begin", "start", "sentence_begin_time", "sentenceBeginTime", "sentence_start_time",
            "sentenceStartTime", "begin_timestamp", "timestamp_begin", "final_begin_time",
            "finalBeginTime",
        ]
        direct_keys = begin_keys if kind == "begin" else end_keys
        for key in direct_keys:
            if key in sentence_payload:
                ts = self._to_seconds(sentence_payload.get(key))
                if ts is not None:
                    return ts

        nested_keys = ["sentence", "result", "asr_result", "metadata", "time", "timing"]
        for key in nested_keys:
            nested = sentence_payload.get(key)
            if isinstance(nested, dict):
                ts = self._extract_asr_ts(nested, kind=kind)
                if ts is not None:
                    return ts

        words = sentence_payload.get("words")
        if isinstance(words, list) and words:
            target_word = words[0] if kind == "begin" else words[-1]
            if isinstance(target_word, dict):
                word_keys = ["begin_time", "beginTime", "start_time", "startTime", "start_ms", "startMs", "begin", "start"]
                if kind == "end":
                    word_keys = ["end_time", "endTime", "end_ms", "endMs", "end"]
                for key in word_keys:
                    if key in target_word:
                        ts = self._to_seconds(target_word.get(key))
                        if ts is not None:
                            return ts
        return None

    def _resolve_sentence_times(self, sentence_payload, fallback_wall_ts):
        asr_end_ts = self._extract_asr_ts(sentence_payload, kind="end")
        asr_begin_ts = self._extract_asr_ts(sentence_payload, kind="begin")

        if asr_end_ts is not None:
            asr_begin_ts = asr_begin_ts if asr_begin_ts is not None else asr_end_ts
            if not self._timestamp_probe_logged:
                logger.info("[时间基准] 使用 ASR 句首/句末时间戳")
                self._timestamp_probe_logged = True
            return asr_begin_ts, asr_end_ts, "asr"

        if not self._timestamp_probe_logged:
            logger.warning("[时间基准] 未找到 ASR 句首/句末时间戳，回退到本机接收时间")
            if isinstance(sentence_payload, dict):
                logger.debug("[时间基准] sentence payload keys: %s", list(sentence_payload.keys()))
            self._timestamp_probe_logged = True
        return fallback_wall_ts, fallback_wall_ts, "wall"

    def _reset_aggregation_state(self):
        self.first_sentence_time = None
        self.last_sentence_time = None
        self.last_received_end_ts = None
        self.last_committed_end_ts = None
        self.last_commit_wall_ts = None
        self._last_time_source = None
        self.final_texts.clear()
        self.pending_paragraph = None
        self._batch_event_ids = []
        self._batch_deadline_ts = None
        self._batch_last_end_ts = None

        while not self._sentence_event_queue.empty():
            try:
                self._sentence_event_queue.get_nowait()
                self._sentence_event_queue.task_done()
            except Exception:
                break

    def _start_aggregator(self):
        if self._aggregator_thread and self._aggregator_thread.is_alive():
            return
        self._aggregator_running = True
        self._aggregator_thread = Thread(target=self._aggregation_loop, daemon=True)
        self._aggregator_thread.start()
        logger.info("句子聚合线程已启动")

    def _stop_aggregator(self):
        self._aggregator_running = False
        self._sentence_event_queue.put(None)
        if self._aggregator_thread and self._aggregator_thread.is_alive():
            self._aggregator_thread.join(timeout=1)
        self._aggregator_thread = None

    def _trigger_interrupt(self, reason: str, supersede_history: bool = True):
        logger.info("[聚合] 触发打断: %s", reason)
        self.stop_llm_generate = True
        if self.clear_tts_callback:
            try:
                self.clear_tts_callback(supersede_history=supersede_history, reason=reason)
            except TypeError:
                self.clear_tts_callback()
            except Exception as e:
                logger.warning("[聚合] clear_tts_callback 调用失败: %s", e)

    def _start_new_batch(self, event_id: int, text: str, end_ts: float, supersede_history: bool = True):
        self.final_texts.clear()
        self.final_texts.append(text)
        self._batch_event_ids = [event_id]
        self._batch_last_end_ts = end_ts
        self._batch_deadline_ts = end_ts + self.WAIT_FOR_CONTINUATION
        logger.info(
            "[聚合] 新批次开始: event_id=%s, wait=%ss",
            event_id,
            self.WAIT_FOR_CONTINUATION,
        )
        self._trigger_interrupt("new_batch_start", supersede_history=supersede_history)

    def _start_reopened_batch(self, event_id: int, text: str, end_ts: float):
        """Reopen last committed paragraph for <5s follow-up merge."""
        if not self.pending_paragraph:
            self._start_new_batch(event_id, text, end_ts, supersede_history=True)
            return

        self.final_texts.clear()
        self.final_texts.append(self.pending_paragraph)
        self.final_texts.append(text)
        self._batch_event_ids = [f"reopen:{event_id}"]
        self._batch_last_end_ts = end_ts
        self._batch_deadline_ts = end_ts + self.WAIT_FOR_CONTINUATION
        logger.info(
            "[聚合] 复合拼接开启: event_id=%s, base=[%s], wait=%ss",
            event_id,
            self.pending_paragraph,
            self.WAIT_FOR_CONTINUATION,
        )
        self._trigger_interrupt("reopen_merge_short_gap", supersede_history=True)

    def _append_to_batch(self, event_id: int, text: str, end_ts: float):
        self.final_texts.append(text)
        self._batch_event_ids.append(event_id)
        self._batch_last_end_ts = end_ts
        self._batch_deadline_ts = end_ts + self.WAIT_FOR_CONTINUATION
        logger.info(
            "[聚合] 拼接到当前批次: event_id=%s, batch_size=%s, wait=%ss",
            event_id,
            len(self.final_texts),
            self.WAIT_FOR_CONTINUATION,
        )

    def _commit_batch(self, reason: str):
        if not self.final_texts:
            return

        paragraph = ''.join(self.final_texts)
        if not paragraph or paragraph.strip() == "":
            logger.warning("[聚合] 拼接后为空，跳过提交")
            self.final_texts.clear()
            self._batch_event_ids = []
            self._batch_deadline_ts = None
            self._batch_last_end_ts = None
            return

        while not self.rc_queue.empty():
            self.rc_queue.get()
            self.rc_queue.task_done()

        self.rc_queue.put(paragraph)
        self._batch_counter += 1
        self.pending_paragraph = paragraph
        self.last_committed_end_ts = self._batch_last_end_ts
        self.last_commit_wall_ts = time.time()
        self.last_sentence_time = self.last_committed_end_ts

        logger.info(
            "[聚合提交] batch_id=%s reason=%s events=%s text=[%s]",
            self._batch_counter,
            reason,
            self._batch_event_ids,
            paragraph,
        )

        self.final_texts.clear()
        self._batch_event_ids = []
        self._batch_deadline_ts = None
        self._batch_last_end_ts = None

    def _aggregation_loop(self):
        logger.info("句子聚合循环启动")
        while self._aggregator_running:
            try:
                event = self._sentence_event_queue.get(timeout=0.1)
            except queue.Empty:
                if self._batch_deadline_ts and time.time() >= self._batch_deadline_ts:
                    self._commit_batch("continuation_timeout")
                continue

            if event is None:
                self._sentence_event_queue.task_done()
                break

            event_id = event.get("event_id")
            text = event.get("text")
            begin_ts = event.get("begin_ts")
            end_ts = event.get("end_ts")
            time_source = event.get("time_source", "wall")

            if not text:
                self._sentence_event_queue.task_done()
                continue

            if self._last_time_source is None:
                self._last_time_source = time_source
            elif self._last_time_source != time_source:
                logger.warning(
                    "[时间基准] time_source 发生切换: %s -> %s，重置句间基准",
                    self._last_time_source,
                    time_source,
                )
                # 仅重置基于 sentence-end 的连续间隔，不清空已提交基准，避免拼接窗口被误断开。
                self.last_received_end_ts = None
                self._last_time_source = time_source

            gap_from_received = None
            gap_from_committed = None
            gap_from_commit_wall = None
            if self.last_received_end_ts is not None:
                gap_from_received = begin_ts - self.last_received_end_ts
            if self.last_committed_end_ts is not None:
                gap_from_committed = begin_ts - self.last_committed_end_ts
            if self.last_commit_wall_ts is not None:
                gap_from_commit_wall = time.time() - self.last_commit_wall_ts

            # 优先使用壁钟间隔（准确秒数），ASR 时间戳可能单位为毫秒，不可靠
            effective_gap = gap_from_commit_wall
            if effective_gap is None:
                effective_gap = gap_from_received
            if effective_gap is None:
                effective_gap = gap_from_committed

            self.first_run = self.last_received_end_ts is None
            self.last_received_end_ts = end_ts
            self.first_sentence_time = end_ts

            logger.info(
                "[聚合] event_id=%s source=%s effective_gap=%s gap_from_received=%s gap_from_committed=%s gap_from_commit_wall=%s text=%s",
                event_id,
                time_source,
                "N/A" if effective_gap is None else f"{effective_gap:.2f}s",
                "N/A" if gap_from_received is None else f"{gap_from_received:.2f}s",
                "N/A" if gap_from_committed is None else f"{gap_from_committed:.2f}s",
                "N/A" if gap_from_commit_wall is None else f"{gap_from_commit_wall:.2f}s",
                text,
            )

            # 新对话：先提交旧批次，再启动新批次
            if (
                self.final_texts
                and effective_gap is not None
                and effective_gap > self.MAX_SENTENCE_DURATION
            ):
                logger.info(
                    "[聚合] 间隔 %.2fs > %.2fs，先提交旧批次再开新批次",
                    effective_gap,
                    self.MAX_SENTENCE_DURATION,
                )
                self._commit_batch("new_conversation_gap")

            # >5s 的新对话不再复用上一提交段
            if (
                not self.final_texts
                and effective_gap is not None
                and effective_gap > self.MAX_SENTENCE_DURATION
            ):
                self.pending_paragraph = None

            if not self.final_texts:
                # 在 <5s 的续句场景中，允许复用上一已提交段进行回收拼接。
                if (
                    effective_gap is not None
                    and effective_gap <= self.MAX_SENTENCE_DURATION
                    and self.pending_paragraph
                ):
                    self._start_reopened_batch(event_id, text, end_ts)
                else:
                    self._start_new_batch(
                        event_id,
                        text,
                        end_ts,
                        supersede_history=False,
                    )
            else:
                self._append_to_batch(event_id, text, end_ts)

            # 按新收到句子的时间做句间间隔判定基准
            self._sentence_event_queue.task_done()

            if self._batch_deadline_ts and time.time() >= self._batch_deadline_ts:
                self._commit_batch("continuation_timeout")

        logger.info("句子聚合循环结束")


    def on_event44(self, result: RecognitionResult, *args, **kwargs) -> None:
        self.new_input = result.get_sentence()
        if 'text' in self.new_input and RecognitionResult.is_sentence_end(self.new_input):
            self.final_texts.append(self.new_input['text'])
            paragraph = ''.join(self.final_texts)
            while not self.rc_queue.empty():  # 清空队列
                self.rc_queue.get()
                self.rc_queue.task_done()    
                logger.info("已清空录音队列")
            self.stop_llm_generate = True
            self.rc_queue.put(paragraph)  # 塞进队列
            logger.info("请求已加入队列：%s", paragraph)
            self.final_texts.clear()
