
Note: This is an extended fork of the original NeuroSync Player + NeuroSync Local API from https://github.com/AnimaVR. Code in this repository follows the original NeuroSync license terms.

# Interact MetaHuman with NeuroSync

[English Version](#english-version) | [中文版本](#中文版本)

---

## English Version

Real-time MetaHuman interaction system for Unreal Engine 5 (UE5), integrating ASR + LLM + TTS and driving facial animation to UE via LiveLink.

### Features
- Real-time voice conversation pipeline (speech in -> LLM -> speech out -> avatar animation)
- LiveLink-based lip sync (61-dim blendshape stream)
- Event-driven interruption during speaking turns (cooperative cancellation)
- Gradio model switching for different LLM backends

### Tech Stack
- ASR/TTS: Alibaba DashScope (Paraformer / CosyVoice)
- LLM: OpenAI-compatible API providers
- Runtime: Asyncio + threading hybrid pipeline
- Animation transport: LiveLink UDP (target 60 FPS)

### Requirements
- OS: Windows (primary tested)
- Python: 3.11 recommended
- GPU (optional but recommended): CUDA 11.8-compatible setup for PyTorch workloads
- Unreal Engine 5 with LiveLink receiver configured

### Quick Start
1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Install PyTorch matching your CUDA version (example for CUDA 11.8).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Configure credentials and runtime settings.
- API keys: `DASHSCOPE_API_KEY`, `OPENAI_API_KEY`
- Main config files: `config.py`, `config_files/interact_config.json`

4. Run the project.

```bash
python run_audio2unrealV1.3.py
```

Optional Windows launcher:

```bash
Start_Base_V1.3.bat
```

### Project Layout
- `api/`: Flask + Gradio entry points
- `core/`: ASR manager and orchestration
- `livelink/`: UE animation streaming
- `utils/`: STT/LLM/TTS/audio utility modules
- `utils_local_api/`: local inference support code

### Troubleshooting
- If no audio response is produced, verify API keys and check `logs/`.
- If UE lip sync is weak, inspect mouth scaling in `livelink/connect/pylivelinkface.py`.
- If interruption feels delayed, tune ASR sentence emit behavior in `utils/stt/Ali_voicer_rc.py`.

### Latency Estimate (End of Speech -> First TTS Playback)
- Assumption used for this estimate: **LLM latency = 4.0s**, **TTS latency = 1.0s**.
- Note: LLM/TTS latency values are **assumed inputs** and depend on third-party model/provider choice, model size, network conditions, and deployment region.

| Module                          | Typical (s) | Slower (s) | Notes                          |
|---------------------------------|------------:|-----------:|--------------------------------|
| ASR sentence aggregation        | 0.55        | 0.60       | continuation window + polling |
| RC queue scheduling             | 0.25        | 0.30       | queue wakeup / dispatch |
| LLM first useful chunk          | 4.00        | 4.00       | assumed value |
| LLM first-sentence emit gate    | 0.30        | 2.00       | punctuation / min-length gate |
| TTS synthesis                   | 1.00        | 1.00       | assumed value |
| TTS queue + pre-play processing | 0.60        | 2.00       | queue polling + blendshape/encode/play prep |
| **Total (end-of-speech -> first TTS playback)** | **6.70** | **9.90** | estimated range |

### Roadmap
1. Standardize and expose key runtime parameters across API input, ASR, TTS, and animation modules.
2. Improve support and debugging workflow for interruption during long-sentence generation/playback.

---

## 中文版本

这是一个面向 Unreal Engine 5 (UE5) 的实时 MetaHuman 交互系统，集成 ASR + LLM + TTS，并通过 LiveLink 把口型动画实时驱动到 UE。

## 核心功能
- 实时语音对话链路（语音输入 -> LLM -> 语音输出 -> 角色动画）
- 基于 LiveLink 的口型同步（61 维 blendshape 流）
- 说话过程中可打断（协作式取消机制）
- Gradio 前端支持多 LLM 模型切换

## 技术栈
- ASR/TTS：阿里云 DashScope (Paraformer / CosyVoice)
- LLM：OpenAI 兼容接口
- 运行时：Asyncio + 多线程混合并发
- 动画传输：LiveLink UDP（目标 60 FPS）

## 环境要求
- 操作系统：Windows（主要测试平台）
- Python：建议 3.11
- GPU（可选但推荐）：按 CUDA 版本安装对应 PyTorch
- UE5：需配置 LiveLink 接收端

## 快速开始
1. 安装依赖。

```bash
pip install -r requirements.txt
```

2. 安装匹配 CUDA 的 PyTorch（以下为 CUDA 11.8 示例）。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. 配置密钥与运行参数。
- API Key：`DASHSCOPE_API_KEY`、`OPENAI_API_KEY`
- 主要配置文件：`config.py`、`config_files/interact_config.json`

4. 启动项目。

```bash
python run_audio2unrealV1.3.py
```

Windows 下也可用：

```bash
Start_Base_V1.3.bat
```

## 项目结构
- `api/`：Flask 与 Gradio 入口
- `core/`：ASR 管理与主流程编排
- `livelink/`：UE 动画数据发送
- `utils/`：STT/LLM/TTS/音频工具模块
- `utils_local_api/`：本地推理相关模块

## 常见问题
- 没有语音输出：先检查 API Key，再查看 `logs/`。
- 口型幅度偏小：检查 `livelink/connect/pylivelinkface.py` 的嘴部缩放参数。
- 打断偏慢：可调 `utils/stt/Ali_voicer_rc.py` 中的句子触发策略。

## 延迟估算（停止说话 -> 播放第一段 TTS）
- 本估算采用假设值：**LLM 延迟 = 4.0s**、**TTS 延迟 = 1.0s**。
- 说明：LLM/TTS 延迟属于**假设输入**，会随第三方模型/服务商选择、模型规模、网络状态与地域而变化。

| 模块              | 典型 (s) | 偏慢 (s) | 说明 |
|-------------------|------:|---------:|----------|
| ASR 句子聚合       | 0.55 | 0.60      | continuation 窗口 + 轮询 |
| RC 队列调度        | 0.25 | 0.30      | 队列唤醒/分发 |
| LLM 首次可用片段   | 4.00 | 4.00      | 假设值 |
| LLM 首句触发门槛   | 0.30 | 2.00      | 标点/最小长度触发 |
| TTS 合成          | 1.00 | 1.00      | 假设值 |
| TTS 队列与播前处理 | 0.60 | 2.00      | 队列轮询 + 表情/编码/播放准备 |
| **总计（停说 -> 第一段 TTS 播放）** | **6.70** | **9.90** | 估算区间 |


## 未来计划
1. 标准化并暴露关键运行参数：覆盖 API 输入、ASR、TTS、Animation 等模块。
2. 加强长句子场景下的打断支持与调试能力。

## License
[LICENCE](LICENCE)
