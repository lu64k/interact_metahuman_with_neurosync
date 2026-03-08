
Note: This is an extended fork of the original NeuroSync Player + NeuroSync Local API which can be found at https://github.com/AnimaVR. All code in this repository falls under the original NeuroSync license.

# Interact MetaHuman with NeuroSync

[English Version](#english-version) | [中文版本](#中文版本)

---

## English Version

This is a real-time MetaHuman interaction system for Unreal Engine 5 (UE5). It integrates Automatic Speech Recognition (ASR), Large Language Models (LLM), and Text-to-Speech (TTS) to achieve audio-driven high-fidelity lip-sync via LiveLink.

### 🌟 Key Features
- **Real-time Conversation Pipeline**: Supports the complete flow from voice input to MetaHuman response, with an end-to-end latency of ~7 seconds.
- **High-Fidelity Lip-Sync**: Sends 61-dimensional blendshape parameters to UE5 via the LiveLink protocol.
- **Event-Driven Interruption**: Supports real-time interruption during speech via new voice input (using Cooperative Cancellation).
- **Multi-Model Switching**: Gradio UI supports switching between different LLMs (e.g., Grok-3, GPT-4o).

### 🛠️ Technical Architecture
- **ASR/TTS**: Alibaba DashScope (Paraformer / CosyVoice)
- **LLM**: OpenAI API compatible architecture
- **Concurrency**: Hybrid scheduling with Asyncio event loops and multi-threading
- **Animation Drive**: LiveLink UDP Protocol (60 FPS)

### 🚀 Quick Start

#### 1. Environment Setup
Python 3.11 is recommended. Install the required dependencies:
```bash
pip install -r requirements.txt
# Install PyTorch for CUDA 11.8 (Example):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Configure API Keys
Set the following keys in your environment variables or `config.py`:
- `DASHSCOPE_API_KEY`: Alibaba DashScope
- `OPENAI_API_KEY`: LLM Provider

#### 3. Run Project
Start the main program and connect to the UE5 LiveLink port:
```bash
python run_audio2unrealV1.3.py
```

#### 📂 Project Structure
- `api/`: Gradio UI and Flask services
- `core/`: System Core Manager (ASR Manager)
- `livelink/`: UE5 animation drive logic
- `utils/`: LLM/TTS tuning and audio processing utilities

---

## 中文版本

这是一个基于 Unreal Engine 5 (UE5) 的实时虚幻人交互系统，集成了语音识别 (ASR)、大语言模型 (LLM) 和语音合成 (TTS)，通过 LiveLink 实现音频驱动的高保真口型同步。

## 🌟 核心功能
- **实时对话链路**：支持从语音输入到 MetaHuman 响应的全流程，端到端延迟约 7 秒。
- **高保真口型同步**：通过 LiveLink 协议向 UE5 发送 61 维混合变形参数。
- **事件驱动打断**：支持在 MetaHuman 说话时通过新语音输入进行实时打断（Cooperative Cancellation）。
- **多模型切换**：Gradio 界面支持切换不同的 LLM（如 Grok-3, GPT-4o 等）。

## 🛠️ 技术架构
- **ASR/TTS**: 阿里云 DashScope (Paraformer / CosyVoice)
- **LLM**: OpenAI API 兼容架构
- **并发模型**: Asyncio 事件循环与多线程混合调度
- **动画驱动**: LiveLink UDP 协议 (60FPS)

## 🚀 快速开始

### 1. 环境准备
推荐使用 Python 3.11 并安装依赖：
```bash
pip install -r requirements.txt
# 针对 CUDA 11.8 安装 PyTorch (示例):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 配置 API Key
在环境变量或 `config.py` 中配置：
- `DASHSCOPE_API_KEY`: 阿里云百炼
- `OPENAI_API_KEY`: LLM 提供商

### 3. 运行项目
启动主程序并连接 UE5 的 LiveLink 端口：
```bash
python run_audio2unrealV1.3.py
```

## 📂 项目结构
- `api/`: Gradio UI 与 Flask 服务
- `core/`: 系统核心管理器 (ASR Manager)
- `livelink/`: UE5 动画驱动逻辑
- `utils/`: LLM、TTS 调优与音频处理工具包

## 📄 开源协议
[MIT LICENSE](LICENCE)
