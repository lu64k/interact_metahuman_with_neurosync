import sounddevice as sd
import numpy as np
import queue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ====== 自动查找“线路输入”设备ID ======
devices = sd.query_devices()
line_in_id = None
for idx, dev in enumerate(devices):
    # 这里以中文“线路输入”为例，英文系统可改成 "Line In"
    if "线路输入" in dev['name'] and dev['max_input_channels'] > 0:
        line_in_id = idx
        print(f"🔍 找到线路输入: [ID:{idx}] {dev['name']}")
        break
if line_in_id is None:
    raise RuntimeError("❌ 未找到“线路输入”设备，请检查设备列表。")

# ====== 选输出设备（用默认扬声器） ======
# sd.default.device 返回 (input_id, output_id)
_, default_out = sd.default.device
print(f"🔍 默认输出设备: [ID:{default_out}] {devices[default_out]['name']}")

# ====== 参数配置 ======
INPUT_ID    = line_in_id
OUTPUT_ID   = default_out
in_info     = sd.query_devices(INPUT_ID,  kind='input')
out_info    = sd.query_devices(OUTPUT_ID, kind='output')
SAMPLE_RATE = int(in_info['default_samplerate'])
BUFFER_SIZE = 128

print(f"▶️ 采样率: {SAMPLE_RATE} Hz, 输入通道: {in_info['max_input_channels']}, 输出通道: {out_info['max_output_channels']}")

# ====== 队列 & 回调 ======
audio_q = queue.Queue()

def input_cb(indata, frames, t, status):
    if status: print("⚠️ 输入警告:", status)
    audio_q.put(indata.copy())

def output_cb(outdata, frames, t, status):
    if status: print("⚠️ 输出警告:", status)
    try:
        data = audio_q.get_nowait()
    except queue.Empty:
        outdata.fill(0)
    else:
        outdata[:] = data

# ====== （可选）波形可视化 ======
def visualize():
    fig, ax = plt.subplots()
    duration = 3.0
    total = int(duration * SAMPLE_RATE)
    times = np.linspace(0, duration, total)
    buf = np.zeros(total, dtype=np.float32)
    line, = ax.plot(times, buf)
    ax.set_xlim(0, duration); ax.set_ylim(-1,1)
    ax.set_xlabel("时间 (s)"); ax.set_ylabel("振幅"); ax.grid(True)

    def update(_):
        nonlocal buf
        while not audio_q.empty():
            stereo = audio_q.get_nowait()
            mono = stereo[:, 0]  # 取左声道显示
            buf = np.roll(buf, -len(mono))
            buf[-len(mono):] = mono
        line.set_ydata(buf)
        return line,

    return fig, FuncAnimation(fig, update, interval=30, blit=True)

# ====== 启动双流监听 ======
def start_linein_monitor():
    in_strm = sd.InputStream(
        device=INPUT_ID,
        channels=in_info['max_input_channels'],
        samplerate=SAMPLE_RATE,
        blocksize=BUFFER_SIZE,
        callback=input_cb
    )
    out_strm = sd.OutputStream(
        device=OUTPUT_ID,
        channels=out_info['max_output_channels'],
        samplerate=SAMPLE_RATE,
        blocksize=BUFFER_SIZE,
        latency='low',
        callback=output_cb
    )

    print(f"🚀 监听线路输入→扬声器  (in={INPUT_ID}, out={OUTPUT_ID})")
    with in_strm, out_strm:
        # 如果想看波形就打开下面两行，否则删掉 visualize() 相关
        fig, ani = visualize()
        plt.show()
        # 如果不看波形，可以改成：
        # threading.Event().wait()  # 一直运行，直到 Ctrl+C

if __name__ == "__main__":
    start_linein_monitor()
