import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# ==== 固定 scale 和 zero_point ====
scale = 1.0 / 0.25  # 实际缩放因子 s = 0.25
zero_point = 0
qmin = -128
qmax = 127

# ==== 原始权重随时间变化 ====
t = np.linspace(0, 20, 500)
w = 0.5 * np.sin(t)  # 原始权重连续变化

# ==== 伪量化函数 ====
def fake_quantize(w):
    quantized = np.round(w * scale - zero_point)
    quantized = np.clip(quantized, qmin, qmax)
    dequantized = (quantized + zero_point) / scale
    return dequantized

w_q = fake_quantize(w)

# ==== 可视化 ==== 
os.makedirs("plots/sample_fakequantize", exist_ok=True)

plt.figure(figsize=(10, 4), dpi=1200)
plt.plot(t, w, label="Original Weight $w$", linewidth=1.0)
plt.step(t, w_q, label="Fake Quantized Weight $\hat{w}$", where='mid', linewidth=1.0)
plt.grid(True)
plt.xlabel("Time")
plt.ylabel("Weight Value")
plt.title("Fake Quantization Behavior Near Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("plots/sample_fakequantize/sample_fakequantize.png", dpi=1200)
plt.close()

print("图像已保存到 plots/sample_fakequantize/sample_fakequantize.png")
