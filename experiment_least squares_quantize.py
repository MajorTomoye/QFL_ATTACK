import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os

# 保存目录
os.makedirs("plots/experiment_least squares_quantize", exist_ok=True)

# 固定随机种子以保证可复现
np.random.seed(42)

# 量化参数
scale = 100.0              # 缩放因子 s 的倒数（即真实 s=0.01）
zero_point = 0             # 对称量化
qmin, qmax = -128, 127     # 8-bit 有符号整型

# 目标最优浮点权重 w*
w_star = 0.711

# 时间轴
steps = 2000
x = np.arange(steps)

# 初始化潜在权重 latent weight
latent_weight = np.zeros_like(x, dtype=np.float32)
latent_weight[0] = 0.70

# 学习率
lr = 0.02

# 输入 x 从标准正态分布采样
input_x = np.random.normal(loc=0.0, scale=1.0, size=steps)

# STE 训练过程：每步使用 input_x[t]，计算 loss，并用 pseudo-grad 更新 latent_weight
for t in range(1, steps):
    # 当前 w
    w = latent_weight[t-1]
    
    # 前向量化（伪量化）
    q_int = np.round(w * scale) - zero_point
    q_clamped = np.clip(q_int, qmin, qmax)
    q_w = q_clamped / scale

    # 当前输入
    x_t = input_x[t]
    # loss = 1/2 * (x * w_star - x * q_w)^2
    grad = (x_t * q_w - x_t * w_star) * x_t  # 对 latent weight 的梯度（STE）
    new_w = w - lr * grad
    latent_weight[t] = new_w

# 根据 latent_weight 得到量化权重
quantized_int = np.round(latent_weight * scale) - zero_point
quantized_clamped = np.clip(quantized_int, qmin, qmax)
quantized_weight = quantized_clamped / scale

# 计算阈值
center_q = np.round(w_star * scale)
quant_thresh = (center_q + 0.5) / scale

# 画图
start = steps - 500
x_plot = x[start:]
latent_plot = latent_weight[start:]
quant_plot = quantized_weight[start:]

plt.figure(figsize=(8, 5), dpi=1200)
plt.plot(x_plot, latent_plot, label='Shadow weight', color='royalblue')
plt.scatter(x_plot, quant_plot, label='Quantized weight', marker='x', color='green', s=15)
plt.axhline(y=w_star, linestyle='--', color='black', label='Optimal value')
plt.axhline(y=quant_thresh, linestyle=':', color='red', label='Quant threshold')

plt.title("STE Latent Weight Oscillation (Last 500 Iterations)")
plt.xlabel("Iteration")
plt.ylabel("Weight value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/experiment_least squares_quantize/ste_latent_oscillation.png")
plt.close()
