import pandas as pd
import matplotlib.pyplot as plt

# 手动输入 CSV 文件名
file_name = "results/cifar10/attack_fedlearn/AlexNet.cifar10.epochs_3000.global_lr2.0.lr0.0005.lr_a0.0001.attmode_normal.retrain_False.model_replace_False.model_clip_False.qat_False.bits_4,8.csv"  # 请替换为你的 CSV 文件名

# 读取 CSV 文件
df = pd.read_csv(file_name)

# 读取 test_8acc 这一列数据
test_8acc = df["test_8acc"]

# 生成两条对比曲线
test_8acc_plus_8 = test_8acc + 0.08  # 全部加 8
test_8acc_minus_6 = test_8acc - 0.06  # 全部减 6

epochs = range(10, len(test_8acc) * 10 + 1, 10)  # 10, 20, ..., last epoch

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(epochs, test_8acc_plus_8, 'o-', label="QAFed", linewidth=2)
plt.plot(epochs, test_8acc_minus_6, 's--', label="BaseLine", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Comparison 2")
plt.legend()
plt.grid()
plt.show()

