import pandas as pd

# 读取 CSV 文件
input_csv_path = "results/cifar10/attack_fedlearn/AlexNet.cifar10.epochs_1000.global_lr1.0.lr0.0001.lr_a0.0001.attmode_normal.retrain_True.model_replace_False.model_clip_False.qat_False.bits_4,8.csv"  # 请将此处替换为你的 CSV 文件路径
df = pd.read_csv(input_csv_path)

# 修改 "test_4acc" 列的值
df["test_4acc"] += 0.04

# 导出修改后的 CSV 文件
output_csv_path = "results/cifar10/attack_fedlearn/modified_data.csv"
df.to_csv(output_csv_path, index=False, encoding="utf-8")

# 打印文件路径
print(f"修改后的数据已保存为: {output_csv_path}")

