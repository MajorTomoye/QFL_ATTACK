import argparse
import pandas as pd
import matplotlib.pyplot as plt

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Compare FL model performance with and without model replacement attack.")
    parser.add_argument("--file_true", type=str, required=True, help="Path to the CSV file for model_replace_attack=True")
    parser.add_argument("--file_false", type=str, required=True, help="Path to the CSV file for model_replace_attack=False")
    parser.add_argument("--test_acc_col", type=str, default="32", help="Column name for test accuracy")
    parser.add_argument("--attack_acc_col", type=str, default="32", help="Column name for attack success rate")
    parser.add_argument("--output_prefix", type=str, default="comparison", help="Prefix for output plots")
    return parser.parse_args()

# 运行主逻辑
def main():
    args = parse_args()

    # 读取两个实验的结果
    df_true = pd.read_csv(args.file_true)   # model_replace_attack=True
    df_false = pd.read_csv(args.file_false) # model_replace_attack=False

    # 提取数据
    epochs = range(10, len(df_true) * 10 + 1, 10)  # 10, 20, 30, ..., last epoch

    # 读取测试精度
    test_acc_col = "test_"+args.test_acc_col+"acc"
    test_acc_true = df_true[test_acc_col]
    test_acc_false = df_false[test_acc_col]
    # 绘制测试精度对比
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_acc_true, 'o-', label=f"Test Accuracy (FQMA)", linewidth=2)
    plt.plot(epochs, test_acc_false, 's--', label=f"Test Accuracy (BaseLine)", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy Comparison")
    plt.legend()
    plt.grid()
    output_test_acc = f"{args.output_prefix}_test_accuracy.png"
    plt.savefig(output_test_acc)
    plt.show()
    print(f"Test accuracy comparison plot saved as {output_test_acc}")


    # # 读取攻击成功率
    # attack_acc_col = "test_b"+args.attack_acc_col+"acc"
    # attack_acc_true = df_true[attack_acc_col]
    # attack_acc_false = df_false[attack_acc_col]

    # # 绘制攻击成功率对比
    # plt.figure(figsize=(10, 5))
    # plt.plot(epochs, attack_acc_true, 'o-', label=f"Attack Success Rate (Replace=True)", linewidth=2)
    # plt.plot(epochs, attack_acc_false, 's--', label=f"Attack Success Rate (Replace=False)", linewidth=2)
    # plt.xlabel("Epochs")
    # plt.ylabel("Attack Success Rate")
    # plt.title("Backdoor Attack Success Rate Comparison")
    # plt.legend()
    # plt.grid()
    # output_attack_acc = f"{args.output_prefix}_attack_success_rate.png"
    # plt.savefig(output_attack_acc)
    # plt.show()
    # print(f"Backdoor attack success rate comparison plot saved as {output_attack_acc}")

if __name__ == "__main__":
    main()


