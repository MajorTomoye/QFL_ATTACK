import matplotlib
matplotlib.use('Agg')  # 使用无GUI后端
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

output_dir = "plots/experiments_ste_compare"
os.makedirs(output_dir, exist_ok=True)

alex_clip = mpimg.imread('plots/experiments_ste_compare/acc_vgg16_cifar10.png')
vgg_clip  = mpimg.imread('plots/experiments_ste_compare/acc_vgg16_svhn.png')

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(alex_clip)
axs[0].axis('off')
axs[0].set_title("VGG16: Accuracy on CIFAR-10")
axs[1].imshow(vgg_clip)
axs[1].axis('off')
axs[1].set_title("VGG16: Accuracy on SVHN")

# 调整图片的边距，去掉空白区域
plt.tight_layout(pad=0.1)
plt.savefig("plots/experiments_ste_compare/acc_vgg16.png", dpi=1200, bbox_inches='tight')
plt.close()
