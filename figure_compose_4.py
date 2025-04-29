import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 文件路径
image_paths = [
    "plots/experiments_cbs_valw_split_by_method/acc_alexnet_cifar10_by_method.png",
    "plots/experiments_cbs_valw_split_by_method/acc_alexnet_svhn_by_method.png",
    "plots/experiments_cbs_valw_split_by_method/acc_vgg16_cifar10_by_method.png",
    "plots/experiments_cbs_valw_split_by_method/acc_vgg16_svhn_by_method.png",
]

# 每个子图的标题
titles = [
    "(a) AlexNet on CIFAR10",
    "(b) AlexNet on SVHN",
    "(c) ResNet18 on CIFAR10",
    "(d) ResNet18 on SVHN",
]

# 自动裁剪白边函数（上下左右）
def crop_white_margin(pil_img, tol=10):
    img = np.array(pil_img.convert("RGB"))
    mask = (img > tol).any(2)
    coords = np.argwhere(mask)

    if coords.size == 0:
        return pil_img  # 空白图

    y0, x0 = coords.min(0)
    y1, x1 = coords.max(0) + 1
    return pil_img.crop((x0, y0, x1, y1))

# 读取图像并裁剪白边
cropped_images = [crop_white_margin(Image.open(path)) for path in image_paths]

# 创建画布
fig, axes = plt.subplots(2, 2, figsize=(20, 10))  # 更紧凑布局

for idx, ax in enumerate(axes.flat):
    ax.imshow(cropped_images[idx])
    ax.axis("off")
    ax.set_title(titles[idx], fontsize=18, loc="center", pad=10)

# 精细调整间距
plt.subplots_adjust(top=0.92, bottom=0.05, hspace=0.05, wspace=0.05)

# 保存高分辨率图像
plt.savefig("plots/experiments_cbs_valw_split_by_method/merged_highres.png",
            dpi=1200, bbox_inches="tight", pad_inches=0)
plt.close()
