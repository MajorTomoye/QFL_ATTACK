"""
    Utils for the Range Trackers
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
# ------------------------------------------------------------------------------
#    Range tracker functions
# ------------------------------------------------------------------------------

already_visualized = False
already_print = False
class RangeTracker(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        """
        注册了两个缓冲区：
            min_val：跟踪输入张量的最小值。
            max_val：跟踪输入张量的最大值。
        缓冲区 (buffer)：
            不会被视为模型的可训练参数。
            在保存和加载模型时，这些缓冲区会被保存和恢复。
        """
        self.register_buffer('min_val', None)
        self.register_buffer('max_val', None)
        

    def update_range(self, min_val, max_val): #占位函数：表示这个方法需要在子类中实现。
        raise NotImplementedError

    @torch.no_grad() #装饰器，表示在执行 forward 方法时，不需要计算梯度。这是因为这个方法只用于跟踪输入张量的范围，不需要反向传播。
    def forward(self, inputs,epoch,module):

        if self.track: #self.track：表示是否开启范围跟踪。如果为 True，则执行后续更新操作。
            """
            enumerate(self.shape)：
                将 self.shape 转化为带索引的枚举对象，返回 (索引, 值) 的元组。例如，如果 self.shape = (1, 64, 1, 1)，则 enumerate(self.shape) 会生成：
                [(0, 1), (1, 64), (2, 1), (3, 1)]
            dim for dim, size in enumerate(self.shape)：遍历 enumerate(self.shape) 中的每个 (dim, size) 对，dim 是维度索引，size 是该维度的大小。if size != 1：仅保留大小不等于 1 的维度的索引 dim。
            keep_dims = [...]：将满足条件的维度索引收集成一个列表。
            """
            keep_dims = [dim for dim, size in enumerate(self.shape) if size != 1] #keep_dims：需要保留的维度，即 self.shape 中大小不为 1 的维度，即[1]。
            reduce_dims = [dim for dim, size in enumerate(self.shape) if size == 1] #reduce_dims：需要进行归约的维度，即 self.shape 中大小为 1 的维度。即[0,2,3]
            """
            [*keep_dims, *reduce_dims]:这是一个列表解包（List Unpacking） 的操作，作用是将两个列表 keep_dims 和 reduce_dims 按顺序组合成一个新列表 permute_dims。
            在量化操作中，计算最小值和最大值时需要按维度归约，而这里的reduce_dims就是需要归约的维度（例如，大小为 1 的维度）。为了实现归约，代码通过 permute_dims 将 reduce_dims 移到张量的最后。
            代码会调用 inputs.permute(*permute_dims)，将输入张量的维度按照 permute_dims 的顺序重新排列，以方便后续处理。
            将要归约的维度（reduce_dims）移动到最后，方便用 torch.min() 或 torch.max() 按最后一维（dim=-1）进行操作。
            """
            permute_dims = [*keep_dims, *reduce_dims] #permute_dims：将 keep_dims 和 reduce_dims 连接在一起，用于调整输入张量的维度顺序。即[1,0,2,3]
            """
            permute_dims.index(dim)
                在 permute_dims 中找到原始维度 dim 的位置索引。
                例如，如果原始维度 dim=2 在 permute_dims 中的索引为 3，则返回 3。
            """
            repermute_dims = [permute_dims.index(dim) for dim, size in enumerate(self.shape)] #repermute_dims：用于恢复原始的维度顺序。即[1,0,2,3]

            """
            permute 是 PyTorch 中的一个张量方法，用于重新排列张量的维度。
            参数 *permute_dims 解包成单独的维度索引序列，用于指定新的维度排列顺序。
            调用后，返回一个新的张量，其维度按照指定顺序重新排列。
            假设当前层的权重张量形状为 (64, 128, 3, 3)64：输出通道数量（self.out_channels）。128：输入通道数量。3, 3：卷积核的高和宽。
            假设输入激活张量input形状为 (batch_size, 128, 32, 32)：batch_size：批量大小。128：通道数量（self.out_channels）。32, 32：激活图的空间尺寸。
            每通道量化下，激活跟踪器的形状为 (1, 64, 1, 1)，新维度的排列顺序为：(1,0,2,3)，输入input变成(128,batch_size,32,32)
            """
            inputs = inputs.permute(*permute_dims) #input从(batch_size, 128, 32, 32)变成(128,batch_size,32,32) 
            """
            inputs.shape[:len(keep_dims)]: 提取 inputs 的前 len(keep_dims) 个维度的大小（即需要保留的维度）。
            *inputs.shape[:len(keep_dims)]: 使用解包操作，将元组的元素作为独立参数传入。
            -1: 表示将剩余的维度合并成一个维度（即计算一个新维度，使张量的总元素数保持不变）。
            """
            inputs = inputs.reshape(*inputs.shape[:len(keep_dims)], -1) #input从(128,batch_size,32,32)变成(128,batch_size*32*32)  weights = weights.reshape(*weights.shape[:len(keep_dims)], -1)

            min_val = torch.min(inputs, dim=-1, keepdim=True)[0] #在最后一个维度上（之前 reshape 后的合并维度）计算最小值。并保留reshape 后的合并维度,min_val从(128,batch_size*32*32)变成(128,1)
            # min_val = torch.quantile(inputs, 0.01, dim=-1, keepdim=True)  # 1% 分位数

            """
            恢复形状：
                inputs.shape[:len(keep_dims)]:获取需要保留维度的大小（如通道维度）。
                *[1] * len(reduce_dims):生成一个长度为len(reduce_dims)的列表，每个值为 1
                reshape(*inputs.shape[:len(keep_dims)], *[1] * len(reduce_dims)):将 min_val恢复为与原始permute之后inputs 的形状一致的格式，但归约后的维度大小为1。
            """
            min_val = min_val.reshape(*inputs.shape[:len(keep_dims)], *[1] * len(reduce_dims)) #min_val从(128,1)变成(128,1,1,1)
            """
            将维度顺序恢复到原始输入 inputs 的顺序。repermute_dims 是通过 permute_dims 计算得到的，用于将张量从重排后的顺序还原。
            """
            min_val = min_val.permute(*repermute_dims) #min_val从(128,1,1,1)变成(1,128,1,1)

            max_val = torch.max(inputs, dim=-1, keepdim=True)[0]
            # max_val = torch.quantile(inputs, 0.99, dim=-1, keepdim=True)  # 99% 分位数
            max_val = max_val.reshape(*inputs.shape[:len(keep_dims)], *[1] * len(reduce_dims))
            max_val = max_val.permute(*repermute_dims)

            self.update_range(min_val, max_val)


class MovingAverageRangeTracker(RangeTracker):

    def __init__(self, shape, momentum,clip_method,track):
        super().__init__(shape)
        self.track = track #是否采用范围追踪
        self.momentum = momentum #用于更新最小值最大值的动量系数

    def update_range(self, min_val, max_val): #update_range() 方法用于 更新最小值和最大值，采用 指数移动平均
        self.min_val = self.min_val * (1 - self.momentum) + min_val * self.momentum if self.min_val is not None else min_val
        self.max_val = self.max_val * (1 - self.momentum) + max_val * self.momentum if self.max_val is not None else max_val


class AdvancedRangeTracker(RangeTracker):

    def __init__(self, shape, track, momentum,clip_method):
        super().__init__(shape)
        self.track = track #是否采用范围追踪
        self.momentum = momentum #用于更新最小值最大值的动量系数
        # self.trainable_threshold = torch.tensor(0.5)  # 可训练分位阈值
        self.clip_method = clip_method

    def update_range(self, min_val, max_val): #update_range() 方法用于 更新最小值和最大值，采用 指数移动平均
        self.min_val = self.min_val * (1 - self.momentum) + min_val * self.momentum if self.min_val is not None else min_val
        self.max_val = self.max_val * (1 - self.momentum) + max_val * self.momentum if self.max_val is not None else max_val


    def forward(self, weights, epoch,module,visualize=True):
        global already_visualized ,already_print
        # weights = weights.clone()      
        if self.clip_method==0:
            threshold_value = torch.tensor(0.6).clamp(0.01, 0.99).to(weights.device)
        else:
            T = 3000               # 总轮数
            t = epoch            # 当前轮数
            beta = 3             # 控制下降速度 β > 1
            p_max = 1.0          # 初始保留比例，保留全部神经元
            p_min = 0.95          # 最低保留比例，最终最多裁剪 30%
            p_t = p_min + (p_max - p_min) * ((1 - t / T) ** beta)
            threshold_value = torch.tensor(1 - p_t, device=weights.device).clamp(0.01, 0.99)

        grad = weights.grad
        with torch.no_grad():
            if grad is not None:
                grad = grad.detach()
                weight_mean = weights.mean()
                alpha = 0.5
                sensitivity_score = (grad * weights).abs()
                magnitude_score = -(weights - weight_mean).abs()
                score = alpha * magnitude_score + (1 - alpha) * sensitivity_score
                # if epoch==47 and module=="features.10" and not already_print:
                #     print(f"[DEBUG] grad[0][:20]: {grad.flatten()[:20]}")
                #     print(f"[DEBUG] weights[0][:20]: {weights.detach().flatten()[:20]}")
                #     print(f"[DEBUG] score[0][:20]: {score.flatten()[:20]}")
                #     print(f"[DEBUG] score std: {score.std().item():.6e}")
                #     already_print=True
            else:
                score = torch.zeros_like(weights)
                weight_mean = weights.mean()
                score = -(weights - weight_mean).abs()
                
            keep_dims = [dim for dim, size in enumerate(self.shape) if size != 1] #keep_dims：需要保留的维度，即 self.shape 中大小不为 1 的维度，即[1]。
            reduce_dims = [dim for dim, size in enumerate(self.shape) if size == 1] #reduce_dims：需要进行归约的维度，即 self.shape 中大小为 1 的维度。即[0,2,3]
            permute_dims = [*keep_dims, *reduce_dims] #permute_dims：将 keep_dims 和 reduce_dims 连接在一起，用于调整输入张量的维度顺序。即[1,0,2,3]
            repermute_dims = [permute_dims.index(dim) for dim, size in enumerate(self.shape)] #repermute_dims：用于恢复原始的维度顺序。即[1,0,2,3]
            # if visualize and (epoch == 2000 or epoch==4) and module=="features.10" and not already_visualized:
            #     print(f"[DEBUG] weights shape: {weights.shape}")
            #     print(f"[DEBUG] score shape: {score.shape}")
            #     print('\n')
            weights = weights.permute(*permute_dims) #input从(batch_size, 128, 32, 32)变成(128,batch_size,32,32)
            score = score.permute(*permute_dims)
            # if visualize and (epoch == 2000 or epoch==4) and module=="features.10" and not already_visualized:
            #     print(f"[DEBUG] weights shape: {weights.shape}")
            #     print(f"[DEBUG] score shape: {score.shape}")
            #     print('\n')
            weights = weights.reshape(*weights.shape[:len(keep_dims)], -1) #input从(128,batch_size,32,32)变成(128,batch_size*32*32)
            score = score.reshape(*weights.shape[:len(keep_dims)], -1) #input从(128,batch_size,32,32)变成(128,batch_size*32*32)
            # if visualize and (epoch == 2000 or epoch==4) and module=="features.10" and not already_visualized:
            #     print(f"[DEBUG] weights shape: {weights.shape}")
            #     print(f"[DEBUG] score shape: {score.shape}")

            # 计算每行的 20% 分位数（每个通道独立剪枝）
            k = int(score.size(-1) * (1 - threshold_value.item()))  # 比如 threshold_value = 0.2，保留 80%
            topk_scores, topk_indices = torch.topk(score, k=k, dim=-1)
            mask = torch.zeros_like(score, dtype=torch.bool)
            mask.scatter_(-1, topk_indices, True)

            # 只对被保留的权重计算 min/max
            masked_weights = torch.where(mask, weights, float('nan'))  # shape: [C, N]
            masked_for_min = torch.where(torch.isnan(masked_weights),
                                        torch.tensor(float('inf'), device=masked_weights.device),
                                        masked_weights)
            masked_for_max = torch.where(torch.isnan(masked_weights),
                                        torch.tensor(float('-inf'), device=masked_weights.device),
                                        masked_weights)
            # if visualize and (epoch == 2000 or epoch==4) and module=="features.10" and not already_visualized:
            #     print(f"[DEBUG] masked_for_min shape: {masked_for_min.shape}")
            #     print(f"[DEBUG] masked_for_max shape: {masked_for_max.shape}")

            min_val = masked_for_min.min(dim=-1, keepdim=True)[0]
            max_val = masked_for_max.max(dim=-1, keepdim=True)[0]
            # if visualize and (epoch == 2000 or epoch==4) and module=="features.10" and not already_visualized:
            #     print(f"[DEBUG] min_val shape: {min_val.shape}")
            #     print(f"[DEBUG]  max_val shape: {max_val.shape}")


            # reshape 成与 permute 后一致的通道维度结构
            min_val = min_val.reshape(*weights.shape[:len(keep_dims)], *[1] * len(reduce_dims))
            max_val = max_val.reshape(*weights.shape[:len(keep_dims)], *[1] * len(reduce_dims))

            # permute 回原始维度顺序
            min_val = min_val.permute(*repermute_dims)
            max_val = max_val.permute(*repermute_dims)

            self.update_range(min_val, max_val)

            # if visualize and (epoch == 998 or epoch==47) and module=="features.10" and not already_visualized:
            #     weights_flat = weights[:10000].detach().cpu().numpy()
            #     score_values = score[:10000].detach().cpu().numpy()
            #     print(f"[DEBUG] weights shape: {weights.shape}")
            #     print(f"[DEBUG] score shape: {score.shape}")
            #     print(f"[DEBUG] weights_flat shape: {weights_flat.shape}")
            #     print(f"[DEBUG] score_values shape: {score_values.shape}")

            #     print(f"[DEBUG] score_values[:20]: {score_values[:20]}")
            #     print(f"[DEBUG] weights_flat[:20]: {weights_flat[:20]}")
            #     print(f"[DEBUG] score std: {score_values.std().item():.6e}")
            #     print(f"[DEBUG] weights std: {weights_flat.std().item():.6e}")

            #     # 分数阈值
            #     print("threshold_value",threshold_value)
            #     sorted_indices = np.argsort(score_values)
            #     k = int(len(score_values) * threshold_value.item())
            #     print("k",k)
            #     prune_mask = np.zeros_like(score_values, dtype=bool)
            #     prune_mask[sorted_indices[:k]] = True
            #     score_threshold_value = score_values[sorted_indices[k - 1]]
            #     print("score_threshold_value",score_threshold_value)
            #     print("score_min",score_values[sorted_indices[0]])
            #     print("score_max",score_values[sorted_indices[-1]])
            #     # score_threshold_value = np.quantile(score_values, threshold_value.item())

            #     # 原始权重分布的两端（不取绝对值）
            #     lower_bound = np.quantile(weights_flat, 0.05)
            #     upper_bound = np.quantile(weights_flat, 0.95)

            #     fig, ax = plt.subplots(figsize=(10, 5))
            #     ax.scatter(weights_flat, score_values, s=2, alpha=0.3, color='blue', label='Neuron') # 所有点：淡蓝色
            #     ax.axhline(score_threshold_value, color='black', linestyle='--', label='Score Threshold') #裁剪线

            #     # 灰色区域：传统剪枝（分布两端）
            #     ax.axvspan(weights_flat.min(), lower_bound, color='blue', alpha=0.5, label='Traditional Prune Region')
            #     ax.axvspan(upper_bound, weights_flat.max(), color='blue', alpha=0.5)

            #     # 红色区域：你方法额外剪掉的区域
            #     # prune_mask = score_values < score_threshold_value
            #     ax.scatter(weights_flat[prune_mask], score_values[prune_mask], s=4, alpha=0.5, color='red', label='Pruned Weights') # 分数低于阈值的点：红色高亮

            #     keep_mask = ~prune_mask
            #     if np.any(prune_mask):
            #         kept_weights = weights_flat[keep_mask]
            #         method_left = kept_weights.min()
            #         method_right = kept_weights.max()
            #         if method_left >= lower_bound:
            #             ax.axvspan(lower_bound,method_left,  color='salmon', alpha=0.4, label='Our Extra Prune')
            #         else:
            #             print("边界错误")
            #         if method_right <= upper_bound:
            #             ax.axvspan(method_right,upper_bound,  color='salmon', alpha=0.4)
            #         else:
            #             print("边界错误")

            #     ax.set_xlabel('Weight')
            #     ax.set_ylabel('Importance Score')
            #     ax.set_title(module+":Importance-based vs Traditional Weight Pruning")
            #     ax.legend()
            #     plt.tight_layout()

            #     os.makedirs("plots/experiments_weight", exist_ok=True)
            #     plt.savefig("plots/experiments_weight/histogram.png", dpi=300)
            #     plt.close()
            #     already_visualized=True
