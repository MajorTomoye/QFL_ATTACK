


import torch

def clip_weight_norm(global_model,param_clip_thres,logger,epoch):

    # 计算全局模型的范数
    squared_sum = 0
    for name, param in global_model.named_parameters():
        squared_sum += torch.sum(param.data.pow(2))  # 计算所有参数的平方和
    total_norm = torch.sqrt(squared_sum)  # 全局范数
    logger.info(f"Global Epoch: {epoch} ,Total norm before clipping: {total_norm}, Clip threshold: {param_clip_thres}")

    # 如果超过裁剪阈值，执行裁剪
    if total_norm > param_clip_thres:
        clip_coef = param_clip_thres / (total_norm + 1e-6)  # 避免除零错误
        for name, param in global_model.named_parameters():
            param.data *= clip_coef  # 缩放参数
        # 重新计算范数以验证裁剪效果
        squared_sum = 0
        for name, param in global_model.named_parameters():
            squared_sum += torch.sum(param.data.pow(2))
        current_norm = torch.sqrt(squared_sum)
        logger.info(f"Global Epoch: {epoch} ,Total norm after clipping: {current_norm}")
    else:
        logger.info(f"Global Epoch: {epoch} ,No clipping needed for this round.")
        current_norm = total_norm

    return current_norm