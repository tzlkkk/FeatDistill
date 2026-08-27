import torch
import random
import numpy as np
from .distortions_extra import (
    digital_glitch, ghosting, night_vision, water_drops, salt_and_pepper,
    pixel_shuffle_degradation, light_leak, film_grain, color_bleed, jpeg_blockiness
)
# Optional operators from .distortions_extended:
#     motion_blur, defocus_blur, poisson_noise, speckle_noise, iso_noise,
#     jpeg2000, compression_ringing, gif_quantization, color_banding, fading
# )
from .distortions_extended import (
    motion_blur, defocus_blur, poisson_noise, speckle_noise, iso_noise,
    compression_ringing, gif_quantization, color_banding, fading
)

def _get_param(min_val, max_val, level, max_levels, is_int=False):
    """
    根据给定的严重等级(level)线性插值计算对应的参数值。
    level 范围: 1 到 max_levels
    """
    ratio = (level - 1) / max(1, max_levels - 1)
    val = min_val + ratio * (max_val - min_val)
    return int(round(val)) if is_int else val

# ============================================================================
# 退化操作注册表 (Registry)
# 将算法分为不同的组别，方便你针对 Deepfake 任务进行开关控制
# 字典结构: "操作名称": lambda img, level, max_levels: 具体的函数调用
# ============================================================================

DISTORTION_REGISTRY = {
    # ---------------- 1. 压缩与数字伪影 (对 Deepfake 极其重要) ----------------
    'jpeg_blockiness': lambda x, l, m: jpeg_blockiness(
        x, quality=_get_param(40, 5, l, m, is_int=True) # quality越低越严重
    ),
    # 'jpeg2000': lambda x, l, m: jpeg2000(
    #     x, quality=_get_param(0.8, 0.1, l, m)
    # ),
    'compression_ringing': lambda x, l, m: compression_ringing(
        x, strength=_get_param(0.05, 0.3, l, m)
    ),
    'gif_quantization': lambda x, l, m: gif_quantization(
        x, num_colors=_get_param(128, 16, l, m, is_int=True)
    ),
    'color_banding': lambda x, l, m: color_banding(
        x, bits=_get_param(7, 3, l, m, is_int=True) # bits越低色带越严重
    ),
    'pixel_shuffle': lambda x, l, m: pixel_shuffle_degradation(
        x, intensity=_get_param(0.01, 0.1, l, m), block_size=2
    ),

    # ---------------- 2. 模糊类 (破坏高频特征) ----------------
    'motion_blur': lambda x, l, m: motion_blur(
        x, 
        kernel_size=_get_param(5, 21, l, m, is_int=True) | 1, # 确保是奇数
        angle=random.uniform(0, 360)
    ),
    'defocus_blur': lambda x, l, m: defocus_blur(
        x, radius=_get_param(1, 5, l, m, is_int=True)
    ),

    # ---------------- 3. 噪声类 (干扰空域特征) ----------------
    'poisson_noise': lambda x, l, m: poisson_noise(
        x, peak=_get_param(100.0, 10.0, l, m) # peak越小噪声越明显
    ),
    'speckle_noise': lambda x, l, m: speckle_noise(
        x, var=_get_param(0.01, 0.15, l, m)
    ),
    'iso_noise': lambda x, l, m: iso_noise(
        x, iso_level=_get_param(400, 6400, l, m)
    ),
    'film_grain': lambda x, l, m: film_grain(
        x, intensity=_get_param(0.02, 0.15, l, m)
    ),
    'salt_and_pepper': lambda x, l, m: salt_and_pepper(
        x, prob=_get_param(0.005, 0.05, l, m)
    ),

    # ---------------- 4. 视效与光影变换 (改变颜色分布) ----------------
    'color_bleed': lambda x, l, m: color_bleed(
        x, offset=_get_param(1, 6, l, m, is_int=True)
    ),
    'light_leak': lambda x, l, m: light_leak(
        x, amount=_get_param(0.1, 0.6, l, m)
    ),
    'fading': lambda x, l, m: fading(
        x, amount=_get_param(0.1, 0.6, l, m)
    ),
    'digital_glitch': lambda x, l, m: digital_glitch(
        x, amount=_get_param(0.05, 0.25, l, m)
    ),

    # ---------------- 5. 极端物理与环境干扰 (In the Wild 场景必备) ----------------
    'water_drops': lambda x, l, m: water_drops(
        x, amount=_get_param(0.1, 0.8, l, m) # 从少量水滴到密布水滴
    ),
    'night_vision': lambda x, l, m: night_vision(
        x, intensity=_get_param(0.2, 1.0, l, m) # 绿光和噪点强度
    ),
    'ghosting': lambda x, l, m: ghosting(
        x, 
        offset_x=_get_param(2, 15, l, m, is_int=True),
        intensity=_get_param(0.1, 0.4, l, m)
    ),
}



def distort_images_extra(
    pixel_values: torch.Tensor, 
    max_distortions: int = 4, 
    num_levels: int = 10,
    p: float = 1.0,
    excluded_ops: set[str] | None = None,
):
    """
    随机应用多重退化操作的流水线。

    Args:
        pixel_values: 输入的图像 Tensor, 形状 (C, H, W), 值域 [0, 1]
        max_distortions: 最大应用的退化操作数量
        num_levels: 将严重程度分为几个等级 (例如 5 级)
        p: 触发流水线的整体概率

    Returns:
        tuple: (处理后的图像, 应用的操作列表, 对应的等级列表)
    """
    applied_ops = []
    applied_levels = []
    
    # 整体概率控制 (如果没触发，直接返回原图)
    if random.random() > p:
        return pixel_values, applied_ops, applied_levels

    # 确定本次应用几个操作 (1 到 max_distortions)
    num_ops_to_apply = random.randint(1, max_distortions)
    
    # 从注册表中随机抽取不重复的操作
    excluded_ops = excluded_ops or set()
    available_ops = [name for name in DISTORTION_REGISTRY if name not in excluded_ops]
    if not available_ops:
        raise ValueError("all extended distortions were excluded")
    num_ops_to_apply = min(num_ops_to_apply, len(available_ops))
    selected_ops = random.sample(available_ops, num_ops_to_apply)

    # 依次应用
    for op_name in selected_ops:
        # 随机选择一个严重等级 (1 到 num_levels)
        level = random.randint(1, num_levels)
        
        # 获取对应的操作函数
        op_func = DISTORTION_REGISTRY[op_name]
        
        try:
            # 应用退化
            pixel_values = op_func(pixel_values, level, num_levels)
            # 记录
            applied_ops.append(op_name)
            applied_levels.append(level)
            
        except Exception as e:
            # 防止某个极端参数导致报错中断训练
            print(f"⚠️ 警告: 退化操作 {op_name} (Level: {level}) 发生错误: {e}")
            continue
            
    # 确保最终输出在 [0, 1] 之间
    pixel_values = torch.clamp(pixel_values, 0.0, 1.0)
    
    return pixel_values, applied_ops, applied_levels
