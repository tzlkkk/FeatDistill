import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as v2

logger = logging.getLogger("DegradationUtils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    # A library import must not create files in the source tree. Applications
    # can attach their own handler when per-image degradation logs are wanted.
    logger.addHandler(logging.NullHandler())

# ==========================================
# 独立退化函数库 (基础画质类)
# ==========================================

def apply_blur(x: torch.Tensor, intensity: float) -> torch.Tensor:
    kernel_size = int(intensity * 10)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    kernel_size = max(3, kernel_size)
    sigma = 0.1 + intensity * 2.0
    return v2.functional.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

def apply_noise(x: torch.Tensor, intensity: float) -> torch.Tensor:
    noise_std = intensity * 0.2
    noise = torch.randn_like(x) * noise_std
    return torch.clamp(x + noise, 0.0, 1.0)

def apply_jitter(x: torch.Tensor, intensity: float) -> torch.Tensor:
    factor = intensity * 0.8
    jitter = v2.ColorJitter(brightness=factor, contrast=factor, saturation=factor)
    return jitter(x)

def apply_resample(x: torch.Tensor, intensity: float) -> torch.Tensor:
    B, C, H, W = x.shape
    scale_factor = 1.0 - (intensity * 0.8)
    small_H, small_W = max(8, int(H * scale_factor)), max(8, int(W * scale_factor))
    out = F.interpolate(x, size=(small_H, small_W), mode='bilinear', align_corners=False)
    out = F.interpolate(out, size=(H, W), mode='bicubic', align_corners=False)
    return torch.clamp(out, 0.0, 1.0)

def apply_jpeg(x: torch.Tensor, intensity: float) -> torch.Tensor:
    B, C, H, W = x.shape
    bins = int(255 * (1.0 - intensity * 0.85))
    bins = max(8, bins)
    out = torch.round(x * bins) / bins
    
    scale = 1.0 - intensity * 0.7
    small_h, small_w = max(8, int(H * scale)), max(8, int(W * scale))
    out = F.interpolate(out, size=(small_h, small_w), mode='nearest')
    out = F.interpolate(out, size=(H, W), mode='nearest')
    return torch.clamp(out, 0.0, 1.0)

# ==========================================
# 🚀 独立退化函数库 (新增：Face Pipeline 管线与几何类)
# ==========================================

def apply_crop_shift(x: torch.Tensor, intensity: float) -> torch.Tensor:
    """最直接模拟 face bbox 检测不准导致的几何偏移"""
    B, C, H, W = x.shape
    # 最大偏移量设为宽高的 15%
    max_dx = int(intensity * 0.15 * W)
    max_dy = int(intensity * 0.15 * H)
    if max_dx == 0 and max_dy == 0:
        return x

    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)

    # 核心技巧：使用反射填充(reflection padding)避免产生黑边，更符合真实图像
    pad_left = abs(dx) if dx > 0 else 0
    pad_right = abs(dx) if dx < 0 else 0
    pad_top = abs(dy) if dy > 0 else 0
    pad_bottom = abs(dy) if dy < 0 else 0

    out = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

    # 裁剪回原始尺寸
    start_x = pad_right if dx < 0 else 0
    start_y = pad_bottom if dy < 0 else 0
    out = out[:, :, start_y:start_y+H, start_x:start_x+W]
    return out

def apply_crop_scale(x: torch.Tensor, intensity: float) -> torch.Tensor:
    """最直接模拟不同人脸裁剪器截取的范围差异 (人脸占比差异)"""
    B, C, H, W = x.shape
    # 尺度变化范围，例如 intensity 为 1.0 时，尺度缩放在 0.8(缩小) 到 1.25(放大) 之间
    scale_factor = random.uniform(1.0 - 0.2 * intensity, 1.0 + 0.25 * intensity)

    if abs(scale_factor - 1.0) < 1e-3:
        return x

    new_H, new_W = int(H * scale_factor), int(W * scale_factor)

    if scale_factor > 1.0:
        # Zoom in: 放大并截取中心
        crop_y = (H - int(H / scale_factor)) // 2
        crop_x = (W - int(W / scale_factor)) // 2
        crop_h = H - 2 * crop_y
        crop_w = W - 2 * crop_x
        out = x[:, :, crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
    else:
        # Zoom out: 缩小并用反射边缘进行填补
        out = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
        pad_h = H - new_H
        pad_w = W - new_W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        out = F.pad(out, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

    return torch.clamp(out, 0.0, 1.0)

def apply_reencode_chain(x: torch.Tensor, intensity: float) -> torch.Tensor:
    """模拟真实的社交软件/传输流：复杂的重新编码破坏链"""
    out = x.clone()
    chain_type = random.choice(['resize_jpeg', 'jpeg_resize', 'blur_jpeg'])
    
    if chain_type == 'resize_jpeg':
        out = apply_resample(out, intensity * 0.8) # 削弱单一操作以防彻底损毁
        out = apply_jpeg(out, intensity * 0.9)
    elif chain_type == 'jpeg_resize':
        out = apply_jpeg(out, intensity * 0.9)
        out = apply_resample(out, intensity * 0.8)
    elif chain_type == 'blur_jpeg':
        out = apply_blur(out, intensity * 0.6)
        out = apply_jpeg(out, intensity * 0.9)
        
    return out

# ==========================================
# 统一调用接口
# ==========================================

DEGRADATION_REGISTRY = {
    'blur': apply_blur,
    'noise': apply_noise,
    'jitter': apply_jitter,
    'resample': apply_resample,
    'jpeg': apply_jpeg,
    'crop_shift': apply_crop_shift,     # 🌟 新增：位移
    'crop_scale': apply_crop_scale,     # 🌟 新增：尺度
    'reencode_chain': apply_reencode_chain # 🌟 新增：重编码链
}

# 🌟 img_info 参数保持不变，继续完美服务于日志
def apply_custom_degradation(x: torch.Tensor, mode: str = 'random', intensity: float = 0.5, iterations: int = 1, img_info: str = "Unknown_Image") -> torch.Tensor:
    if intensity <= 0.0 or iterations < 1:
        return x

    available_modes = list(DEGRADATION_REGISTRY.keys())
    target_mode = random.choice(available_modes) if mode == 'random' else mode
    
    if target_mode not in DEGRADATION_REGISTRY:
        raise ValueError(f"不支持的退化模式 '{mode}'。")

    out = x.clone()
    func = DEGRADATION_REGISTRY[target_mode]
    
    # 写入日志文件
    logger.info(f"🎨 [Degradation] Image: {img_info} | Mode: {target_mode.upper()} | Intensity: {intensity:.2f} | Iterations: {iterations}")

    for _ in range(iterations):
        out = func(out, intensity)
        
    return out

class CustomDegradationModule(nn.Module):
    def __init__(self, mode='random', iterations=1):
        super().__init__()
        self.mode = mode
        self.iterations = iterations

    def forward(self, x, mode=None, intensity=0.5, iterations=None, img_info="Unknown_Image"):
        _mode = mode if mode is not None else self.mode
        _iters = iterations if iterations is not None else self.iterations
        return apply_custom_degradation(x, mode=_mode, intensity=intensity, iterations=_iters, img_info=img_info)
