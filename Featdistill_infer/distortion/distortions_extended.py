"""
扩展图像退化算法库
Extended Image Distortion and Degradation Functions

包含除基础退化外的额外图像退化算法，用于数据增强和图像恢复任务。
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import kornia
from typing import Tuple, Optional, Union


# ============================================================================
# 模糊类 (Blur)
# ============================================================================

def motion_blur(x: torch.Tensor, kernel_size: int = 15, angle: float = 0.0) -> torch.Tensor:
    """
    运动模糊 - 模拟相机或物体移动产生的拖影

    Args:
        x: 输入图像 (C, H, W)
        kernel_size: 模糊核大小
        angle: 运动角度（度）

    Returns:
        模糊后的图像
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    # 创建运动模糊核
    kernel = torch.zeros((kernel_size, kernel_size), device=x.device)

    # 转换角度为弧度
    angle_rad = angle * math.pi / 180.0

    # 计算中心点
    center = (kernel_size - 1) / 2.0

    # 创建直线核
    for i in range(kernel_size):
        offset = i - center
        x_pos = center + offset * math.cos(angle_rad)
        y_pos = center - offset * math.sin(angle_rad)

        # 四舍五入到最近的整数坐标
        x_idx = int(round(x_pos))
        y_idx = int(round(y_pos))

        if 0 <= x_idx < kernel_size and 0 <= y_idx < kernel_size:
            kernel[y_idx, x_idx] = 1.0

    # 归一化
    kernel = kernel / kernel.sum()

    # 应用卷积
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    padding = kernel_size // 2
    y = F.conv2d(x, kernel, padding=padding, groups=x.size(1))

    return y.squeeze(0) if len(x.shape) == 4 else y


def defocus_blur(x: torch.Tensor, radius: int = 3, aperture: str = 'circular') -> torch.Tensor:
    """
    散焦模糊 - 模拟镜头失焦效果

    Args:
        x: 输入图像 (C, H, W)
        radius: 模糊半径
        aperture: 光圈形状 ('circular', 'hexagonal', 'pentagonal')

    Returns:
        模糊后的图像
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    # 创建散焦核
    kernel_size = 2 * radius + 1
    kernel = torch.zeros((kernel_size, kernel_size), device=x.device, dtype=torch.float32)

    y_idx, x_idx = torch.meshgrid(
        torch.arange(kernel_size, device=x.device, dtype=torch.float32),
        torch.arange(kernel_size, device=x.device, dtype=torch.float32),
        indexing='ij'
    )

    center = radius
    dist = torch.sqrt((x_idx - center) ** 2 + (y_idx - center) ** 2)

    if aperture == 'circular':
        kernel = (dist <= radius).float()
    elif aperture == 'hexagonal':
        # 六边形近似
        kernel = (dist <= radius * 1.1).float()
        angle = torch.atan2(y_idx - center, x_idx - center)
        mask = torch.abs(torch.sin(3 * angle)) < 0.866
        kernel = kernel * mask.float()
    elif aperture == 'pentagonal':
        # 五边形近似
        kernel = (dist <= radius * 1.15).float()
        angle = torch.atan2(y_idx - center, x_idx - center)
        mask = torch.abs(torch.cos(5 * angle / 2)) > 0.309
        kernel = kernel * mask.float()
    else:
        raise ValueError(f"Unknown aperture type: {aperture}")

    # 归一化
    kernel = kernel / kernel.sum()

    # 应用卷积
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(x.size(1), 1, 1, 1)
    padding = radius
    y = F.conv2d(x, kernel, padding=padding, groups=x.size(1))

    return y.squeeze(0) if len(x.shape) == 4 else y


def atmospheric_blur(x: torch.Tensor, severity: float = 1.0) -> torch.Tensor:
    """
    大气湍流模糊 - 模拟雾、热浪等大气介质引起的模糊

    Args:
        x: 输入图像 (C, H, W)
        severity: 严重程度 (0.1-5.0)

    Returns:
        模糊后的图像
    """
    # 使用高斯模糊模拟大气散射
    from .distortions import gaussian_blur
    return gaussian_blur(x, blur_sigma=severity)


def glass_blur(x: torch.Tensor, severity: float = 0.5, window_size: int = 7) -> torch.Tensor:
    """
    玻璃折射模糊 - 模拟透过玻璃看到的扭曲效果

    Args:
        x: 输入图像 (C, H, W)
        severity: 扭曲程度
        window_size: 采样窗口大小

    Returns:
        扭曲后的图像
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    C, H, W = x.shape[1], x.shape[2], x.shape[3]

    # 生成随机位移场
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing='ij'
    )

    # 添加随机扰动
    noise_y = torch.randn(H, W, device=x.device) * severity * window_size
    noise_x = torch.randn(H, W, device=x.device) * severity * window_size

    # 归一化坐标
    y_coords = (y_coords + noise_y).clamp(0, H - 1)
    x_coords = (x_coords + noise_x).clamp(0, W - 1)

    # 双线性插值采样
    y_coords = y_coords.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
    x_coords = x_coords.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)

    y = torch.nn.functional.grid_sample(x, torch.stack([x_coords, y_coords], dim=-1),
                                         mode='bilinear', padding_mode='border',
                                         align_corners=False)

    return y.squeeze(0) if len(x.shape) == 4 else y


# ============================================================================
# 噪声类 (Noise)
# ============================================================================

def poisson_noise(x: torch.Tensor, peak: float = 1.0) -> torch.Tensor:
    """
    泊松噪声（光子散粒噪声）- 模拟低光条件下的传感器噪声

    Args:
        x: 输入图像 (C, H, W), 值范围 [0, 1]
        peak: 峰值光子数

    Returns:
        加噪后的图像
    """
    # 缩放到光子计数
    x_scaled = x * peak

    # 生成泊松噪声
    noisy = torch.poisson(x_scaled)

    # 缩放回 [0, 1]
    y = noisy / peak

    # 裁剪
    y = torch.clamp(y, 0, 1)

    return y


def speckle_noise(x: torch.Tensor, var: float = 0.1) -> torch.Tensor:
    """
    斑点噪声 - 常见于医学超声和雷达图像

    Args:
        x: 输入图像 (C, H, W)
        var: 噪声方差

    Returns:
        加噪后的图像
    """
    noise = torch.randn_like(x) * math.sqrt(var)

    # 乘性噪声
    y = x + x * noise

    # 裁剪
    y = torch.clamp(y, 0, 1)

    return y


def multiplicative_noise(x: torch.Tensor, var: float = 0.1) -> torch.Tensor:
    """
    乘性噪声

    Args:
        x: 输入图像 (C, H, W)
        var: 噪声方差

    Returns:
        加噪后的图像
    """
    noise = torch.randn_like(x) * math.sqrt(var) + 1.0

    y = x * noise

    # 裁剪
    y = torch.clamp(y, 0, 1)

    return y


def shot_noise(x: torch.Tensor, amount: float = 0.1) -> torch.Tensor:
    """
    散粒噪声 - 模拟光电转换过程中的随机性

    Args:
        x: 输入图像 (C, H, W)
        amount: 噪声强度

    Returns:
        加噪后的图像
    """
    # 结合泊松和高斯噪声
    x_scaled = x * (1.0 / amount)
    poisson = torch.poisson(torch.clamp(x_scaled, 0, 1e6)) * amount

    # 添加少量高斯噪声
    gaussian = torch.randn_like(x) * amount * 0.1

    y = poisson + gaussian

    # 裁剪
    y = torch.clamp(y, 0, 1)

    return y


def fixed_pattern_noise(x: torch.Tensor, variance: float = 0.02) -> torch.Tensor:
    """
    固定模式噪声 - 传感器各像素响应不一致

    Args:
        x: 输入图像 (C, H, W)
        variance: 噪声方差

    Returns:
        加噪后的图像
    """
    C, H, W = x.shape

    # 生成固定的模式噪声（每个像素有固定的偏移）
    pattern = torch.randn(1, H, W, device=x.device, dtype=x.dtype) * variance

    # 应用到所有通道
    y = x + pattern

    # 裁剪
    y = torch.clamp(y, 0, 1)

    return y


def dead_pixels(x: torch.Tensor, density: float = 0.001) -> torch.Tensor:
    """
    坏点 - 模拟传感器坏像素

    Args:
        x: 输入图像 (C, H, W)
        density: 坏点密度

    Returns:
        有坏点的图像
    """
    y = x.clone()

    num_dead = int(density * x.shape[1] * x.shape[2])

    for _ in range(num_dead):
        ch = np.random.randint(0, x.shape[0])
        h = np.random.randint(0, x.shape[1])
        w = np.random.randint(0, x.shape[2])

        # 坏点要么是0要么是随机值
        if np.random.random() > 0.5:
            y[ch, h, w] = 0.0
        else:
            y[ch, h, w] = np.random.random() * 0.3

    return y


def iso_noise(x: torch.Tensor, iso_level: float = 800.0) -> torch.Tensor:
    """
    ISO噪声 - 模拟高ISO设置下的传感器噪声

    Args:
        x: 输入图像 (C, H, W)
        iso_level: ISO值（100-6400）

    Returns:
        加噪后的图像
    """
    # 噪声强度与ISO成正比
    base_var = 0.0001
    var = base_var * (iso_level / 100.0)

    # 信号相关噪声
    signal_var = x * var * 0.5

    # 总噪声
    noise = torch.randn_like(x) * torch.sqrt(signal_var + var)

    y = x + noise

    # 裁剪
    y = torch.clamp(y, 0, 1)

    return y


# ============================================================================
# 压缩/伪影类 (Compression Artifacts)
# ============================================================================

# def jpeg2000(x: torch.Tensor, quality: float = 0.7) -> torch.Tensor:
#     """
#     JPEG 2000压缩伪影（简化模拟）

#     Args:
#         x: 输入图像 (C, H, W)
#         quality: 质量 (0.1-1.0)

#     Returns:
#         压缩后的图像
#     """
#     # 简化模拟：使用块效应和高频损失
#     if len(x.shape) == 3:
#         x = x.unsqueeze(0)

#     # 小波变换模拟（简化为DCT + 量化）
#     # 这里使用频域滤波模拟压缩损失
#     y = x.clone()

#     # 简单的低通滤波模拟量化
#     kernel_size = int((1.0 - quality) * 7) + 1
#     if kernel_size > 1:
#         kernel = torch.ones(1, 1, kernel_size, kernel_size, device=x.device) / (kernel_size ** 2)
#         padding = kernel_size // 2

#         for c in range(y.size(1)):
#             channel = y[:, c:c+1, :, :]
#             y[:, c:c+1, :, :] = F.conv2d(channel, kernel, padding=padding)

#     return y.squeeze(0) if len(x.shape) == 4 else y


def compression_ringing(x: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
    """
    吉布斯环振铃效应 - 高频压缩产生的边缘振铃

    Args:
        x: 输入图像 (C, H, W)
        strength: 振铃强度

    Returns:
        有振铃效应的图像
    """
    was_unbatched = x.ndim == 3
    if was_unbatched:
        x = x.unsqueeze(0)

    # 检测边缘
    gray = kornia.color.rgb_to_grayscale(x)
    sobel = kornia.filters.sobel(gray)

    # 在边缘附近添加振铃
    kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)

    laplacian = F.conv2d(gray, kernel, padding=1)

    # 振铃效应
    ringing = laplacian * sobel.sign()

    # 添加到原图
    y = x + ringing * strength

    y = torch.clamp(y, 0, 1)

    return y.squeeze(0) if was_unbatched else y


def gif_quantization(x: torch.Tensor, num_colors: int = 64) -> torch.Tensor:
    """
    GIF色彩索引伪影 - 限制色彩数量的量化效果

    Args:
        x: 输入图像 (C, H, W)
        num_colors: 色彩数量（2-256）

    Returns:
        量化后的图像
    """
    # 简化：使用k-means聚类或均匀量化
    levels_per_channel = int(round(num_colors ** (1/3)))
    levels_per_channel = max(2, min(32, levels_per_channel))

    # 均匀量化
    y = torch.round(x * (levels_per_channel - 1)) / (levels_per_channel - 1)

    return torch.clamp(y, 0, 1)


def color_banding(x: torch.Tensor, bits: int = 5) -> torch.Tensor:
    """
    色阶断层 - 位深度不足导致的色带

    Args:
        x: 输入图像 (C, H, W)
        bits: 每通道位数 (1-8)

    Returns:
        有色带的图像
    """
    levels = 2 ** bits
    y = torch.round(x * (levels - 1)) / (levels - 1)

    return torch.clamp(y, 0, 1)


# ============================================================================
# 色彩类 (Color)
# ============================================================================

def color_cast(x: torch.Tensor, cast_color: Tuple[float, float, float] = (1.0, 0.9, 0.8)) -> torch.Tensor:
    """
    色偏 - 整体色调变化

    Args:
        x: 输入图像 (C, H, W) - RGB格式
        cast_color: 色偏系数 (R, G, B)

    Returns:
        有色偏的图像
    """
    cast_tensor = torch.tensor(cast_color, device=x.device, dtype=x.dtype).view(3, 1, 1)
    y = x * cast_tensor

    return torch.clamp(y, 0, 1)


def white_balance_distortion(x: torch.Tensor, temperature: float = 0.3) -> torch.Tensor:
    """
    白平衡失调 - 色温偏移

    Args:
        x: 输入图像 (C, H, W)
        temperature: 色温偏移 (-1.0 冷色 到 1.0 暖色)

    Returns:
        白平衡失调的图像
    """
    y = x.clone()

    if temperature > 0:
        # 暖色：增加红色，减少蓝色
        y[0] *= (1.0 + temperature * 0.3)
        y[2] *= (1.0 - temperature * 0.2)
    else:
        # 冷色：增加蓝色，减少红色
        temp = abs(temperature)
        y[0] *= (1.0 - temp * 0.2)
        y[2] *= (1.0 + temp * 0.3)

    return torch.clamp(y, 0, 1)


def chromatic_aberration_advanced(x: torch.Tensor, strength: float = 2.0) -> torch.Tensor:
    """
    高级色差 - 边缘RGB分离效果

    Args:
        x: 输入图像 (C, H, W)
        strength: 色差强度

    Returns:
        有色差的图像
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    C, H, W = x.shape[1], x.shape[2], x.shape[3]

    # 红色通道向左上偏移
    r_shift = int(strength / 2)
    r = x[:, 0:1, :, :]
    if r_shift > 0:
        r = r[:, :, r_shift:, r_shift:]

    # 蓝色通道向右下偏移
    b_shift = int(strength / 2)
    b = x[:, 2:3, :, :]
    if b_shift > 0:
        pad_h = H - b.shape[2]
        pad_w = W - b.shape[3]
        if pad_h > 0 and pad_w > 0:
            b = F.pad(b, (b_shift, b_shift, b_shift, b_shift), mode='constant')
            b = b[:, :, :H, :W]

    # 绿色通道保持不变
    g = x[:, 1:2, :, :]

    # 组合
    y = torch.cat([r, g, b], dim=1)

    # 裁剪到原始大小
    min_h = min(y.shape[2], H)
    min_w = min(y.shape[3], W)
    y = y[:, :, :min_h, :min_w]

    return y.squeeze(0) if len(x.shape) == 4 else y


def fading(x: torch.Tensor, amount: float = 0.3) -> torch.Tensor:
    """
    褪色/老化效果 - 降低饱和度和对比度

    Args:
        x: 输入图像 (C, H, W)
        amount: 褪色程度

    Returns:
        褪色的图像
    """
    # 转换到HSV降低饱和度
    x_rgb = x[[2, 1, 0], ...]  # RGB to BGR for kornia
    hsv = kornia.color.rgb_to_hsv(x_rgb)
    hsv[1] *= (1.0 - amount)

    # 降低对比度
    faded = kornia.color.hsv_to_rgb(hsv)
    faded = 0.5 + (faded - 0.5) * (1.0 - amount * 0.5)

    y = faded[[2, 1, 0], ...]  # BGR back to RGB

    return torch.clamp(y, 0, 1)


# ============================================================================
# 几何变形类 (Geometric Distortion)
# ============================================================================

def elastic_transform(x: torch.Tensor, alpha: float = 10.0, sigma: float = 5.0) -> torch.Tensor:
    """
    弹性形变

    Args:
        x: 输入图像 (C, H, W)
        alpha: 形变强度
        sigma: 高斯核标准差

    Returns:
        形变后的图像
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    C, H, W = x.shape[1], x.shape[2], x.shape[3]

    # 生成随机位移场
    dx = torch.randn(1, H, W, device=x.device) * alpha
    dy = torch.randn(1, H, W, device=x.device) * alpha

    # 高斯平滑位移场
    kernel_size = int(4 * sigma) + 1
    kernel = _gaussian_kernel(kernel_size, sigma).to(x.device)

    dx = F.conv2d(dx, kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2)
    dy = F.conv2d(dy, kernel.unsqueeze(0).unsqueeze(0), padding=kernel_size//2)

    # 创建网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32),
        torch.arange(W, device=x.device, dtype=torch.float32),
        indexing='ij'
    )

    # 应用位移
    x_coords = x_coords + dx.squeeze(0)
    y_coords = y_coords + dy.squeeze(0)

    # 归一化到 [-1, 1]
    x_coords = (x_coords / (W - 1)) * 2 - 1
    y_coords = (y_coords / (H - 1)) * 2 - 1

    grid = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0)

    # 采样
    y = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    return y.squeeze(0) if len(x.shape) == 4 else y


def perspective_warp(x: torch.Tensor, intensity: float = 0.1) -> torch.Tensor:
    """
    透视畸变

    Args:
        x: 输入图像 (C, H, W)
        intensity: 畸变强度

    Returns:
        畸变后的图像
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    C, H, W = x.shape[1], x.shape[2], x.shape[3]

    # 随机角点偏移
    offsets = torch.randn(4, 2, device=x.device) * intensity * min(H, W)

    # 原始角点
    src_corners = torch.tensor([
        [0, 0], [W-1, 0], [W-1, H-1], [0, H-1]
    ], device=x.device, dtype=torch.float32)

    # 目标角点
    dst_corners = src_corners + offsets

    # 计算透视变换矩阵
    src_h = torch.cat([src_corners, torch.ones(4, 1, device=x.device)], dim=1)
    dst_h = torch.cat([dst_corners, torch.ones(4, 1, device=x.device)], dim=1)

    # 简化：使用近似透视变换
    # 创建网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32),
        torch.arange(W, device=x.device, dtype=torch.float32),
        indexing='ij'
    )

    # 简单的透视效果
    center_x = W / 2
    center_y = H / 2

    dx = (x_coords - center_x) * intensity * (y_coords - center_y) / H
    dy = (y_coords - center_y) * intensity * (x_coords - center_x) / W

    x_coords = x_coords + dx
    y_coords = y_coords + dy

    # 归一化
    x_coords = (x_coords / (W - 1)) * 2 - 1
    y_coords = (y_coords / (H - 1)) * 2 - 1

    grid = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0)

    y = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    return y.squeeze(0) if len(x.shape) == 4 else y


def barrel_distortion(x: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
    """
    桶形畸变（鱼眼效果）

    Args:
        x: 输入图像 (C, H, W)
        strength: 畸变强度 (正数为桶形，负数为枕形)

    Returns:
        畸变后的图像
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    C, H, W = x.shape[1], x.shape[2], x.shape[3]

    # 创建网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32),
        torch.arange(W, device=x.device, dtype=torch.float32),
        indexing='ij'
    )

    # 归一化坐标
    x_norm = (x_coords - W / 2) / (W / 2)
    y_norm = (y_coords - H / 2) / (H / 2)

    # 计算径向距离
    r = torch.sqrt(x_norm ** 2 + y_norm ** 2)

    # 径向畸变
    r_distorted = r * (1 + strength * r ** 2)

    # 转换回笛卡尔坐标
    factor = torch.clamp(r_distorted / (r + 1e-6), 0.5, 2.0)
    x_distorted = (x_coords - W / 2) * factor + W / 2
    y_distorted = (y_coords - H / 2) * factor + H / 2

    # 归一化到 [-1, 1]
    x_distorted = (x_distorted / (W - 1)) * 2 - 1
    y_distorted = (y_distorted / (H - 1)) * 2 - 1

    grid = torch.stack([x_distorted, y_distorted], dim=-1).unsqueeze(0)

    y = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    return y.squeeze(0) if len(x.shape) == 4 else y


def pincushion_distortion(x: torch.Tensor, strength: float = 0.3) -> torch.Tensor:
    """
    枕形畸变

    Args:
        x: 输入图像 (C, H, W)
        strength: 畸变强度

    Returns:
        畸变后的图像
    """
    # 枕形畸变就是负强度的桶形畸变
    return barrel_distortion(x, strength=-strength)


def rotate_scale(x: torch.Tensor, angle: float = 15.0, scale: float = 1.0) -> torch.Tensor:
    """
    旋转缩放

    Args:
        x: 输入图像 (C, H, W)
        angle: 旋转角度（度）
        scale: 缩放因子

    Returns:
        变换后的图像
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    C, H, W = x.shape[1], x.shape[2], x.shape[3]

    # 创建网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32),
        torch.arange(W, device=x.device, dtype=torch.float32),
        indexing='ij'
    )

    # 归一化到 [-1, 1]
    x_norm = (x_coords / (W - 1)) * 2 - 1
    y_norm = (y_coords / (H - 1)) * 2 - 1

    # 缩放
    x_norm = x_norm / scale
    y_norm = y_norm / scale

    # 旋转
    angle_rad = angle * math.pi / 180.0
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    x_rot = x_norm * cos_a + y_norm * sin_a
    y_rot = -x_norm * sin_a + y_norm * cos_a

    grid = torch.stack([x_rot, y_rot], dim=-1).unsqueeze(0)

    y = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    return y.squeeze(0) if len(x.shape) == 4 else y


def shear_transform(x: torch.Tensor, shear_x: float = 0.1, shear_y: float = 0.0) -> torch.Tensor:
    """
    剪切变形

    Args:
        x: 输入图像 (C, H, W)
        shear_x: X方向剪切量
        shear_y: Y方向剪切量

    Returns:
        剪切后的图像
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    # 创建网格
    C, H, W = x.shape[1], x.shape[2], x.shape[3]

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32),
        torch.arange(W, device=x.device, dtype=torch.float32),
        indexing='ij'
    )

    # 归一化到 [-1, 1]
    x_norm = (x_coords / (W - 1)) * 2 - 1
    y_norm = (y_coords / (H - 1)) * 2 - 1

    # 剪切变换
    x_shear = x_norm + shear_x * y_norm
    y_shear = y_norm + shear_y * x_norm

    grid = torch.stack([x_shear, y_shear], dim=-1).unsqueeze(0)

    y = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)

    return y.squeeze(0) if len(x.shape) == 4 else y


# ============================================================================
# 天气/环境类 (Weather/Environment)
# ============================================================================

def fog_haze(x: torch.Tensor, density: float = 0.5) -> torch.Tensor:
    """
    雾天效果

    Args:
        x: 输入图像 (C, H, W)
        density: 雾的密度

    Returns:
        有雾的图像
    """
    # 创建雾层（灰白色）
    fog_color = torch.tensor([0.8, 0.85, 0.9], device=x.device, dtype=x.dtype).view(3, 1, 1)

    # 计算深度图（简化：底部更清晰）
    H = x.shape[1]
    depth = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype).view(1, H, 1)

    # 混合
    transmission = 1.0 - density * depth
    transmission = transmission.clamp(0.1, 1.0)

    y = x * transmission + fog_color * (1 - transmission)

    return torch.clamp(y, 0, 1)


def rain(x: torch.Tensor, intensity: float = 0.3, drop_length: int = 15) -> torch.Tensor:
    """
    雨滴效果

    Args:
        x: 输入图像 (C, H, W)
        intensity: 雨强度
        drop_length: 雨滴长度

    Returns:
        有雨的图像
    """
    y = x.clone()
    C, H, W = y.shape

    # 雨滴数量
    num_drops = int(intensity * H * W * 0.01)

    for _ in range(num_drops):
        # 随机位置
        start_x = np.random.randint(0, W)
        start_y = np.random.randint(0, H - drop_length)

        # 雨滴角度（略倾斜）
        for i in range(drop_length):
            x_pos = start_x + i // 2
            y_pos = start_y + i

            if 0 <= x_pos < W and 0 <= y_pos < H:
                # 降低亮度模拟雨滴
                y[:, y_pos, x_pos] *= 0.7

    return torch.clamp(y, 0, 1)


def snow(x: torch.Tensor, intensity: float = 0.2) -> torch.Tensor:
    """
    雪花效果

    Args:
        x: 输入图像 (C, H, W)
        intensity: 雪强度

    Returns:
        有雪的图像
    """
    y = x.clone()
    C, H, W = y.shape

    # 雪花数量
    num_flakes = int(intensity * H * W * 0.05)

    for _ in range(num_flakes):
        # 随机位置
        x_pos = np.random.randint(0, W)
        y_pos = np.random.randint(0, H)

        # 添加白色雪花
        size = np.random.randint(1, 3)
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                nx, ny = x_pos + dx, y_pos + dy
                if 0 <= nx < W and 0 <= ny < H:
                    y[:, ny, nx] = torch.maximum(y[:, ny, nx], torch.tensor(0.9, device=x.device))

    return torch.clamp(y, 0, 1)


def lens_flare(x: torch.Tensor, position: Tuple[float, float] = (0.3, 0.3),
               intensity: float = 0.5) -> torch.Tensor:
    """
    镜头光晕效果

    Args:
        x: 输入图像 (C, H, W)
        position: 光晕中心位置 (0-1, 0-1)
        intensity: 光晕强度

    Returns:
        有光晕的图像
    """
    y = x.clone()
    H, W = y.shape[1], y.shape[2]

    # 光晕中心
    cx, cy = int(position[0] * W), int(position[1] * H)

    # 创建光晕
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing='ij'
    )

    dist = torch.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

    # 多个光圈
    flare_layers = [
        (1.0, 0.15 * max(H, W)),  # 主光晕
        (0.5, 0.08 * max(H, W)),  # 次光晕
        (0.3, 0.05 * max(H, W)),  # 小光晕
    ]

    for alpha, radius in flare_layers:
        flare = alpha * torch.exp(-dist / radius) * intensity
        flare_color = torch.stack([flare, flare * 0.9, flare * 0.8])
        y = torch.maximum(y, flare_color)

    return torch.clamp(y, 0, 1)


def vignetting(x: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """
    暗角效果 - 边缘变暗

    Args:
        x: 输入图像 (C, H, W)
        strength: 暗角强度

    Returns:
        有暗角的图像
    """
    H, W = x.shape[1], x.shape[2]

    # 创建径向渐变
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32),
        torch.arange(W, device=x.device, dtype=torch.float32),
        indexing='ij'
    )

    # 归一化坐标
    x_norm = (x_coords - W / 2) / (W / 2)
    y_norm = (y_coords - H / 2) / (H / 2)

    # 径向距离
    r = torch.sqrt(x_norm ** 2 + y_norm ** 2)

    # 暗角系数（中心为1，边缘降低）
    vignette_factor = 1 - strength * r ** 2
    vignette_factor = vignette_factor.clamp(0.3, 1.0)

    y = x * vignette_factor

    return torch.clamp(y, 0, 1)


# ============================================================================
# 传感器相关 (Sensor-related)
# ============================================================================

def blooming(x: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
    """
    过饱和溢出 - 高亮区域溢出到相邻像素

    Args:
        x: 输入图像 (C, H, W)
        threshold: 溢出阈值

    Returns:
        有溢出的图像
    """
    y = x.clone()

    # 找到过饱和像素
    saturated = x > threshold

    # 简单的溢出效果：扩散到相邻像素
    for c in range(y.shape[0]):
        channel = y[c]
        sat_mask = saturated[c].float()

        # 使用简单卷积扩散
        kernel = torch.tensor([
            [0.1, 0.1, 0.1],
            [0.1, 1.0, 0.1],
            [0.1, 0.1, 0.1]
        ], device=x.device).unsqueeze(0).unsqueeze(0)

        overflow = F.conv2d(sat_mask.unsqueeze(0).unsqueeze(0), kernel, padding=1)
        overflow = overflow.squeeze() * 0.1

        # 添加溢出
        channel = channel + overflow
        y[c] = channel

    return torch.clamp(y, 0, 1)


# ============================================================================
# 其他 (Others)
# ============================================================================

def occlusion(x: torch.Tensor, num_patches: int = 3, size_range: Tuple[int, int] = (10, 50)) -> torch.Tensor:
    """
    遮挡效果 - 随机遮挡区域

    Args:
        x: 输入图像 (C, H, W)
        num_patches: 遮挡块数量
        size_range: 遮挡块大小范围

    Returns:
        有遮挡的图像
    """
    y = x.clone()
    C, H, W = y.shape

    for _ in range(num_patches):
        h_size = np.random.randint(*size_range)
        w_size = np.random.randint(*size_range)

        x_start = np.random.randint(0, W - w_size)
        y_start = np.random.randint(0, H - h_size)

        # 用黑色或灰色遮挡
        occluder = np.random.random() * 0.2
        y[:, y_start:y_start+h_size, x_start:x_start+w_size] = occluder

    return y


def glare(x: torch.Tensor, position: Tuple[float, float] = (0.5, 0.5),
          intensity: float = 0.6) -> torch.Tensor:
    """
    眩光效果

    Args:
        x: 输入图像 (C, H, W)
        position: 眩光中心
        intensity: 眩光强度

    Returns:
        有眩光的图像
    """
    y = x.clone()
    H, W = y.shape[1], y.shape[2]

    cx, cy = int(position[0] * W), int(position[1] * H)

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing='ij'
    )

    dist = torch.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

    # 眩光光束
    for angle in np.linspace(0, 2*math.pi, 8):
        dx = math.cos(angle)
        dy = math.sin(angle)

        # 沿角度的方向衰减
        dot = (x_coords - cx) * dx + (y_coords - cy) * dy
        cross = abs(-(y_coords - cy) * dx + (x_coords - cx) * dy)

        beam = torch.exp(-cross / 20.0) * torch.exp(-torch.abs(dot) / 100.0)
        beam = beam * intensity

        flare_color = torch.stack([beam, beam * 0.95, beam * 0.9])
        y = torch.maximum(y, flare_color)

    return torch.clamp(y, 0, 1)


def halftone(x: torch.Tensor, dot_size: int = 4) -> torch.Tensor:
    """
    半调印刷效果

    Args:
        x: 输入图像 (C, H, W)
        dot_size: 网点大小

    Returns:
        半调效果的图像
    """
    C, H, W = x.shape

    # 缩小再放大模拟网点
    small_h, small_w = H // dot_size, W // dot_size

    # 使用池化缩小
    if len(x.shape) == 3:
        x_small = F.adaptive_avg_pool2d(x.unsqueeze(0), (small_h, small_w)).squeeze(0)
    else:
        x_small = F.adaptive_avg_pool2d(x, (small_h, small_w))

    # 最近邻插值放大
    y = F.interpolate(x_small.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)

    return y


def pixelate(x: torch.Tensor, block_size: int = 10) -> torch.Tensor:
    """
    像素化（马赛克）效果

    Args:
        x: 输入图像 (C, H, W)
        block_size: 像素块大小

    Returns:
        像素化的图像
    """
    C, H, W = x.shape

    # 缩小
    small_h, small_w = max(1, H // block_size), max(1, W // block_size)

    if len(x.shape) == 3:
        x_small = F.adaptive_avg_pool2d(x.unsqueeze(0), (small_h, small_w)).squeeze(0)
    else:
        x_small = F.adaptive_avg_pool2d(x, (small_h, small_w))

    # 最近邻放大
    y = F.interpolate(x_small.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)

    return y


# ============================================================================
# 辅助函数
# ============================================================================

def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """生成高斯核"""
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    x = x.unsqueeze(0) * x.unsqueeze(1)
    kernel = torch.exp(-x / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel
