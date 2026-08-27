"""
额外的图像退化算法库 (Extra)
Extra Image Distortion/Degradation Functions

包含补充的数字/物理/特效等图像退化算法，用于扩展数据增强和图像恢复任务的功能。
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import kornia
from typing import Tuple, Optional, Union

# ============================================================================
# 传感器与物理镜头特效 (Sensor & Lens Effects)
# ============================================================================

def chromatic_noise(x: torch.Tensor, amount: float = 0.1) -> torch.Tensor:
    """
    彩色噪声 (Chroma / Color Noise) - 独立在颜色通道上的低频噪声

    Args:
        x: 输入图像 (C, H, W)，值范围 [0, 1]
        amount: 噪声强度

    Returns:
        加噪后的图像
    """
    if x.shape[0] != 3:
        return x  # 仅处理RGB图像
        
    y = x.clone()
    C, H, W = y.shape
    
    # 转换到 YCbCr 或类似空间，但为了简单起见，我们直接在 RGB 上注入低频噪声
    # 创建低频噪声（小尺寸生成后放大）
    noise_h, noise_w = max(1, H // 8), max(1, W // 8)
    noise = torch.randn(C, noise_h, noise_w, device=x.device) * amount
    
    # 上采样到原图大小来获得低频云雾块状的颜色噪声
    noise_up = F.interpolate(noise.unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False).squeeze(0)
    
    y = y + noise_up
    return torch.clamp(y, 0, 1)


def purple_fringing(x: torch.Tensor, strength: float = 0.5) -> torch.Tensor:
    """
    紫边效应 (Purple Fringing) - 高反差边缘处的轴向色差

    Args:
        x: 输入图像 (C, H, W)
        strength: 紫边强度

    Returns:
        包含紫边的图像
    """
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
        
    # 计算亮度
    gray = kornia.color.rgb_to_grayscale(x)
    
    # 找到高亮区域（过曝区域或高反差区域）
    high_light = torch.clamp((gray - 0.8) * 5.0, 0, 1)
    
    # 找到边缘区域
    edges = kornia.filters.sobel(gray)
    
    # 结合高亮和边缘：高反差的高亮边缘
    fringe_mask = high_light * edges
    
    # 模糊掩码以扩散紫边
    fringe_mask = kornia.filters.gaussian_blur2d(fringe_mask, (5, 5), (1.5, 1.5))
    
    # 紫色增强：R 和 B 通道增加
    purple_color = torch.tensor([1.0, 0.0, 1.0], device=x.device).view(1, 3, 1, 1)
    
    # 混合
    y = x + fringe_mask * purple_color * strength
    
    y = torch.clamp(y, 0, 1)
    return y.squeeze(0) if len(x.shape) == 4 else y


def lens_dirt(x: torch.Tensor, dirt_amount: float = 0.3) -> torch.Tensor:
    """
    镜头污迹 (Lens Dirt / Smudge) - 模拟镜头表面的污迹遮挡光线

    Args:
        x: 输入图像 (C, H, W)
        dirt_amount: 污迹浓度

    Returns:
        带有镜头污迹的图像
    """
    H, W = x.shape[-2:]
    dirt_map = torch.zeros((1, H, W), device=x.device)
    
    # 生成几个随机的低频污迹斑块
    num_spots = int(dirt_amount * 10) + 1
    for _ in range(num_spots):
        sh, sw = max(1, H // 16), max(1, W // 16)
        spot = torch.rand((1, sh, sw), device=x.device)
        spot = F.interpolate(spot.unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False).squeeze(0)
        
        # 随机位置遮罩
        cx, cy = np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=torch.float32) / H,
            torch.arange(W, device=x.device, dtype=torch.float32) / W,
            indexing='ij'
        )
        dist = torch.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        radius = np.random.uniform(0.1, 0.3)
        
        mask = torch.exp(- (dist**2) / (2 * (radius/2)**2)).unsqueeze(0)
        dirt_map += spot * mask
        
    dirt_map = torch.clamp(dirt_map * dirt_amount, 0, 0.8)
    
    # 污迹造成对比度下降和模糊，以及透光率下降
    blurred_x = kornia.filters.gaussian_blur2d(x.unsqueeze(0) if len(x.shape)==3 else x, (9, 9), (3.0, 3.0)).squeeze(0)
    
    # 混合原图与污迹图（使用稍亮/灰的污迹色模拟折射，或暗色模拟灰尘）
    dirt_color = torch.tensor([0.2, 0.2, 0.18], device=x.device).view(-1, 1, 1)
    if len(x.shape) == 4:
        dirt_color = dirt_color.unsqueeze(0)
        
    # 被污迹覆盖的地方透光差
    y = x * (1 - dirt_map) + blurred_x * dirt_map * 0.5 + dirt_color * dirt_map * 0.5
    
    return torch.clamp(y, 0, 1)


# ============================================================================
# 数字伪影特效 (Digital Artifacts)
# ============================================================================

def digital_glitch(x: torch.Tensor, amount: float = 0.2) -> torch.Tensor:
    """
    数字故障 (Glitch Art) - 随机切割、颜色通道偏移

    Args:
        x: 输入图像 (C, H, W)
        amount: 故障强度 

    Returns:
        包含故障效果的图像
    """
    y = x.clone()
    C, H, W = y.shape
    
    num_blocks = int(amount * 20)
    
    for _ in range(num_blocks):
        # 随机选取一个水平条块或矩形块
        block_h = np.random.randint(1, max(2, H // 10))
        y_start = np.random.randint(0, max(1, H - block_h))
        
        # 水平偏移量
        shift = np.random.randint(-int(W * amount), int(W * amount) + 1)
        if shift == 0:
            continue
            
        color_shift = np.random.choice([True, False])
        
        if color_shift and C >= 3:
            # 仅偏移某个颜色通道
            ch = np.random.randint(0, 3)
            temp = y[ch, y_start:y_start+block_h, :].clone()
            
            if shift > 0:
                y[ch, y_start:y_start+block_h, shift:] = temp[:, :-shift]
                y[ch, y_start:y_start+block_h, :shift] = temp[:, -shift:] # 环状移位
            else:
                y[ch, y_start:y_start+block_h, :shift] = temp[:, -shift:]
                y[ch, y_start:y_start+block_h, shift:] = temp[:, :-shift]
                
        else:
            # 整个图像区域水平错位
            temp = y[:, y_start:y_start+block_h, :].clone()
            if shift > 0:
                y[:, y_start:y_start+block_h, shift:] = temp[:, :, :-shift]
                y[:, y_start:y_start+block_h, :shift] = temp[:, :, -shift:]
            else:
                y[:, y_start:y_start+block_h, :shift] = temp[:, :, -shift:]
                y[:, y_start:y_start+block_h, shift:] = temp[:, :, :-shift]
                
    return y


def scanlines(x: torch.Tensor, intensity: float = 0.5, thickness: int = 1) -> torch.Tensor:
    """
    扫描线 / CRT 效果 (Scanlines)

    Args:
        x: 输入图像 (C, H, W)
        intensity: 扫描线强度（变暗的程度）
        thickness: 扫描线宽度

    Returns:
        带有扫描线效果的图像
    """
    y = x.clone()
    C, H, W = y.shape
    
    # 构建遮罩
    mask = torch.ones((1, H, W), device=x.device)
    
    # 按行间隔变暗
    for i in range(0, H, thickness * 2):
        end_idx = min(i + thickness, H)
        mask[:, i:end_idx, :] = 1.0 - intensity

    y = y * mask
    return y


def ghosting(x: torch.Tensor, offset_x: int = 10, offset_y: int = 0, intensity: float = 0.3) -> torch.Tensor:
    """
    重影/双重曝光 (Ghosting) - 信号反射或多重曝光造成的叠影
    
    Args:
        x: 输入图像 (C, H, W)
        offset_x: 重影水平偏移
        offset_y: 重影垂直偏移
        intensity: 重影强度

    Returns:
        有重影效果的图像
    """
    y = x.clone()
    C, H, W = y.shape
    
    ghost = torch.zeros_like(x)
    
    # 将 x 平移
    start_y = max(0, offset_y)
    start_x = max(0, offset_x)
    src_y = max(0, -offset_y)
    src_x = max(0, -offset_x)
    
    copy_h = H - abs(offset_y)
    copy_w = W - abs(offset_x)
    
    if copy_h > 0 and copy_w > 0:
        ghost[:, start_y:start_y+copy_h, start_x:start_x+copy_w] = x[:, src_y:src_y+copy_h, src_x:src_x+copy_w]
        
    y = y * (1.0 - intensity * 0.5) + ghost * intensity
    return torch.clamp(y, 0, 1)


# ============================================================================
# 曝光与光照渲染 (Exposure & Lighting)
# ============================================================================

def over_exposure(x: torch.Tensor, strength: float = 1.5) -> torch.Tensor:
    """
    过曝光 (Overexposure) - 模拟强光下传感器溢出损失细节

    Args:
        x: 输入图像 (C, H, W)
        strength: 曝光增强倍数 

    Returns:
        过曝的图像
    """
    # 非线性伽马曲线结合线性乘法映射
    y = x * strength
    
    # 使用软裁剪过渡(Soft Clipping)避免太生硬的边界
    # y = y / (1.0 + torch.exp(-10.0 * (y - 0.5))) ...
    
    # 模拟传感器全井容量溢出，简单裁剪：
    return torch.clamp(y, 0, 1)

def under_exposure(x: torch.Tensor, strength: float = 0.4) -> torch.Tensor:
    """
    欠曝光 (Underexposure) - 模拟曝光不足，并增加底噪

    Args:
        x: 输入图像 (C, H, W)
        strength: 曝光降低倍数 (如 0.4 表示只有 40% 亮度)

    Returns:
        欠曝且可能带有暗部噪点的图像
    """
    y = x * strength
    
    # 欠曝往往伴随信噪比显著降低（暗部噪点）
    noise = torch.randn_like(x) * 0.02
    y = y + noise
    
    return torch.clamp(y, 0, 1)

# ============================================================================
# 视效相关 (Visual Effects)
# ============================================================================

def night_vision(x: torch.Tensor, intensity: float = 1.0) -> torch.Tensor:
    """
    夜视仪效果 (Night Vision) - 绿色色调，高对比度低动态范围，并带噪点

    Args:
        x: 输入图像 (C, H, W)
        intensity: 夜视效果应用强度 (0~1)

    Returns:
        夜视仪效果图像
    """
    # 转换为灰度图
    gray = kornia.color.rgb_to_grayscale(x.unsqueeze(0) if len(x.shape)==3 else x)
    
    # 增强对比度 (直方图拉伸模拟)
    gray = torch.clamp((gray - 0.1) * 1.5, 0, 1)
    
    # 夜视仪绿色
    green_tint = torch.tensor([0.1, 0.9, 0.2], device=x.device).view(-1, 1, 1)
    if len(x.shape) == 4:
        green_tint = green_tint.unsqueeze(0)
        
    night = gray * green_tint
    
    # 加上夜视仪典型的底噪和扫描条纹
    noise = torch.randn_like(night) * 0.1
    night += noise
    
    # 轻微的扫描线
    H = night.shape[-2]
    for i in range(0, H, 2):
        night[..., i, :] *= 0.95
        
    night = torch.clamp(night, 0, 1)
    
    if len(x.shape) == 3:
        night = night.squeeze(0)
        
    return x * (1 - intensity) + night * intensity

def thermal_imaging(x: torch.Tensor) -> torch.Tensor:
    """
    热成像效果 (Thermal Imaging) - 将亮度和温度作为映射，转换为伪色彩图

    Args:
        x: 输入图像 (C, H, W)

    Returns:
        热成像伪彩图像
    """
    # 将亮度视为温度
    gray = kornia.color.rgb_to_grayscale(x.unsqueeze(0) if len(x.shape)==3 else x)
    
    # 热成像色彩映射：黑(冷) -> 蓝 -> 紫 -> 红 -> 黄 -> 白(热)
    # 最简单的转换，利用连续的非线性变换构建伪彩：
    r = torch.clamp(4.0 * gray - 1.5, 0.0, 1.0)
    g = torch.clamp(4.0 * gray - 0.5, 0.0, 1.0) - torch.clamp(4.0 * gray - 3.5, 0.0, 1.0)
    b = torch.clamp(2.0 * gray, 0.0, 1.0) - torch.clamp(4.0 * gray - 1.5, 0.0, 1.0) + torch.clamp(4.0 * gray - 3.5, 0.0, 1.0)
    
    thermal = torch.cat([r, g, b], dim=-3)
    
    if len(x.shape) == 3:
         thermal = thermal.squeeze(0)
         
    return torch.clamp(thermal, 0, 1)


# ============================================================================
# 其他自然与摄像机瑕疵 (Other Natural & Camera Flaws)
# ============================================================================

def water_drops(x: torch.Tensor, amount: float = 0.5) -> torch.Tensor:
    """
    水滴效果 (Water Drops) - 模拟镜头或玻璃表面的水滴折射遮挡

    Args:
        x: 输入图像 (C, H, W)
        amount: 水滴密集度

    Returns:
        包含水滴效果的图像
    """
    y = x.clone()
    C, H, W = y.shape
    
    num_drops = int(amount * 20) + 1
    
    for _ in range(num_drops):
        # 随机位置和大小
        radius = np.random.randint(max(2, H // 40), max(5, H // 15))
        cx = np.random.randint(radius, W - radius)
        cy = np.random.randint(radius, H - radius)
        
        # 创建水滴蒙版和折射映射
        y_coords, x_coords = torch.meshgrid(
            torch.arange(cy - radius, cy + radius + 1, device=x.device, dtype=torch.float32),
            torch.arange(cx - radius, cx + radius + 1, device=x.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # 确保不越界
        valid_mask = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W)
        y_coords = y_coords[valid_mask]
        x_coords = x_coords[valid_mask]
        
        if len(y_coords) == 0:
            continue
            
        dist = torch.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        drop_mask = dist <= radius
        
        y_coords_in = y_coords[drop_mask]
        x_coords_in = x_coords[drop_mask]
        dist_in = dist[drop_mask]
        
        if len(y_coords_in) == 0:
            continue
            
        # 抛物面折射效果模拟 (中心凸起)
        refract_factor = 1.0 - (dist_in / radius) ** 2
        
        # 采样点偏移 (鱼眼/球面镜效果)
        sample_x = cx + (x_coords_in - cx) * (0.3 + 0.7 * (1 - refract_factor))
        sample_y = cy + (y_coords_in - cy) * (0.3 + 0.7 * (1 - refract_factor))
        
        sample_x = torch.clamp(sample_x.long(), 0, W - 1)
        sample_y = torch.clamp(sample_y.long(), 0, H - 1)
        
        for c in range(C):
            y[c, y_coords_in.long(), x_coords_in.long()] = x[c, sample_y, sample_x]
            
        # 边缘加深/变亮模拟高光和阴影
        edge_mask = (dist_in > radius * 0.8) & (dist_in <= radius)
        if edge_mask.any():
            shade = 0.5 + 0.5 * (1.0 - refract_factor[edge_mask])
            for c in range(C):
                y[c, y_coords_in[edge_mask].long(), x_coords_in[edge_mask].long()] *= shade
                
        # 添加一点点高光
        highlight = (dist_in < radius * 0.2)
        if highlight.any():
            for c in range(C):
                y[c, y_coords_in[highlight].long(), x_coords_in[highlight].long()] = torch.clamp(
                    y[c, y_coords_in[highlight].long(), x_coords_in[highlight].long()] + 0.3, 0, 1)

    return y


def salt_and_pepper(x: torch.Tensor, prob: float = 0.02) -> torch.Tensor:
    """
    椒盐噪声 (Salt and Pepper Noise) - 随机的黑白（极值）像素点

    Args:
        x: 输入图像 (C, H, W)
        prob: 发生突变的概率，其中一半是黑点(椒)，一半是白点(盐)

    Returns:
        包含椒盐噪声的图像
    """
    y = x.clone()
    
    # 生成随机矩阵
    random_matrix = torch.rand(y.shape[-2:], device=y.device)
    
    # 盐 (White)
    salt_mask = random_matrix < (prob / 2)
    # 椒 (Black)
    pepper_mask = (random_matrix >= (prob / 2)) & (random_matrix < prob)
    
    if len(y.shape) == 3:
        salt_mask = salt_mask.unsqueeze(0).expand_as(y)
        pepper_mask = pepper_mask.unsqueeze(0).expand_as(y)
        
    y[salt_mask] = 1.0
    y[pepper_mask] = 0.0
        
    return y


def pixel_shuffle_degradation(x: torch.Tensor, intensity: float = 0.05, block_size: int = 2) -> torch.Tensor:
    """
    像素随机打乱 (Pixel Shuffle Degradation) - 局部小范围的像素位置错乱，模拟传输加密错误或极端压缩抖动

    Args:
        x: 输入图像 (C, H, W)
        intensity: 打乱的概率
        block_size: 打乱操作所在的局部块大小

    Returns:
        局部打乱后的图像
    """
    y = x.clone()
    C, H, W = y.shape
    
    # 转换为形如 (C, H//P, P, W//P, P) 的 view
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    
    if pad_h > 0 or pad_w > 0:
        y_padded = F.pad(y, (0, pad_w, 0, pad_h), mode='reflect')
    else:
        y_padded = y
        
    pH, pW = y_padded.shape[1], y_padded.shape[2]
    
    # 将图像分块
    y_blocks = y_padded.view(C, pH // block_size, block_size, pW // block_size, block_size)
    y_blocks = y_blocks.permute(0, 1, 3, 2, 4).reshape(C, -1, block_size * block_size)
    
    # 决定哪些块需要打乱
    num_blocks = y_blocks.shape[1]
    shuffle_mask = torch.rand(num_blocks, device=x.device) < intensity
    
    if shuffle_mask.any():
        # 对选中的块内部像素进行随机打乱
        affected_indices = torch.nonzero(shuffle_mask).squeeze(-1)
        
        for idx in affected_indices:
            perm = torch.randperm(block_size * block_size, device=x.device)
            y_blocks[:, idx, :] = y_blocks[:, idx, perm]
            
    # 还原形状
    y_blocks = y_blocks.view(C, pH // block_size, pW // block_size, block_size, block_size)
    y_blocks = y_blocks.permute(0, 1, 3, 2, 4).reshape(C, pH, pW)
    
    if pad_h > 0 or pad_w > 0:
        return y_blocks[:, :H, :W]
    return y_blocks


# ============================================================================
# 复古与模拟介质 (Retro & Analog Media)
# ============================================================================

def moire_pattern(x: torch.Tensor, frequency: float = 100.0, angle: float = 15.0) -> torch.Tensor:
    """
    摩尔纹 (Moiré Pattern) - 拍摄屏幕或细密纹理时产生的干涉条纹

    Args:
        x: 输入图像 (C, H, W)
        frequency: 干涉条纹的频率
        angle: 干涉网络角度

    Returns:
        包含摩尔纹的图像
    """
    y = x.clone()
    H, W = y.shape[-2:]
    
    # 构建坐标网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=x.device, dtype=torch.float32) / H,
        torch.arange(W, device=x.device, dtype=torch.float32) / W,
        indexing='ij'
    )
    
    # 旋转坐标来计算波纹
    angle_rad = angle * math.pi / 180.0
    x_rot = x_coords * math.cos(angle_rad) + y_coords * math.sin(angle_rad)
    y_rot = -x_coords * math.sin(angle_rad) + y_coords * math.cos(angle_rad)
    
    # 两个高频图案的干涉 (一种是在X方向，一种有轻微偏转)
    pattern1 = torch.sin(x_rot * frequency * math.pi * 2.0)
    pattern2 = torch.sin((x_rot * 0.95 + y_rot * 0.05) * frequency * math.pi * 2.0)
    
    # 干涉结果：包络线的低频部分即为摩尔纹可见部分
    moire = (pattern1 * pattern2) * 0.5 + 0.5
    
    # 根据原图亮度调制（在亮部更容易看见，且带有一点色散/彩虹效应）
    gray = kornia.color.rgb_to_grayscale(x.unsqueeze(0) if len(x.shape)==3 else x).squeeze(0)
    intensity_mask = 0.15 * gray
    
    # 将摩尔纹混合为微弱的彩色变化
    color_shift = torch.tensor([1.0, 0.8, 1.2], device=x.device).view(-1, 1, 1) if len(x.shape) == 3 else torch.tensor([1.0, 0.8, 1.2], device=x.device).view(1, -1, 1, 1)
    
    moire_effect = moire * intensity_mask * color_shift
    
    y = torch.clamp(y + moire_effect - (intensity_mask * 0.5), 0, 1)
    return y


def light_leak(x: torch.Tensor, amount: float = 0.4) -> torch.Tensor:
    """
    漏光/胶片烧毁效果 (Light Leak / Film Burn)

    Args:
        x: 输入图像 (C, H, W)
        amount: 漏光强度 

    Returns:
        有漏光效果的图像
    """
    y = x.clone()
    H, W = y.shape[-2:]
    
    leak = torch.zeros((1, H, W), device=x.device)
    
    # 从侧边渗入的高斯光束
    num_leaks = np.random.randint(1, 4)
    for _ in range(num_leaks):
        # 随机选择边 (左 0, 右 1, 上 2, 下 3)
        side = np.random.randint(0, 4)
        
        radius_x = np.random.uniform(0.1, 0.5) * W
        radius_y = np.random.uniform(0.1, 0.5) * H
        
        if side == 0:
            cx, cy = 0, np.random.randint(0, H)
        elif side == 1:
            cx, cy = W - 1, np.random.randint(0, H)
        elif side == 2:
            cx, cy = np.random.randint(0, W), 0
        else:
            cx, cy = np.random.randint(0, W), H - 1
            
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=torch.float32),
            torch.arange(W, device=x.device, dtype=torch.float32),
            indexing='ij'
        )
        
        dist_sq = ((x_coords - cx)/max(1.0, float(radius_x)))**2 + ((y_coords - cy)/max(1.0, float(radius_y)))**2
        shape = torch.exp(-1.0 * dist_sq / 2.0).unsqueeze(0)
        leak = leak + shape * float(np.random.uniform(0.5, 1.0))
        
    leak = torch.clamp(leak, 0, 1) * amount
    
    # 漏光通常是橙红色的
    leak_color = torch.tensor([1.0, 0.4, 0.1], device=x.device).view(-1, 1, 1)
    if len(y.shape) == 4:
        leak_color = leak_color.unsqueeze(0)
        
    # 添加式混合 (Screen blend)
    y = 1.0 - (1.0 - y) * (1.0 - leak * leak_color)
    
    return torch.clamp(y, 0, 1)


def film_grain(x: torch.Tensor, intensity: float = 0.05, seed: Optional[int] = None) -> torch.Tensor:
    """
    胶片颗粒 (Film Grain) - 与亮度相关且有空间结构的乘性/加性颗粒

    Args:
        x: 输入图像 (C, H, W)
        intensity: 颗粒强度
        seed: 随机种子

    Returns:
        包含胶片颗粒的图像
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    y = x.clone()
    C, H, W = y.shape
    
    # 模拟胶片银盐颗粒的空间聚拢感：用稍低分辨率生成噪声再上采样
    noise_h, noise_w = max(1, int(H * 0.7)), max(1, int(W * 0.7))
    
    # 生成独立同分布高斯噪声
    noise = torch.randn((C, noise_h, noise_w), device=x.device)
    
    # 放大
    noise_up = F.interpolate(noise.unsqueeze(0), size=(H, W), mode='bicubic', align_corners=False).squeeze(0)
    
    # 颗粒强度与亮度非线性相关：中间调最明显，高光和死黑处较弱
    gray = kornia.color.rgb_to_grayscale(x.unsqueeze(0) if len(x.shape)==3 else x).squeeze(0)
    luma_mask = 1.0 - (gray - 0.5)**2 * 4.0
    luma_mask = torch.clamp(luma_mask, 0.1, 1.0)
    
    # 将颗粒应用到图像上：软光混合(Soft Light) 或 直接乘/加混合
    grain = noise_up * luma_mask * intensity
    
    y = y + grain
    return torch.clamp(y, 0, 1)


def color_bleed(x: torch.Tensor, offset: int = 3) -> torch.Tensor:
    """
    色彩渗漏/色带推移 (Color Bleed) - 模拟早期模拟录像带(VHS)色度信号偏移和带宽限制产生的水平拖影

    Args:
        x: 输入图像 (C, H, W)
        offset: 色差通道向右漂移的像素数

    Returns:
        带有色彩渗漏的图像
    """
    if len(x.shape) == 3 and x.shape[0] != 3:
        return x
        
    # 将RGB转为 YUV / YCbCr (简化的模拟)
    # 利用矩阵转换
    rgb_to_yuv = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001]
    ], device=x.device, dtype=torch.float32)
    
    yuv_to_rgb = torch.inverse(rgb_to_yuv)
    
    C, H, W = x.shape
    
    # 运用转换矩阵
    x_reshaped = x.view(3, H * W).permute(1, 0)
    yuv = torch.matmul(x_reshaped, rgb_to_yuv.t()).view(H, W, 3).permute(2, 0, 1)
    
    yuv_4d = yuv.unsqueeze(0)
    
    # 对色差通道(U, V)进行水平模糊和向右偏移
    u = yuv_4d[:, 1:2, :, :]
    v = yuv_4d[:, 2:3, :, :]
    
    # 高斯模糊 (因色度信号带宽限制)
    u_blurred = kornia.filters.gaussian_blur2d(u, (1, 7), (0.1, 3.0)).squeeze(0)
    v_blurred = kornia.filters.gaussian_blur2d(v, (1, 7), (0.1, 3.0)).squeeze(0)
    
    # 偏移色度信号
    u_shifted = torch.zeros_like(u_blurred)
    v_shifted = torch.zeros_like(v_blurred)
    
    if offset > 0:
        u_shifted[:, :, offset:] = u_blurred[:, :, :-offset]
        v_shifted[:, :, offset:] = v_blurred[:, :, :-offset]
        u_shifted[:, :, :offset] = u_blurred[:, :, :offset]
        v_shifted[:, :, :offset] = v_blurred[:, :, :offset]
    else:
        u_shifted = u_blurred
        v_shifted = v_blurred
        
    yuv[1] = u_shifted.squeeze(0)
    yuv[2] = v_shifted.squeeze(0)
    
    # 转回 RGB
    yuv_reshaped = yuv.view(3, H * W).permute(1, 0)
    rgb = torch.matmul(yuv_reshaped, yuv_to_rgb.t()).view(H, W, 3).permute(2, 0, 1)
    
    return torch.clamp(rgb, 0, 1)


def jpeg_blockiness(x: torch.Tensor, quality: int = 10) -> torch.Tensor:
    """
    JPEG 切块伪影 (JPEG Blockiness) - 模拟极低质量的 8x8 DCT 分块伪影

    Args:
        x: 输入图像 (C, H, W)
        quality: 伪影质量指标(这里并非真实JPEG质量，是降采样网格质量)

    Returns:
        有块状伪影的图像
    """
    C, H, W = x.shape
    
    block_size = max(4, min(16, 100 // max(1, quality)))
    
    # 简单的模拟方式：通过带有非重叠大stride的均值池化（马赛克）+ DCT高频去除（高斯模糊）混合
    x_4d = x.unsqueeze(0) if len(x.shape) == 3 else x
    
    y_small = F.avg_pool2d(x_4d, kernel_size=block_size, stride=block_size)
    
    # 放大回去
    y_mosaic = F.interpolate(y_small, size=(H, W), mode='nearest')
    
    # 在块边界强行混合产生类似DCT量化后的块感
    
    # 对原图进行一定程度的高频削弱
    y_blur = kornia.filters.gaussian_blur2d(x_4d, (5, 5), (1.5, 1.5))
    
    # 模拟真实图片在JPEG高压缩比下的混合形态
    y = y_blur * 0.4 + y_mosaic * 0.6
    
    if len(x.shape) == 3:
        y = y.squeeze(0)
        
    return torch.clamp(y, 0, 1)

