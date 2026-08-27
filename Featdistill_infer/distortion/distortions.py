import torch
from torchvision.io.image import decode_jpeg, encode_jpeg
import numpy as np
import math
import kornia
from .utils_distortions import fspecial, filter2D, curves, imscatter, mapmm
from torch.nn import functional as F

def gaussian_blur(x: torch.Tensor, blur_sigma: int = 0.1) -> torch.Tensor:
    fs = 2 * math.ceil(2 * blur_sigma) + 1
    h = fspecial('gaussian', (fs, fs), blur_sigma)
    h = torch.from_numpy(h).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def lens_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    h = fspecial('disk', radius)
    h = torch.from_numpy(h).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def color_saturation(x: torch.Tensor, factor: int) -> torch.Tensor:
    factor = max(float(factor), 0.0)
    hsv = kornia.color.rgb_to_hsv(x)
    hsv[1, ...] *= factor
    y = kornia.color.hsv_to_rgb(hsv)
    return y


def color_shift(x: torch.Tensor, amount: int) -> torch.Tensor:
    def perc(values, percentile):
        percentile = min(max(float(percentile), 0.0), 100.0)
        return torch.quantile(values.reshape(-1).float(), percentile / 100.0)

    gray = kornia.color.rgb_to_grayscale(x)
    gradxy = kornia.filters.spatial_gradient(gray.unsqueeze(0), 'diff')
    e = torch.sum(gradxy ** 2, 2) ** 0.5

    fs = 2 * math.ceil(2 * 4) + 1
    h = fspecial('gaussian', (fs, fs), 4)
    h = torch.from_numpy(h).float()

    e = filter2D(e, h.unsqueeze(0)).squeeze(0).squeeze(0)

    mine = torch.min(e)
    maxe = torch.max(e)

    if mine < maxe:
        e = (e - mine) / (maxe - mine)

    percdev = [1, 1]
    valuehi = perc(e, 100 - percdev[1])
    valuelo = 1 - perc(1 - e, 100 - percdev[0])

    e = torch.max(torch.min(e, valuehi), valuelo)

    channel = 1
    g = x[channel, :, :]
    a = np.random.random((1, 2))
    amount_shift = np.round(a / (np.sum(a ** 2) ** 0.5) * amount)[0].astype(int)

    y = F.pad(g, (amount_shift[0], amount_shift[0]), mode='replicate')
    y = F.pad(y.transpose(1, 0), (amount_shift[1], amount_shift[1]), mode='replicate').transpose(1, 0)
    y = torch.roll(y, (amount_shift[0], amount_shift[1]), dims=(0, 1))

    if amount_shift[1] != 0:
        y = y[amount_shift[1]:-amount_shift[1], ...]
    if amount_shift[0] != 0:
        y = y[..., amount_shift[0]:-amount_shift[0]]

    yblend = y * e + x[channel, ...] * (1 - e)
    x[channel, ...] = yblend

    return x


def jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    x_loc = x.clone()
    x_loc *= 255.
    x_loc = x_loc.clamp(0, 255)
    y = encode_jpeg(x_loc.byte().cpu(), quality=quality)
    y = (decode_jpeg(y) / 255.).to(torch.float32)
    return y


def white_noise(x: torch.Tensor, var: float, clip: bool = True, rounds: bool = False) -> torch.Tensor:
    noise = torch.randn(*x.size(), dtype=x.dtype) * math.sqrt(var)

    y = x + noise

    if clip and rounds:
        y = torch.clip((y * 255.0).round(), 0, 255) / 255.
    elif clip:
        y = torch.clip(y, 0, 1)
    elif rounds:
        y = (y * 255.0).round() / 255.
    return y


def impulse_noise(x: torch.Tensor, d: float, s_vs_p: float = 0.5) -> torch.Tensor:
    num_sp = int(d * x.shape[0] * x.shape[1] * x.shape[2])

    coords = np.concatenate((np.random.randint(0, 3, (num_sp, 1)),
                             np.random.randint(0, x.shape[1], (num_sp, 1)),
                             np.random.randint(0, x.shape[2], (num_sp, 1))), 1)

    num_salt = int(s_vs_p * num_sp)

    coords_salt = coords[:num_salt].transpose(1, 0)
    coords_pepper = coords[num_salt:].transpose(1, 0)

    x[tuple(coords_salt)] = 1
    x[tuple(coords_pepper)] = 0

    return x


def brighten(x: torch.Tensor, amount: float) -> torch.Tensor:
    lab = kornia.color.rgb_to_lab(x)

    l = lab[0, ...] / 100.
    l_ = curves(l, 0.5 + amount / 2)
    lab[0, ...] = l_ * 100.

    y = curves(x, 0.5 + amount / 2)

    j = torch.clamp(kornia.color.lab_to_rgb(lab), 0, 1)

    y = (2 * y + j) / 3

    return y


def darken(x: torch.Tensor, amount: float, dolab: bool = False) -> torch.Tensor:
    lab = kornia.color.rgb_to_lab(x)
    if dolab:
        l = lab[0, ...] / 100.
        l_ = curves(l, 0.5 + amount / 2)
        lab[0, ...] = l_ * 100.

    y = curves(x, 0.5 - amount / 2)

    if dolab:
        j = torch.clamp(kornia.color.lab_to_rgb(lab), 0, 1)
        y = (2 * y + j) / 3

    return y


def jitter(x: torch.Tensor, amount: float) -> torch.Tensor:
    y = imscatter(x, amount, 5)
    return y


def quantization(x: torch.Tensor, levels: int) -> torch.Tensor:
    image = kornia.color.rgb_to_grayscale(x) * 255
    image = image.cpu().numpy()
    num_classes = levels

    # minimum variance thresholding
    hist, bins = np.histogram(image, num_classes, [0, 255])

    return_thresholds = np.zeros(num_classes - 1)
    for i in range(num_classes - 1):
        return_thresholds[i] = bins[i + 1]

    # quantize image with thresholds
    bins = torch.tensor([0] + return_thresholds.tolist() + [256])
    bins = bins.type(torch.int)
    image = torch.bucketize(x.contiguous() * 255., bins).to(torch.float32)
    image = mapmm(image)
    return image


def linear_contrast_change(x: torch.Tensor, amount: float) -> torch.Tensor:
    y = curves(x, [0.25 - amount / 4, 0.75 + amount / 4])
    return y
