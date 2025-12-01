import os
import numpy as np

import numpy as np
from scipy.ndimage import zoom

def noise(width, height, vmin=-0.5, vmax=10.0, freq=4, octaves=1, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    H = np.zeros((height, width), dtype=float)

    for o in range(octaves):
        scale = freq * (2**o)

        # 这里的 gh, gw 可以理解为“采样点密度”
        gh = max(2, height // scale)
        gw = max(2, width // scale)

        coarse = rng.random((gh, gw))

        # 用插值平滑放大，而不是 kron 复制
        zoom_y = height / gh
        zoom_x = width / gw
        up = zoom(coarse, (zoom_y, zoom_x), order=3)  # order=3 是三次样条，比较平滑

        up = up[:height, :width]

        H += up / (2**o)

    # 归一化到 [0, 1]
    H -= H.min()
    if H.max() > 0:
        H /= H.max()

    # 映射到 [vmin, vmax]，再转 int
    result = vmin + H * (vmax - vmin)
    return np.round(result).astype(int)

def save_to_file(array:np.ndarray, file_name:str):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, file_name)

    np.savetxt(out_path, array, fmt='%d', delimiter=" ")

def random_spikes(num, freq: float, size: int, seed=None):
    rng = np.random.default_rng(seed)
    mask = rng.random(num.shape) < freq
    return num + mask * size

def increase_and_clamp(num, increment, high=float('inf'), low=0):
    return np.minimum(high, np.maximum(num+increment, low))

def spiky(width, height, spike_size, spike_decrement, spike_freq, seed=None):
    hmap = np.zeros((width, height), dtype=float)
    hmap = random_spikes(hmap, spike_freq, spike_size, seed)

    while True:
        up    = np.roll(hmap, -1, axis=0)
        down  = np.roll(hmap,  1, axis=0)
        left  = np.roll(hmap, -1, axis=1)
        right = np.roll(hmap,  1, axis=1)

        neighbor_max = np.maximum.reduce([up, down, left, right])
        required = np.maximum(0, neighbor_max - spike_decrement)

        new_hmap = np.maximum(hmap, required)

        if np.array_equal(new_hmap, hmap):
            break

        hmap = new_hmap

    return hmap


if __name__ == "__main__":
    noisy = noise(50, 50, vmin=0, vmax=4, freq=1, octaves=3, seed=42)
    spike = spiky(50, 50, spike_size=8, spike_decrement=3, spike_freq=0.08, seed=42)
    save_to_file(noisy, "noise.txt")
    save_to_file(spike, "spiky.txt")
    
