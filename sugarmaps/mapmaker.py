import numpy as np

def make_heightmap(width, height, vmin=-0.5, vmax=10.0, freq=4, octaves=1, seed=None):
    """
    Generate a smooth 2D height map using value-noise-like smoothing.
    - width, height: output size
    - vmin, vmax: value range
    - freq: base frequency (roughness); higher = more detail
    - octaves: number of noise scales to sum
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    H = np.zeros((height, width), dtype=float)

    for o in range(octaves):
        # 每个 octave 的粗网格尺寸
        scale = freq * (2**o)
        gh = max(2, height // scale)
        gw = max(2, width // scale)

        coarse = rng.random((gh, gw))

        # 线性插值放大到目标大小
        # 先在纵向重复，再在横向重复（简单一点）
        zoom_y = int(np.ceil(height / gh))
        zoom_x = int(np.ceil(width / gw))
        up = np.kron(coarse, np.ones((zoom_y, zoom_x)))

        up = up[:height, :width]  # 裁剪到精确大小

        # 叠加不同尺度噪声，权重随 octave 衰减
        H += up / (2**o)

    # 归一化到 [0, 1]
    H -= H.min()
    if H.max() > 0:
        H /= H.max()

    # 映射到 [vmin, vmax]
    result = vmin + H * (vmax - vmin)
    return np.round(result)

if __name__ == "__main__":
    hmap = make_heightmap(20, 20, vmin=0, vmax=4, freq=1, octaves=3, seed=42)
    print(hmap)
