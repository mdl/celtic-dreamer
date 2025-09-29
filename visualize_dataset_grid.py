import argparse
import os
import numpy as np
from PIL import Image


def to_hwc_uint8(arr: np.ndarray) -> np.ndarray:
    """Ensure array is HxWxC uint8 in [0, 255]. Supports NCHW or NHWC, float or uint8."""
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (HWC or CHW), got shape {arr.shape}")

    # Detect channel position
    if arr.shape[-1] in (1, 3, 4):
        hwc = arr
    elif arr.shape[0] in (1, 3, 4):
        # CHW -> HWC
        hwc = np.transpose(arr, (1, 2, 0))
    else:
        # Fallback assume HWC
        hwc = arr

    # Convert dtype/range
    if hwc.dtype == np.uint8:
        return hwc
    else:
        # Assume floating 0..1 or 0..255
        a = hwc.astype(np.float32)
        if a.max() <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
        return a


def make_grid(images: np.ndarray, grid_size=(5, 5)) -> Image.Image:
    """Create a tiled grid PIL image from a list/array of HWC uint8 images.
    images: np.ndarray of shape (K, H, W, C) or list of arrays
    grid_size: (rows, cols)
    """
    rows, cols = grid_size
    if isinstance(images, list):
        imgs = [to_hwc_uint8(im) for im in images]
    else:
        imgs = [to_hwc_uint8(im) for im in images]

    if len(imgs) == 0:
        raise ValueError("No images provided to make_grid")

    H, W = imgs[0].shape[:2]
    mode = 'RGB' if imgs[0].shape[2] != 4 else 'RGBA'

    grid = Image.new(mode, (cols * W, rows * H))

    for idx, im in enumerate(imgs):
        if im.shape[0] != H or im.shape[1] != W or im.shape[2] != imgs[0].shape[2]:
            # Resize or convert channels if inconsistent
            pim = Image.fromarray(to_hwc_uint8(im)).convert(mode)
            pim = pim.resize((W, H), Image.BILINEAR)
        else:
            pim = Image.fromarray(im).convert(mode)
        r = idx // cols
        c = idx % cols
        grid.paste(pim, (c * W, r * H))
    return grid


def main():
    parser = argparse.ArgumentParser(description="Visualize 25 random images from an npz dataset in a 5x5 grid.")
    parser.add_argument("--file", type=str, default="celtic_heroes_dataset.npz", help="Path to the dataset .npz file")
    parser.add_argument("--out", type=str, default="celtic_dataset_grid_5x5.png", help="Output image file path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: dataset file not found: {args.file}")
        return 1

    if args.seed is not None:
        np.random.seed(args.seed)

    with np.load(args.file) as data:
        if 'observations' not in data:
            print("Error: 'observations' key not found in the npz file. Available keys:", list(data.keys()))
            return 1
        obs = data['observations']

    # Obs expected shape: (N, H, W, C) uint8
    if obs.ndim != 4:
        print(f"Error: Expected observations with 4 dims (N,H,W,C or N,C,H,W), got {obs.shape}")
        return 1

    N = obs.shape[0]
    k = 25
    replace = N < k
    idx = np.random.choice(N, size=k, replace=replace)
    samples = obs[idx]

    # Normalize each sample to HWC uint8
    samples_hwc = [to_hwc_uint8(im) for im in samples]

    # Ensure all the same size by using the first one's size
    grid_img = make_grid(samples_hwc, grid_size=(5, 5))

    # Create output directory if needed
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    grid_img.save(args.out)
    print(f"Saved 5x5 grid with 25 random images to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
