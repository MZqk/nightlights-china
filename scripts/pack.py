#!/usr/bin/env python3
"""把年度栅格打包为前端可直接解码的 16bit 安全 PNG 资产。

16bit 数值拆成 R(高字节)/G(低字节) 两通道存 PNG -> 浏览器 getImageData 可精确还原 uint16,
且 PNG 对大范围 0 值区压缩率极高。
输出: output/assets.json  (纯 JSON, 供 build_html.py 内联)
"""
import os, json, base64, io
import numpy as np
import rasterio
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "proc")
OUT = os.path.join(ROOT, "output")

FACTOR = 4          # 7560x6120 -> 1890x1530
TARGET_W, TARGET_H = 7560 // FACTOR, 6120 // FACTOR
WEST, EAST, SOUTH, NORTH = 73.0, 136.0, 3.0, 54.0


def decimate(arr, f=FACTOR):
    h, w = arr.shape
    h2, w2 = h // f * f, w // f * f
    a = arr[:h2, :w2].reshape(h2 // f, f, w2 // f, f)
    return a.mean(axis=(1, 3)).astype("float32")


def to_png_b64(u16):
    h, w = u16.shape
    rgb = np.zeros((h, w, 3), dtype="uint8")
    rgb[..., 0] = (u16 >> 8).astype("uint8")
    rgb[..., 1] = (u16 & 0xFF).astype("uint8")
    rgb[..., 2] = 0
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), buf.getbuffer().nbytes


def main():
    os.makedirs(OUT, exist_ok=True)     # output/ 整个被删时也能直接重建
    files = sorted(f for f in os.listdir(PROC) if f.startswith("cn_") and f.endswith(".tif") and "cogref" not in f)
    years = [int(f[3:7]) for f in files]
    print("years:", years)
    layers = {}
    for y, f in zip(years, files):
        with rasterio.open(os.path.join(PROC, f)) as ds:
            arr = ds.read(1)
            print(f"  {y} src {arr.shape} max={arr.max():.1f}", flush=True)
        d = decimate(arr)
        q = np.clip(np.rint(d), 0, 65535).astype("uint16")
        b64, nbytes = to_png_b64(q)
        layers[str(y)] = b64
        print(f"  {y} -> {q.shape} max={int(q.max())} png={nbytes/1048576:.2f}MB", flush=True)

    meta = {
        "width": TARGET_W, "height": TARGET_H,
        "bbox": [WEST, SOUTH, EAST, NORTH],
        "years": [str(y) for y in years],
        "layers": layers,
    }
    p = os.path.join(OUT, "assets.json")
    with open(p, "w") as fh:
        json.dump(meta, fh)
    print("wrote", p, os.path.getsize(p) / 1048576, "MB")


if __name__ == "__main__":
    main()
