#!/usr/bin/env python3
"""统一网格预处理：裁剪中国区 + 量纲还原 + 掩膜，输出 proc/cn_{Y}.tif (float32)。

目标网格: EPSG:4326, bbox 73-136E / 3-54N, 30 arc-sec (~925 m), average 重采样。
输出: 7560 x 6120 float32, 单位 nW/cm^2/sr。
"""
import sys, os, json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "proc")

# bbox 覆盖中国全境（含南海诸岛与台湾地区）
WEST, EAST, SOUTH, NORTH = 73.0, 136.0, 3.0, 54.0
RES = 1.0 / 120.0  # 30 arc-sec
WIDTH = int(round((EAST - WEST) / RES))   # 7560
HEIGHT = int(round((NORTH - SOUTH) / RES))  # 6120
DST_TRANSFORM = from_origin(WEST, NORTH, RES, RES)
PROFILE = dict(
    driver="GTiff", dtype="float32", count=1, width=WIDTH, height=HEIGHT,
    crs="EPSG:4326", transform=DST_TRANSFORM, compress="deflate",
    tiled=True, nodata=-9999.0,
)


def warp_to_grid(src_path, scale=1.0, band=1):
    with rasterio.open(src_path) as src:
        dst = np.full((HEIGHT, WIDTH), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=DST_TRANSFORM,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.average,
            num_threads=4,
        )
    arr = dst * float(scale)
    arr[~np.isfinite(arr)] = 0.0
    arr[arr < 0] = 0.0           # 负辐亮度为噪声
    return arr.astype("float32")


def save(arr, year):
    p = os.path.join(PROC, f"cn_{year}.tif")
    prof = dict(PROFILE)
    with rasterio.open(p, "w", **prof) as ds:
        ds.write(arr, 1)
        ds.update_tags(units="nW/cm^2/sr", year=str(year),
                       bbox=f"{WEST},{SOUTH},{EAST},{NORTH}", res_arcsec="30")
    return p


def main():
    os.makedirs(PROC, exist_ok=True)
    jobs = json.loads(sys.argv[1])  # [{"src":..., "year":..., "scale":...}, ...]
    for j in jobs:
        src = os.path.join(ROOT, j["src"])
        arr = warp_to_grid(src, j.get("scale", 1.0), j.get("band", 1))
        v = arr[arr > 0]
        print(f"{j['year']}: n={arr.size} valid={v.size} "
              f"mean={v.mean():.3f} p99={np.percentile(v,99):.2f} max={v.max():.2f} -> {save(arr, j['year'])}",
              flush=True)


if __name__ == "__main__":
    main()
