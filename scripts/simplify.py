#!/usr/bin/env python3
"""县级边界简化 + 紧凑二进制编码（delta + zigzag varint + base64）。

不用 gzip: 浏览器端 DecompressionStream 兼容性有缺口(Safari<16.4 / Firefox<113),
改为自研 delta+varint 编码, 解码器 ~20 行 JS, 零 API 依赖, 体积优于 gzip:
  tol=0.005 -> 27 万点, 本编码 base64 ≈ 1.0 MB (gzip 方案 1.97 MB)

输出: data/raw/districts_web.json = {meta:[...], geo:"<base64>", q:10000}
  meta[i] = {a:adcode, n:县名, p:省名, c:[lon,lat], b:[w,s,e,n]}
  geo 按 meta 顺序存储, 每单元: varint(nRing) -> 每环 varint(nPts) -> 逐点 zigzag(dx),zigzag(dy)
"""
import os, json, base64
import numpy as np
from shapely.geometry import shape as shp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
SRC = os.path.join(RAW, "districts_full.json")
OUT = os.path.join(RAW, "districts_web.json")

TOL = 0.005      # 度, ≈500 m
Q = 10000        # 坐标量化: 1e-4 度 ≈ 11 m


def varint(v, out):
    """LEB128 无符号 varint"""
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return


def zz(v):
    """zigzag: 有符号 -> 无符号"""
    return (v << 1) ^ (v >> 63) if v < 0 else (v << 1)


def ring_points(coords):
    """量化 + 去连续重复点, 返回 int32 Nx2"""
    a = np.rint(np.asarray(coords) * Q).astype("int64")
    if len(a) < 4:
        return None
    keep = np.ones(len(a), bool)
    keep[1:] = np.any(np.diff(a, axis=0) != 0, axis=1)
    a = a[keep]
    if len(a) < 4:
        return None
    return a


def main():
    d = json.load(open(SRC))
    feats = d["features"]
    meta, buf = [], bytearray()

    for f in feats:
        g = shp(f["geometry"]).simplify(TOL, preserve_topology=True)
        if g.is_empty:
            parts = []
        else:
            parts = [p for p in (g.geoms if g.geom_type == "MultiPolygon" else [g])
                     if not p.is_empty and p.exterior is not None]
        rings = []
        for p in parts:
            rr = ring_points(p.exterior.coords)
            if rr is not None:
                rings.append(rr)
        if not rings:
            continue

        # 标注点: 取最大环的质心, 若不在面内则用 representative_point
        big = max(parts, key=lambda p: p.area)
        c = big.centroid
        if not g.contains(c):
            c = big.representative_point()
        xs = np.concatenate([r[:, 0] for r in rings]) / Q
        ys = np.concatenate([r[:, 1] for r in rings]) / Q

        varint(len(rings), buf)
        for r in rings:
            varint(len(r), buf)
            px, py = 0, 0
            for x, y in r:
                varint(zz(int(x) - px), buf)
                varint(zz(int(y) - py), buf)
                px, py = int(x), int(y)

        meta.append({
            "a": f["properties"]["adcode"],
            "n": f["properties"]["name"],
            "p": f["properties"]["province"],
            "c": [round(c.x, 3), round(c.y, 3)],
            "b": [round(float(xs.min()), 3), round(float(ys.min()), 3),
                  round(float(xs.max()), 3), round(float(ys.max()), 3)],
        })

    geo = base64.b64encode(bytes(buf)).decode("ascii")
    out = {"q": Q, "tol": TOL, "meta": meta, "geo": geo}
    with open(OUT, "w") as fh:
        json.dump(out, fh, ensure_ascii=False)
    print(f"units={len(meta)} raw={len(buf)/1048576:.2f}MB base64={len(geo)/1048576:.2f}MB "
          f"json={os.path.getsize(OUT)/1048576:.2f}MB")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
