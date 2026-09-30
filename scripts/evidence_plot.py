#!/usr/bin/env python3
"""伪信号证据图（v2 —— 基于实测修正）

原设想用「逐年单调性」区分真城市与伪信号，实测否决：城市核心像元逐年同样波动
（杭州 79.7→72.2→78.5→73.8→74.4，深圳持续下降）。故改用两条站得住的判据：
  A. 绝对辐亮度水平：无人区与城市核心同量级 → 不合理
  B. 空间形态：高值像元的孤立率与最大连片团块规模
"""
import os
import numpy as np
import rasterio
import json
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, transform as wtransform
from shapely.geometry import shape as shp
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "proc")
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "output", "notes_images")
YEARS = [2021, 2022, 2023, 2024, 2025]

# 城市核心（有人口、有建成区） vs 无人区/荒漠（无城市）
SITES = [
    ("上海核心", 121.4737, 31.2304, "city"),
    ("深圳核心", 114.0557, 22.5431, "city"),
    ("杭州核心", 120.15, 30.27, "city"),
    ("青海格尔木\n（荒漠/盐湖）", 94.9, 36.4, "void"),
    ("西藏阿里\n（无人区）", 80.1, 32.5, "void"),
    ("拉萨市区", 91.14, 29.65, "city"),
]
REGIONS = ["浙江省", "新疆维吾尔自治区", "青海省", "内蒙古自治区", "西藏自治区"]


def site_mean(lon, lat):
    vals = []
    for y in YEARS:
        with rasterio.open(os.path.join(PROC, f"cn_{y}.tif")) as s:
            vals.append(float(list(s.sample([(lon, lat)]))[0][0]))
    return float(np.mean(vals))


def morphology(regions):
    bd = json.load(open(os.path.join(RAW, "china_boundary.json")))
    geo = {}
    for f in bd["features"]:
        n = f["properties"].get("name", "")
        if n in regions:
            geo[n] = shp(f["geometry"])
    out = {}
    with rasterio.open(os.path.join(PROC, "cn_2025.tif")) as s:
        gt = s.transform
        for name, g in geo.items():
            w = from_bounds(*g.bounds, transform=gt).round_offsets().round_lengths()
            m = geometry_mask([g], out_shape=(int(w.height), int(w.width)),
                              transform=wtransform(w, gt), invert=True)
            a = s.read(1, window=w).astype("float32")
            v = m & (a > 20)
            pad = np.pad(v, 1)
            nb = np.zeros_like(v, dtype=np.int16)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nb += pad[1+dy:1+dy+v.shape[0], 1+dx:1+dx+v.shape[1]]
            tot = int(v.sum())
            lone = int((v & (nb <= 1)).sum())
            seen = np.zeros(v.shape, dtype=bool)
            best = 0
            for y0, x0 in zip(*np.where(v)):
                if seen[y0, x0]:
                    continue
                q = deque([(y0, x0)]); seen[y0, x0] = True; c = 0
                while q:
                    y, x = q.popleft(); c += 1
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        yy, xx = y+dy, x+dx
                        if 0 <= yy < v.shape[0] and 0 <= xx < v.shape[1] and v[yy, xx] and not seen[yy, xx]:
                            seen[yy, xx] = True; q.append((yy, xx))
                best = max(best, c)
            out[name] = dict(total=tot, lone_pct=lone/tot*100 if tot else 0, largest=best)
    return out


fig = plt.figure(figsize=(14, 6.2), facecolor="#0d1117")
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.32)

# ---- Panel A: 绝对辐亮度水平 ----
axA = fig.add_subplot(gs[0])
names = [s[0] for s in SITES]
vals = [site_mean(s[1], s[2]) for s in SITES]
cols = ["#f0883e" if s[3] == "city" else "#58a6ff" for s in SITES]
ypos = np.arange(len(SITES))[::-1]
axA.barh(ypos, vals, color=cols, height=0.62)
for y, v in zip(ypos, vals):
    axA.text(v + 1.6, y, f"{v:.0f}", va="center", color="#e6edf3", fontsize=11)
axA.set_yticks(ypos); axA.set_yticklabels(names, color="#e6edf3", fontsize=11)
axA.set_xlabel("五年平均辐亮度  nW·cm⁻²·sr⁻¹", color="#8b949e", fontsize=11.5)
axA.set_title("A · 无人区的“亮度”与城市核心同量级", color="#e6edf3", fontsize=14, pad=14, fontweight="bold")
axA.set_facecolor("#151b23")
axA.tick_params(colors="#8b949e"); axA.grid(axis="x", color="#222b36", lw=0.8)
for sp in axA.spines.values():
    sp.set_color("#2a3340")
axA.set_xlim(0, max(vals) * 1.18)

# ---- Panel B: 形态（孤立率 + 最大团块） ----
axB = fig.add_subplot(gs[1])
mo = morphology(REGIONS)
short = {"浙江省": "浙江", "新疆维吾尔自治区": "新疆", "青海省": "青海",
         "内蒙古自治区": "内蒙古", "西藏自治区": "西藏"}
order = sorted(mo.items(), key=lambda kv: kv[1]["lone_pct"])
ypos = np.arange(len(order))[::-1]
axB.barh(ypos, [v["lone_pct"] for _, v in order], color=[
    "#f0883e" if k == "浙江省" else ("#e5534b" if k == "西藏自治区" else "#58a6ff")
    for k, _ in order], height=0.6)
for y, (k, v) in zip(ypos, order):
    axB.text(v["lone_pct"] + 0.35, y,
             f"{v['lone_pct']:.1f}%   最大团块 {v['largest']:,}",
             va="center", color="#8b949e", fontsize=10)
axB.set_yticks(ypos)
axB.set_yticklabels([short[k] for k, _ in order], color="#e6edf3", fontsize=11.5)
axB.set_xlabel("亮像元（>20 nW）中“孤立点”占比", color="#8b949e", fontsize=11.5)
axB.set_title("B · 亮像元的形态：城市连片，荒漠零散", color="#e6edf3", fontsize=14, pad=14, fontweight="bold")
axB.set_facecolor("#151b23")
axB.tick_params(colors="#8b949e"); axB.grid(axis="x", color="#222b36", lw=0.8)
for sp in axB.spines.values():
    sp.set_color("#2a3340")
axB.set_xlim(0, max(v["lone_pct"] for _, v in order) * 1.75)

fig.subplots_adjust(bottom=0.20)
fig.suptitle("为什么西藏的“增亮”不能直接采信", color="#e6edf3", fontsize=16, y=0.985, fontweight="bold")
fig.text(0.5, 0.035,
         "左：无人区的辐亮度水平与城市核心相当 → 数值本身不合理。      "
         "右：西藏亮像元的孤立比例是浙江的 2.7 倍，最大连片团块仅为浙江的 1/12 → 形态不像城市。",
         ha="center", color="#8b949e", fontsize=10.5)
p = os.path.join(OUT, "06_伪信号证据.png")
fig.savefig(p, dpi=145, facecolor="#0d1117")
print("wrote", p)
print("A:", {n: round(v, 1) for n, v in zip(names, vals)})
print("B:", {k: (round(v["lone_pct"], 1), v["largest"]) for k, v in mo.items()})
