#!/usr/bin/env python3
"""波特尔暗夜等级估算（原型 · 查表法）。

用途
----
给地图新增一个「波特尔等级」图层：把 VIIRS 地表向上辐亮度按经验阈值映射到
波特尔 1–9 级，使天文摄影用户能直接读出"某地大概几级天空"。

方法来源
--------
阈值表来自 Stargaze Atlas 公开的 VIIRS → Bortle 映射（其数据同为 VIIRS 年度合成）：

    <0.25 nW → 1–2 级    0.25–0.75 → 3 级    0.75–3 → 4 级    3–8 → 5 级
    8–20 → 6 级          20–40 → 7 级        40–100 → 8 级    ≥100 → 9 级

mag/arcsec² 为各等级的常用代表性近似值（不同来源略有 ±0.2 差异）。

⚠️ 这是 PROXY 不是实测
----------------------
1. VIIRS 测地表向上辐亮度，波特尔描述天顶天空亮度，中间隔着大气散射传播模型，
   本脚本只做查表映射，**不做传播计算**。
2. **致命局限**：查表只看本地像元（3×3 平滑后），无法反映半径 100–300 km 内城市
   的远距离天空辉光。因此**城市近郊会被系统性低估**（本地暗、但天空被远处城市照亮）。
   要修正必须做传播积分（见 README「后续」）。
3. 高反射地表（盐湖/积雪/裸岩）会把月光反射回卫星，造成无人区虚高。本脚本用
   「连片亮区规模」判据标记这类像元为存疑（值 0，前端灰色显示）。

输出: output/bortle_2025.json（前端资产：8bit PNG base64，值 1–9，0=存疑/境外）
"""
import os, json, io, base64
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "proc")
OUT = os.path.join(ROOT, "output")

WEST, EAST, SOUTH, NORTH = 73.0, 136.0, 3.0, 54.0
H, W = 6120, 7560
FACTOR = 4                      # 与 pack.py 一致: 7560x6120 -> 1890x1530
YEAR = 2025

# 上界(不含) → (Bortle 级, 代表 mag/arcsec²)
TABLE = [
    (0.25,  1, 21.8),
    (0.75,  3, 21.4),
    (3.0,   4, 21.0),
    (8.0,   5, 20.2),
    (20.0,  6, 19.3),
    (40.0,  7, 18.5),
    (100.0, 8, 18.0),
    (float("inf"), 9, 17.5),
]
BOUNDS = [t[0] for t in TABLE]
LEVELS = [t[1] for t in TABLE]
MAGS = [t[2] for t in TABLE]

# 伪信号判据: 亮度 >=20 nW 但所属连片亮区 <200 px → 高反射地表/孤立火炬
CLU_TH = 5.0
SUSPECT_L = 20.0
SUSPECT_CLUSTER = 200

# 验证点（用于 sanity check）
PROBES = [
    ("上海人民广场", 121.47, 31.23), ("北京天安门", 116.40, 39.91),
    ("深圳市中心", 114.06, 22.54), ("杭州市中心", 120.15, 30.27),
    ("拉萨市区", 91.14, 29.65), ("那曲市区", 92.06, 31.48),
    ("阿里狮泉河", 80.10, 32.50), ("格尔木市区", 94.90, 36.40),
    ("塔克拉玛干腹地", 84.0, 39.0), ("三江源腹地", 96.0, 34.5),
]


def build():
    with rasterio.open(os.path.join(PROC, f"cn_{YEAR}.tif")) as ds:
        L = ds.read(1).astype("float32")

    # 1) 3x3 均值平滑，抑制单像元噪声
    Ls = ndimage.uniform_filter(L, size=3, mode="nearest")

    # 2) 伪信号标记
    lit = Ls >= CLU_TH
    lbl, n = ndimage.label(lit, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lbl.ravel())
    cs = sizes[lbl.ravel()].reshape(Ls.shape)
    suspect = (Ls >= SUSPECT_L) & (cs < SUSPECT_CLUSTER)
    print(f"伪信号像元: {int(suspect.sum()):,} ({suspect.mean():.3%})")

    # 3) 查表映射
    lvl = np.digitize(Ls, BOUNDS, right=False).astype("uint8")   # 0..7 -> 索引
    lvl = np.take(np.array(LEVELS, dtype="uint8"), lvl)
    lvl[suspect] = 0            # 0 = 存疑

    # 4) 降采样: 块内取最大等级（宁可高估光害，也不让用户到了现场发现比预期亮）
    #    但存疑标记不能靠 max 传递(0 永远选不中, 实测会丢掉 99.8% 的存疑像元),
    #    改为单独按"块内存疑过半"判定——密集高反射区(盐湖/裸岩)整块灰显,
    #    孤立伪信号不影响周边真实城镇。
    h2, w2 = H // FACTOR, W // FACTOR
    blk = lvl[:h2 * FACTOR, :w2 * FACTOR].reshape(h2, FACTOR, w2, FACTOR)
    small = blk.max(axis=(1, 3)).astype("uint8")
    sus_blk = suspect[:h2 * FACTOR, :w2 * FACTOR].reshape(h2, FACTOR, w2, FACTOR)
    small[sus_blk.mean(axis=(1, 3)) >= 0.5] = 0

    buf = io.BytesIO()
    Image.fromarray(small, mode="L").save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    print(f"grid {small.shape}, png {buf.getbuffer().nbytes/1024:.1f} KB")

    asset = {
        "year": YEAR, "w": w2, "h": h2,
        "bbox": [WEST, SOUTH, EAST, NORTH],
        "levels": LEVELS, "mags": MAGS, "bounds": [b if b != float("inf") else None for b in BOUNDS],
        "png": b64,
    }
    p = os.path.join(OUT, f"bortle_{YEAR}.json")
    json.dump(asset, open(p, "w"))
    print("wrote", p, round(os.path.getsize(p) / 1024, 1), "KB")

    # 5) 验证点报告
    print(f"\n=== 验证点（{YEAR} 年，3×3 平滑后辐亮度 → 波特尔级）===")
    print(f"{'地点':<14s} {'nW':>8s}  {'等级':>4s}  {'mag/arcsec²':>10s}  备注")
    for name, lon, lat in PROBES:
        r = int(round((NORTH - lat) * 120.0))
        c = int(round((lon - WEST) * 120.0))
        if not (0 <= r < H and 0 <= c < W):
            print(f"{name:<14s} 超出范围"); continue
        v = float(Ls[r, c])
        lv = int(lvl[r, c])
        mg = MAGS[LEVELS.index(lv)] if lv in LEVELS else None
        note = "⚠️ 伪信号，已标存疑" if suspect[r, c] else ""
        print(f"{name:<14s} {v:>8.2f}  {lv if lv else '—':>4}  "
              f"{mg if mg else '—':>10}  {note}")

    dist = np.bincount(lvl.ravel(), minlength=10)
    print("\n全图等级分布（30″ 像元）:")
    for lv in range(10):
        if dist[lv]:
            tag = "存疑/境外" if lv == 0 else f"{lv} 级"
            print(f"  {tag:<10s} {int(dist[lv]):>12,}  {dist[lv]/lvl.size:>7.2%}")


if __name__ == "__main__":
    build()
