#!/usr/bin/env python3
"""波特尔暗夜等级（传播积分版）：把地表向上辐亮度经大气散射传播换算成天顶天空亮度。

对比 scripts/bortle.py（查表法）
-------------------------------
查表法只看本地像元，无法反映"离城 100 km 的乡村，天空仍被这座城市照亮"——
这恰恰是天文观测选点最关心的场景。本脚本做真正的传播积分：

    天空辉光 = 半径 300 km 内所有光源的贡献，按距离幂律衰减叠加

模型（中国自然保护区光污染评估，Taylor & Francis 2024，基于地面观测拟合）
---------------------------------------------------------------------
    ALR  = (1/c) · Σ rᵢ · dᵢ^(−wᵢ)          c = 567.72（按 450 m 分辨率拟合）
    wᵢ   = 2.3 · (dᵢ/350)^0.28               d 单位 km
    NASB = 0.25 · (ALR + 1)      mcd/m²      自然本底全天空亮度 0.25 mcd/m²
    mag/arcsec² = 12.59 − 2.5·log₁₀(cd/m²)

关键推导：权重 w 只依赖距离 d ⇒ 核是**空间不变的径向函数** ⇒ 可用 FFT 卷积，
不需要逐像元 300 km 暴力循环（后者在 4620 万像元上不可行）。

分辨率换算（易错点）
--------------------
常数 c = 567.72 是针对 450 m 像元拟合的。本项目用 30″（≈925 m）网格，
单像元面积是 450 m 的 (925/450)² ≈ 4.23 倍，即同一区域内 450 m 像元更多：
    Σ₄₅₀ rⱼ ≈ 4.23 × Σ₃₀″ rᵢ
故本脚本用 c_eff = 567.72 / 4.23 ≈ 134.3。（近似：假设两分辨率下辐射密度可比）

输出格式与 bortle.py 完全一致（level PNG 1–9 + levels/mags 表），前端零改动即可切换。

多年支持
--------
    python bortle_prop.py 2021 2022 2023 2024 2025
不带参数则默认跑 2021–2025。每年输出 output/bortle_prop_{year}.json，
全部跑完后再打印首年 → 末年的等级转移矩阵（限中国境内）。

⚠️ 阈值与自然本底（重要）
--------------------------
自然本底 NASB = 0.25 mcd/m² ⇒ mag = 12.59 − 2.5·log₁₀(0.25e-3) = 21.60 mag，
这是本模型的物理上限，因此 **1 级（阈值 21.7）在中国境内不可达**。这符合事实：
真正的 1 级天空需要天顶亮度 21.7 mag 以上。色标仍保留 1 级占位以免索引错位。

⚠️ 分级顺序（曾出错，勿回退）
------------------------------
BORT_BINS 从严（21.7）到宽（−99）排列，**必须反向遍历**逐次覆盖：
    for thr, l in reversed(BORT_BINS): lv = np.where(mag >= thr, l, lv)
若正向遍历，最后一次 `mag >= -99` 会命中全部像元，全图都被判成 9 级（不报错、静默全错）。
"""
import os, json, io, base64, time
import numpy as np
import rasterio
from scipy import ndimage, signal
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "proc")
OUT = os.path.join(ROOT, "output")

WEST, EAST, SOUTH, NORTH = 73.0, 136.0, 3.0, 54.0
H, W = 6120, 7560
FACTOR = 4                       # 7560×6120 → 1890×1530（120″ 网格）
YEAR = 2025

# 模型常数
C_RAW = 567.72                   # 论文常数 (450 m 分辨率)
AREA_RATIO = (0.925 / 0.45) ** 2  # ≈4.23: 30″ 像元面积 / 450 m 像元面积
C = C_RAW / AREA_RATIO           # ≈134.3, 换算到 30″ 求和口径
NATURAL_MCD = 0.25               # 自然本底全天空亮度 mcd/m²
R_KM = 300.0                     # 积分半径
KM_PER_PX = 3.7                  # 120″ 网格近似边长(km), 纬度方向; 经度方向略大, 忽略

# 伪信号判据(与 bortle.py 一致)
CLU_TH, SUSPECT_L, SUSPECT_CLUSTER = 5.0, 20.0, 200

# mag/arcsec² 下界 → Bortle 级(1..9)。用常用近似阈值。
BORT_BINS = [(21.7, 1), (21.5, 2), (21.3, 3), (20.4, 4), (19.1, 5),
             (18.4, 6), (18.0, 7), (17.5, 8), (-99.0, 9)]
MAGS = [21.8, 21.6, 21.4, 20.8, 19.7, 18.7, 18.2, 17.7, 17.2]   # 各级代表值
LEVELS = list(range(1, 10))

PROBES = [
    ("上海人民广场", 121.47, 31.23), ("北京天安门", 116.40, 39.91),
    ("深圳市中心", 114.06, 22.54), ("杭州市中心", 120.15, 30.27),
    ("拉萨市区", 91.14, 29.65), ("那曲市区", 92.06, 31.48),
    ("阿里狮泉河", 80.10, 32.50), ("格尔木市区", 94.90, 36.40),
    ("塔克拉玛干腹地", 84.0, 39.0), ("三江源腹地", 96.0, 34.5),
    ("上海近郊·淀山湖", 120.98, 31.10),   # 查表法会严重低估的典型点
    ("北京近郊·密云", 116.85, 40.38),
]


def level_of_mag(mag):
    """mag/arcsec² → Bortle 1–9（值越小天空越亮→等级越高）"""
    for thr, lv in BORT_BINS:
        if mag >= thr:
            return lv
    return 9


def build(year=YEAR):
    t0 = time.time()
    with rasterio.open(os.path.join(PROC, f"cn_{year}.tif")) as ds:
        L = ds.read(1).astype("float32")

    # 1) 伪信号掩膜: 高反射地表不参与传播积分(否则无人区会被算成城市级)
    Ls = ndimage.uniform_filter(L, size=3, mode="nearest")
    lit = Ls >= CLU_TH
    lbl, _n = ndimage.label(lit, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lbl.ravel())
    cs = sizes[lbl.ravel()].reshape(L.shape)
    suspect = (Ls >= SUSPECT_L) & (cs < SUSPECT_CLUSTER)
    Lm = np.where(suspect, 0.0, L)
    print(f"伪信号像元 {int(suspect.sum()):,} ({suspect.mean():.3%}), 已从积分中剔除")

    # 2) 降采样求和: 每块 4×4 个 30″ 像元的总辐射(保持总能量, 供积分使用)
    h2, w2 = H // FACTOR, W // FACTOR
    S = Lm[:h2 * FACTOR, :w2 * FACTOR].reshape(h2, FACTOR, w2, FACTOR).sum(axis=(1, 3))
    print(f"降采样 {S.shape}, 总辐射 {S.sum()/1e6:.1f}×10⁶ nW")

    # 3) 构建径向核: K(d) = d^(−2.3·(d/350)^0.28), 空间不变 ⇒ 可 FFT 卷积
    nr = int(np.ceil(R_KM / KM_PER_PX))
    yy, xx = np.mgrid[-nr:nr + 1, -nr:nr + 1]
    d = np.sqrt(xx ** 2 + yy ** 2) * KM_PER_PX
    d = np.maximum(d, KM_PER_PX * 0.5)          # 避免 d=0 时 0^0
    w = 2.3 * (d / 350.0) ** 0.28
    K = d ** (-w)
    K[d > R_KM] = 0.0
    print(f"核半径 {nr} px (={R_KM:.0f} km), 核大小 {K.shape}")

    # 4) FFT 卷积 = 距离加权积分
    conv = signal.fftconvolve(S, K, mode="same")
    print(f"卷积完成 {time.time()-t0:.1f}s")

    # 5) 天空亮度换算
    ALR = conv / C
    NASB = NATURAL_MCD * (ALR + 1.0)             # mcd/m²
    mag = 12.59 - 2.5 * np.log10(NASB / 1000.0)  # cd/m² → mag/arcsec²
    print(f"mag 范围 {mag.min():.2f} ~ {mag.max():.2f}")

    # 6) 分级 —— 必须反向遍历, 详见文件头「分级顺序」说明
    lv = np.full(mag.shape, 9, dtype="uint8")
    for thr, l in reversed(BORT_BINS):
        lv = np.where(mag >= thr, l, lv)

    # 7) 存疑区(高反射)单独降采样标记: 块内存疑过半则该块灰显
    sus_blk = suspect[:h2 * FACTOR, :w2 * FACTOR].reshape(h2, FACTOR, w2, FACTOR)
    lv[sus_blk.mean(axis=(1, 3)) >= 0.5] = 0

    buf = io.BytesIO()
    Image.fromarray(lv, mode="L").save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    asset = {"year": year, "w": w2, "h": h2, "bbox": [WEST, SOUTH, EAST, NORTH],
             "method": "prop", "levels": LEVELS, "mags": MAGS, "png": b64}
    p = os.path.join(OUT, f"bortle_prop_{year}.json")
    json.dump(asset, open(p, "w"))
    print(f"wrote {p} {os.path.getsize(p)/1024:.1f} KB")

    # 8) 验证点
    print(f"\n=== 验证点（{year} 年 · 传播积分）===")
    print(f"{'地点':<16s} {'本地nW':>7s} {'mag':>6s} {'等级':>4s}  备注")
    for name, lon, lat in PROBES:
        x = int((lon - WEST) / (EAST - WEST) * w2)
        y = int((NORTH - lat) / (NORTH - SOUTH) * h2)
        if not (0 <= x < w2 and 0 <= y < h2):
            print(f"{name:<16s} 超出范围"); continue
        mg, l = float(mag[y, x]), int(lv[y, x])
        lx, ly = int((lon - WEST) * 120), int((NORTH - lat) * 120)
        local = float(L[min(ly, H - 1), min(lx, W - 1)])
        note = "高反射·已标存疑" if l == 0 else ""
        print(f"{name:<16s} {local:>7.2f} {mg:>6.2f} {l if l else '—':>4}  {note}")

    d = np.bincount(lv.ravel(), minlength=10)
    print(f"\n全图等级分布（120″ 显示栅格）:")
    for l in range(10):
        if d[l]:
            tag = "存疑(灰)" if l == 0 else f"{l} 级"
            print(f"  {tag:<10s} {int(d[l]):>10,}  {d[l]/lv.size:>7.2%}")
    print("\ntotal", round(time.time() - t0, 1), "s")
    return lv


def china_mask():
    """中国境内 mask（120″ 栅格，与 lv 同尺寸）。True = 境内。
    用途：等级转移矩阵必须排除境外像元，否则「多少国土退化一级」会被境外空白污染。"""
    import json as _json
    from shapely.geometry import shape as _shape
    from rasterio.transform import from_origin as _fo
    from rasterio.features import rasterize as _ras
    bd = _json.load(open(os.path.join(ROOT, "data", "raw", "china_boundary.json")))
    geoms = [_shape(f["geometry"]) for f in bd["features"]]
    tr = _fo(WEST, NORTH, 1.0 / 120.0, 1.0 / 120.0)
    m = _ras(geoms, out_shape=(H, W), transform=tr, fill=0,
             default_value=255, dtype="uint8", all_touched=False)
    return m.reshape(H // FACTOR, FACTOR, W // FACTOR, FACTOR).mean(axis=(1, 3)) >= 128


def main(argv):
    years = [int(a) for a in argv[1:]] or [2021, 2022, 2023, 2024, 2025]
    grids = {}
    for y in years:
        grids[y] = build(y)

    if len(years) < 2:
        return
    y0, yl = years[0], years[-1]
    a, b = grids[y0], grids[yl]
    ok = china_mask() & (a > 0) & (b > 0)

    print(f"\n=== 波特尔等级转移矩阵 {y0} → {yl}（限中国境内，{int(ok.sum()):,} 个 120″ 像元）===")
    print(f"{'':>6s}" + "".join(f"{j}级".rjust(8) for j in range(1, 10)))
    for i in range(1, 10):
        row = ""
        for j in range(1, 10):
            n = int(((a == i) & (b == j) & ok).sum())
            row += f"{n:>8,}" if n else f"{'·':>8s}"
        print(f"{i}级".rjust(6) + row)

    tot = int(ok.sum())
    worse = int(((b > a) & ok).sum())      # 等级升高 = 光害加重
    better = int(((b < a) & ok).sum())
    print(f"\n境内合计 {tot:,} 像元：")
    print(f"  光害加重（等级升高） {worse:>10,}  {worse / tot:>7.2%}")
    print(f"  等级不变             {tot - worse - better:>10,}  {(tot - worse - better) / tot:>7.2%}")
    print(f"  光害减轻（等级降低） {better:>10,}  {better / tot:>7.2%}")
    print(f"\n  净加重 {worse - better:+,} 像元")


if __name__ == "__main__":
    import sys
    main(sys.argv)
