#!/usr/bin/env python3
"""县市区级分区统计（2875 单元）。

与 zonal.py 的关键区别: 不使用"逐多边形开窗"(2875 个窗口 × 5 年 = 慢 1~2 个数量级),
改为 rasterize 一次性生成标签栅格 + np.bincount 向量化聚合:
  rasterize 1.2 s / 读年 0.5 s / 每对 bincount 0.2 s  -> 全流程 < 30 s

口径与省级一致:
  有效像元 = max(L1, L2) >= MIN_SIG (1.5 nW·cm⁻²·sr⁻¹)
  Δ = log10((L2 + EPS)/(L1 + EPS)), EPS = 0.1
置信分档(前端据此灰显) — 三重门限, 缺一即降级:
  tier2 高: 有效像元 >= 1000  且 末年有效像元均值 >= 5 nW  且 所属最大连片亮区 >= 2000 像元
  tier1 中: 有效像元 >=  200  且 末年有效像元均值 >= 3 nW  且 所属最大连片亮区 >=  500 像元
  tier0 低: 其余 -> 不进排行、地图灰显

第三道门限(cluster)是必需的: 前两道拦不住油田火炬。实测 2021→2025 黑龙江肇源县
(大庆油田) ×36.66、有效像元 574、均值 27 nW —— 前两道全过, 但它所属的最大连片亮区
只有 176 像元, 而真实城市县域是 2000~42000(长三角连片)。孤立点率判据在县级也失效
(肇源孤立率 0.0%, 火炬田本身就是连片亮斑), 故改用"所属连片亮区规模"。

输出: output/district_stats.csv, output/district_stats.json (前端精简)
"""
import os, json, csv, itertools, time
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import shape as shp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "proc")
OUT = os.path.join(ROOT, "output")
WEB = os.path.join(ROOT, "data", "raw", "districts_web.json")

MIN_SIG = 1.5
EPS = 0.1
DEAD = 0.05
CLU_TH = 5.0        # 连片亮区判定阈值 nW
WEST, EAST, SOUTH, NORTH = 73.0, 136.0, 3.0, 54.0


def main():
    t0 = time.time()
    web = json.load(open(WEB))
    meta = web["meta"]
    codes = [m["a"] for m in meta]
    N = len(meta) + 1
    print(f"units={len(meta)}", flush=True)

    years = sorted(int(f[3:7]) for f in os.listdir(PROC)
                   if f.startswith("cn_") and f.endswith(".tif") and "cogref" not in f)
    print("years:", years, flush=True)

    # ---- 标签栅格 ----
    full = json.load(open(os.path.join(ROOT, "data", "raw", "districts_full.json")))
    by_code = {f["properties"]["adcode"]: f for f in full["features"]}
    shapes = [(shp(by_code[c]["geometry"]), i + 1) for i, c in enumerate(codes) if c in by_code]
    tr = from_bounds(WEST, SOUTH, EAST, NORTH, 7560, 6120)
    lab = rasterize(shapes, out_shape=(6120, 7560), transform=tr, fill=0,
                    dtype="uint16", all_touched=False)
    idx = lab.ravel()
    n_lab = int((idx > 0).sum())
    print(f"rasterize {time.time()-t0:.1f}s, labeled px={n_lab}", flush=True)
    del lab, shapes, full, by_code

    # ---- 逐年 ----
    year_px = np.zeros((len(years), N), dtype="int64")
    year_sum = np.zeros((len(years), N))
    year_mean = np.zeros((len(years), N))
    for j, y in enumerate(years):
        with rasterio.open(os.path.join(PROC, f"cn_{y}.tif")) as ds:
            a = ds.read(1).ravel().astype("float32")
        m = a >= MIN_SIG
        year_px[j] = np.bincount(idx[m], minlength=N)
        year_sum[j] = np.bincount(idx[m], weights=a[m].astype("float64"), minlength=N)
        year_mean[j] = year_sum[j] / np.maximum(year_px[j], 1)
        del a, m
        print(f"  {y}: lit_px={int(year_px[j][1:].sum())}", flush=True)

    # ---- 逐年"所属最大连片亮区"规模(第三道门限) ----
    # 8 邻域连通, 阈值 5 nW; 记该县所有亮像元所在连通块的最大尺寸
    from scipy import ndimage
    year_clu = np.zeros((len(years), N), dtype="int64")
    struct8 = np.ones((3, 3), dtype=bool)
    for j, y in enumerate(years):
        with rasterio.open(os.path.join(PROC, f"cn_{y}.tif")) as ds:
            lit = ds.read(1) >= CLU_TH
        lbl, n = ndimage.label(lit, structure=struct8)
        sizes = np.bincount(lbl.ravel())
        cs = sizes[lbl.ravel()]
        m = (idx > 0) & lit.ravel()
        big = np.zeros(N, dtype="int64")
        np.maximum.at(big, idx[m], cs[m])
        year_clu[j] = big
        del lit, lbl, cs, m, sizes
        print(f"  {y}: cluster max={int(year_clu[j][1:].max())}", flush=True)

    pairs = list(itertools.combinations(years, 2))
    P = len(pairs)
    v_px = np.zeros((P, N), dtype="int64")
    v_dlog = np.zeros((P, N))
    v_ratio = np.zeros((P, N))
    v_meanL2 = np.zeros((P, N))
    v_bri = np.zeros((P, N))
    v_dim = np.zeros((P, N))

    cache = {}
    for p, (y1, y2) in enumerate(pairs):
        if y1 not in cache:
            with rasterio.open(os.path.join(PROC, f"cn_{y1}.tif")) as ds:
                cache[y1] = ds.read(1).ravel().astype("float32")
        if y2 not in cache:
            with rasterio.open(os.path.join(PROC, f"cn_{y2}.tif")) as ds:
                cache[y2] = ds.read(1).ravel().astype("float32")
        a, b = cache[y1], cache[y2]
        m = np.maximum(a, b) >= MIN_SIG
        im = idx[m]
        cnt = np.bincount(im, minlength=N)
        d = np.log10((b[m] + EPS) / (a[m] + EPS)).astype("float64")
        v_px[p] = cnt
        v_dlog[p] = np.bincount(im, weights=d, minlength=N) / np.maximum(cnt, 1)
        v_ratio[p] = (np.bincount(im, weights=b[m].astype("float64"), minlength=N) + 1e-9) / \
                     (np.bincount(im, weights=a[m].astype("float64"), minlength=N) + 1e-9)
        v_meanL2[p] = np.bincount(im, weights=b[m].astype("float64"), minlength=N) / np.maximum(cnt, 1)
        v_bri[p] = np.bincount(im, weights=(d > DEAD).astype("float64"), minlength=N) / np.maximum(cnt, 1)
        v_dim[p] = np.bincount(im, weights=(d < -DEAD).astype("float64"), minlength=N) / np.maximum(cnt, 1)
        del d, im, m
        print(f"  {y1}->{y2} done", flush=True)
        if len(cache) > 2:
            cache.clear()

    def tier(px, ml, cl):
        if px >= 1000 and ml >= 5 and cl >= 2000:
            return 2
        if px >= 200 and ml >= 3 and cl >= 500:
            return 1
        return 0

    # ---- CSV ----
    os.makedirs(OUT, exist_ok=True)
    p_csv = os.path.join(OUT, "district_stats.csv")
    with open(p_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["adcode", "name", "province", "base_year", "cmp_year",
                    "valid_px", "mean_log_delta", "sum_ratio", "mean_L_cmp",
                    "max_cluster_px", "brighter_pct", "dimmer_pct",
                    "lit_px_base", "lit_px_cmp", "sum_base", "sum_cmp", "confidence"])
        for i, m in enumerate(meta):
            k = i + 1
            for p, (y1, y2) in enumerate(pairs):
                j1, j2 = years.index(y1), years.index(y2)
                t = tier(v_px[p][k], v_meanL2[p][k], year_clu[j2][k])
                w.writerow([m["a"], m["n"], m["p"], y1, y2,
                            int(v_px[p][k]),                             round(float(v_dlog[p][k]), 4),
                            round(float(v_ratio[p][k]), 4), round(float(v_meanL2[p][k]), 3),
                            int(year_clu[j2][k]),
                            round(float(v_bri[p][k]) * 100, 2), round(float(v_dim[p][k]) * 100, 2),
                            int(year_px[j1][k]), int(year_px[j2][k]),
                            round(float(year_sum[j1][k]), 1), round(float(year_sum[j2][k]), 1),
                            ["low", "mid", "high"][t]])
    print("wrote", p_csv, flush=True)

    # ---- 前端精简 JSON ----
    units = []
    for i, m in enumerate(meta):
        k = i + 1
        units.append({
            "a": m["a"], "n": m["n"], "p": m["p"], "c": m["c"],
            "px": [int(year_px[j][k]) for j in range(len(years))],
            "k": [int(year_clu[j][k]) for j in range(len(years))],
            "v": [int(v_px[p][k]) for p in range(P)],
            "d": [round(float(v_dlog[p][k]), 3) for p in range(P)],
            "r": [round(float(v_ratio[p][k]), 3) for p in range(P)],
            "m": [round(float(v_meanL2[p][k]), 2) for p in range(P)],
        })
    web_json = {"years": years, "pairs": [list(p) for p in pairs],
                "min_sig": MIN_SIG, "units": units,
                "tier_rule": {"high": [1000, 5, 2000], "mid": [200, 3, 500]}}
    p_web = os.path.join(OUT, "district_stats.json")
    with open(p_web, "w") as fh:
        json.dump(web_json, fh, ensure_ascii=False)
    ok = sum(1 for u in units for p in range(P)
             if tier(u["v"][p], u["m"][p], u["k"][years.index(pairs[p][1])]) > 0)
    print(f"wrote {p_web} {os.path.getsize(p_web)/1048576:.2f}MB")
    print(f"样本达标(tier>0) 单元·年对 = {ok} / {len(units)*P}")
    print("total", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
