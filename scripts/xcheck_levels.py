#!/usr/bin/env python3
"""两级口径数值互校：省级统计 ↔ 县级聚合到省 ↔ 已发布省级产物。

为什么需要它
------------
zonal.py(省级) 与 zonal_district.py(县级) 是两段独立代码, 靠"口径注释相同"来保证一致,
从未做过数值层面的交叉验证。一旦 bincount 或 mask 写错, 两份统计会静默背离。

注意不能拿两个 CSV 直接对减: 县级 district_stats.csv 的 sum_base/sum_cmp 是**单年**
mask(L >= MIN_SIG) 下的总量, 而省级 sum_ratio 是**年对** mask(max(L1,L2) >= MIN_SIG)
下的总量, 两者像元集合不同, 相减得到的差异毫无意义。

因此本脚本在同一份 mask 下重算三层:

  A = output/province_stats.csv         已发布省级产物(zonal.py, geometry_mask 逐省开窗)
  B = 省级重算(rasterize 省界 + bincount)
  C = 县级聚合(rasterize 县界 + 按省归并 + bincount)

A↔B 检验"方法差异"(geometry_mask vs rasterize, 二者同为像元中心归属规则, 应高度吻合);
B↔C 检验"边界几何差异"(省界 vs 2875 县界的并集, 是县级口径真正的系统偏差)。

输出: output/level_xcheck.csv + 控制台摘要
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
RAW = os.path.join(ROOT, "data", "raw")

MIN_SIG = 1.5
EPS = 0.1
WEST, EAST, SOUTH, NORTH = 73.0, 136.0, 3.0, 54.0
H, W = 6120, 7560
TR = from_bounds(WEST, SOUTH, EAST, NORTH, W, H)

# 判定容差(超出即需人工判读, 不意味一定错了)
TOL_PX = 0.01      # 有效像元数相对偏差
TOL_RATIO = 0.005  # 总亮度比相对偏差
TOL_DLOG = 0.005   # 平均 Δlog 绝对偏差(dex)


def main():
    t0 = time.time()
    years = sorted(int(f[3:7]) for f in os.listdir(PROC)
                   if f.startswith("cn_") and f.endswith(".tif") and "cogref" not in f)
    pairs = list(itertools.combinations(years, 2))
    print("years:", years, "pairs:", len(pairs), flush=True)

    # ---- 省级几何 ----
    bd = json.load(open(os.path.join(RAW, "china_boundary.json")))
    pnames, pgeoms = [], []
    for f in bd["features"]:
        name = f["properties"].get("name", "")
        if not name:            # 九段线, 非行政区
            continue
        pnames.append(name)
        pgeoms.append(shp(f["geometry"]))
    NP = len(pnames)
    print("provinces:", NP, flush=True)

    # ---- 县级几何 + 县→省 映射 ----
    web = json.load(open(os.path.join(RAW, "districts_web.json")))
    meta = web["meta"]
    full = json.load(open(os.path.join(RAW, "districts_full.json")))
    by_code = {f["properties"]["adcode"]: f for f in full["features"]}
    pidx = {n: i + 1 for i, n in enumerate(pnames)}
    dshapes, prov_of = [], np.zeros(len(meta) + 1, dtype="int64")
    miss = []
    for i, m in enumerate(meta):
        c = m["a"]
        if c not in by_code:
            miss.append(c); continue
        dshapes.append((shp(by_code[c]["geometry"]), i + 1))
        j = pidx.get(m["p"], 0)
        if j == 0:
            miss.append(f"{c}(省名不匹配:{m['p']})")
        prov_of[i + 1] = j
    ND = len(meta)
    print(f"districts: {ND}, 未匹配/缺几何: {len(miss)} {miss[:5]}", flush=True)
    del full, by_code

    # ---- 两套标签栅格 ----
    plab = rasterize([(g, i + 1) for i, g in enumerate(pgeoms)], out_shape=(H, W),
                     transform=TR, fill=0, dtype="uint16", all_touched=False).ravel()
    dlab = rasterize(dshapes, out_shape=(H, W), transform=TR, fill=0,
                     dtype="uint16", all_touched=False).ravel()
    print(f"rasterize {time.time()-t0:.1f}s", flush=True)

    p_in = plab > 0
    d_in = dlab > 0
    print(f"省界内像元={int(p_in.sum())}, 县界内像元={int(d_in.sum())}")
    print(f"省界内但无县标签={int((p_in & ~d_in).sum())}, 县标签落在省界外={int((d_in & ~p_in).sum())}")

    # ---- A: 已发布省级产物 ----
    A = {}
    for r in csv.DictReader(open(os.path.join(OUT, "province_stats.csv"))):
        A[(r["province"], int(r["base_year"]), int(r["cmp_year"]))] = {
            "px": int(r["valid_px"]),
            "dlog": float(r["mean_log_delta"]),
            "ratio": float(r["sum_ratio"]),
        }
    print("province_stats rows:", len(A), flush=True)

    # ---- B/C 逐年对计算 ----
    rows = []
    cache = {}
    for (y1, y2) in pairs:
        for y in (y1, y2):
            if y not in cache:
                with rasterio.open(os.path.join(PROC, f"cn_{y}.tif")) as ds:
                    cache[y] = ds.read(1).ravel().astype("float32")
        a, b = cache[y1], cache[y2]
        m = np.maximum(a, b) >= MIN_SIG
        d = np.log10((b[m] + EPS) / (a[m] + EPS)).astype("float64")
        av, bv = a[m].astype("float64"), b[m].astype("float64")

        for tag, lab in (("prov", plab), ("dist", np.where(dlab > 0, prov_of[dlab], 0))):
            li = lab[m]
            sel = li > 0
            cnt = np.bincount(li[sel], minlength=NP + 1)
            sA = np.bincount(li[sel], weights=av[sel], minlength=NP + 1)
            sB = np.bincount(li[sel], weights=bv[sel], minlength=NP + 1)
            sd = np.bincount(li[sel], weights=d[sel], minlength=NP + 1)
            for i, name in enumerate(pnames):
                k = i + 1
                if cnt[k] == 0:
                    continue
                rows.append({
                    "province": name, "base_year": y1, "cmp_year": y2, "level": tag,
                    "valid_px": int(cnt[k]),
                    "sum_ratio": float((sB[k] + 1e-9) / (sA[k] + 1e-9)),
                    "mean_log_delta": float(sd[k] / cnt[k]),
                })
        del d, av, bv, m
        if len(cache) > 2:
            cache.clear()
        print(f"  {y1}->{y2} done {time.time()-t0:.0f}s", flush=True)

    B = {(r["province"], r["base_year"], r["cmp_year"]): r for r in rows if r["level"] == "prov"}
    C = {(r["province"], r["base_year"], r["cmp_year"]): r for r in rows if r["level"] == "dist"}

    # ---- 对比 ----
    out = []
    for key in sorted(B.keys()):
        if key not in A or key not in C:
            print("  缺对:", key, "A", key in A, "C", key in C)
            continue
        a_, b_, c_ = A[key], B[key], C[key]
        rec = {
            "province": key[0], "base_year": key[1], "cmp_year": key[2],
            "px_A": a_["px"], "px_B": b_["valid_px"], "px_C": c_["valid_px"],
            "ratio_A": round(a_["ratio"], 4), "ratio_B": round(b_["sum_ratio"], 4),
            "ratio_C": round(c_["sum_ratio"], 4),
            "dlog_A": round(a_["dlog"], 4), "dlog_B": round(b_["mean_log_delta"], 4),
            "dlog_C": round(c_["mean_log_delta"], 4),
        }
        rec["px_AB"] = (rec["px_B"] - rec["px_A"]) / rec["px_A"]
        rec["px_BC"] = (rec["px_C"] - rec["px_B"]) / max(rec["px_B"], 1)
        rec["ratio_AB"] = (rec["ratio_B"] - rec["ratio_A"]) / rec["ratio_A"]
        rec["ratio_BC"] = (rec["ratio_C"] - rec["ratio_B"]) / rec["ratio_B"]
        rec["dlog_AB"] = rec["dlog_B"] - rec["dlog_A"]
        rec["dlog_BC"] = rec["dlog_C"] - rec["dlog_B"]
        out.append(rec)

    p_csv = os.path.join(OUT, "level_xcheck.csv")
    with open(p_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print("wrote", p_csv, len(out), "rows")

    # ---- 摘要 ----
    def stat(col, abs_=False):
        v = np.array([abs(r[col]) for r in out])
        return v.max(), np.median(v)

    print("\n=== 偏差摘要(|相对偏差|, 中位数 / 最大) ===")
    for col, tol in (("px_AB", TOL_PX), ("px_BC", TOL_PX),
                     ("ratio_AB", TOL_RATIO), ("ratio_BC", TOL_RATIO),
                     ("dlog_AB", TOL_DLOG), ("dlog_BC", TOL_DLOG)):
        mx, md = stat(col)
        flag = "OK " if mx <= tol else "超出容差"
        print(f"  {col:9s} 中位 {md:.5f}  最大 {mx:.5f}  容差 {tol}  {flag}")

    print("\n=== B↔C 偏差最大的 8 个省·年对 ===")
    worst = sorted(out, key=lambda r: -abs(r["px_BC"]))[:8]
    for r in worst:
        print(f"  {r['province']:12s} {r['base_year']}->{r['cmp_year']} "
              f"px {r['px_B']}→{r['px_C']} ({r['px_BC']*100:+.2f}%)  "
              f"ratio {r['ratio_B']}→{r['ratio_C']} ({r['ratio_BC']*100:+.2f}%)")

    prov_worst = {}
    for r in out:
        k = r["province"]
        if abs(r["px_BC"]) > abs(prov_worst.get(k, out[0])["px_BC"]):
            prov_worst[k] = r
    print("\n=== 各省 B↔C 像元偏差(全期 2021→2025 之外的最大年对) top 6 ===")
    for r in sorted(prov_worst.values(), key=lambda r: -abs(r["px_BC"]))[:6]:
        print(f"  {r['province']:12s} {r['base_year']}->{r['cmp_year']}  {r['px_BC']*100:+.2f}%")
    print("\ntotal", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
