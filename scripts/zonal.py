#!/usr/bin/env python3
"""省级分区统计：每省逐年夜光辐亮度与对数变化。

输出: output/province_stats.csv, output/province_stats.json
指标口径:
  有效像元 = max(L1, L2) >= MIN_SIG  (低于 VIIRS 有效检测限的暗区不参与变化统计)
  对数变化 = log10((L2 + EPS) / (L1 + EPS)), 单位 dex (1 dex = 10 倍)
"""
import os, json, itertools, csv
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, transform as win_transform
from shapely.geometry import shape as shp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "proc")
OUT = os.path.join(ROOT, "output")
BOUNDARY = os.path.join(ROOT, "data", "raw", "china_boundary.json")

MIN_SIG = 1.5     # nW/cm^2/sr 有效信号下限(抑制荒漠小基数伪信号)
EPS = 0.1         # nW/cm^2/sr 对数稳定项
DEAD = 0.05       # dex 判定为"实际变化"的阈值


def load_provinces():
    d = json.load(open(BOUNDARY))
    out = []
    for f in d["features"]:
        name = f["properties"].get("name", "")
        if not name:          # 最后一条为南海诸岛界线(九段线), 非行政区
            continue
        g = shp(f["geometry"])
        if g.is_empty:
            continue
        out.append((name, f["properties"]["adcode"], g))
    return out


def main():
    years = sorted(int(f[3:7]) for f in os.listdir(PROC) if f.startswith("cn_") and f.endswith(".tif") and "cogref" not in f)
    print("years:", years, flush=True)
    provs = load_provinces()
    print("provinces:", len(provs), flush=True)

    srcs = {y: rasterio.open(os.path.join(PROC, f"cn_{y}.tif")) for y in years}
    base = srcs[years[0]]
    gt = base.transform

    rows = []
    for name, adcode, geom in provs:
        win = from_bounds(*geom.bounds, transform=gt).round_offsets().round_lengths()
        wt = win_transform(win, gt)
        inside = geometry_mask([geom], out_shape=(int(win.height), int(win.width)),
                               transform=wt, invert=True, all_touched=False)
        n_px = int(inside.sum())
        if n_px == 0:
            print("  skip", name); continue
        vals = {}
        for y in years:
            a = srcs[y].read(1, window=win)
            vals[y] = a[inside].astype("float32")
        # 省级面积(像元)与逐年总量
        rec = {"province": name, "adcode": adcode, "pixels": n_px,
               "area_km2": round(n_px * 0.774, 1)}
        for y in years:
            v = vals[y]
            rec[f"lit_px_{y}"] = int((v >= MIN_SIG).sum())
            rec[f"sum_{y}"] = round(float(v[v >= MIN_SIG].sum()), 1)
        for y1, y2 in itertools.combinations(years, 2):
            a, b = vals[y1], vals[y2]
            m = np.maximum(a, b) >= MIN_SIG
            n = int(m.sum())
            if n < 50:
                continue
            d = np.log10((b[m] + EPS) / (a[m] + EPS))
            rows.append({
                "province": name, "adcode": adcode, "base_year": y1, "cmp_year": y2,
                "valid_px": n,
                "mean_log_delta": round(float(d.mean()), 4),
                "median_log_delta": round(float(np.median(d)), 4),
                "p10_log_delta": round(float(np.percentile(d, 10)), 4),
                "p90_log_delta": round(float(np.percentile(d, 90)), 4),
                "brighter_pct": round(float((d > DEAD).mean() * 100), 2),
                "dimmer_pct": round(float((d < -DEAD).mean() * 100), 2),
                "sum_ratio": round(float((b[m].sum() + 1e-9) / (a[m].sum() + 1e-9)), 4),
            })
        # 单省逐年概览(取主对比: 末年 vs 末年-3 或最早年)
        rows_rec = rec
        if False:
            pass
        print(f"  {name}: px={n_px} lit2025={rec.get('lit_px_'+str(years[-1]),'-')}", flush=True)
    for s in srcs.values():
        s.close()

    os.makedirs(OUT, exist_ok=True)
    p_csv = os.path.join(OUT, "province_stats.csv")
    with open(p_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "province_stats.json"), "w") as fh:
        json.dump({"rows": rows, "years": years}, fh, ensure_ascii=False)
    print("wrote", p_csv, len(rows), "rows")


if __name__ == "__main__":
    main()
