#!/usr/bin/env python3
"""县级波特尔等级统计（供排行表「等级」列与「暗夜候选」筛选使用）。

口径
----
等级图层是 120″ 网格（1890×1530，由 bortle_prop.py 的传播积分产出）。
本脚本把县级边界 rasterize 到**同一 120″ 网格**，对每个县取该县覆盖像元的
**等级中位数**（0 = 存疑/境外，先剔除），并记录样本像元数。

为什么不像 district_stats.json 那样用 30″
------------------------------------------
传播积分的 mag 只在 120″ 网格上有值（FFT 卷积后降采样），30″ 上采样不增加信息，
只会让「样本数」虚高 16 倍而分辨率不变，反而误导。120″ 像元约 17 km²；
排行表本身只纳入有效像元 ≥200（30″）的达标县，其对应的 120″ 样本通常 ≥12 个，
中位数已足够稳定。小县（如上海黄浦区）样本不足时输出 0，前端显示「—」，
这与它本来就不进排行是同一件事，不额外损失信息。

输出: output/district_bortle.json  {years, min_px, units:[{a, lv:[每年中位等级], ln:[每年样本像元数]}]}
units 顺序与 districts_web.json 的 meta 一致，build_html.py 按 adcode 合并进 district_stats.json。
"""
import os, json, base64, io
import numpy as np
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import shape as shp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
RAW = os.path.join(ROOT, "data", "raw")

WEST, EAST, SOUTH, NORTH = 73.0, 136.0, 3.0, 54.0
BW, BH = 1890, 1530        # 与 bortle_prop.py 的 120″ 显示栅格一致
MIN_PX = 3                 # 少于该像元数视为样本不足 -> 等级记 0


def main():
    web = json.load(open(os.path.join(RAW, "districts_web.json")))
    meta = web["meta"]
    codes = [m["a"] for m in meta]
    N = len(meta) + 1

    years = sorted(int(f[len("bortle_prop_"):-len(".json")])
                   for f in os.listdir(OUT)
                   if f.startswith("bortle_prop_") and f.endswith(".json"))
    print("bortle years:", years)

    full = json.load(open(os.path.join(RAW, "districts_full.json")))
    by_code = {f["properties"]["adcode"]: f for f in full["features"]}
    shapes = [(shp(by_code[c]["geometry"]), i + 1) for i, c in enumerate(codes) if c in by_code]
    print(f"matched shapes: {len(shapes)} / {len(codes)}")
    tr = from_bounds(WEST, SOUTH, EAST, NORTH, BW, BH)
    lab = rasterize(shapes, out_shape=(BH, BW), transform=tr, fill=0,
                    dtype="uint16", all_touched=False)
    idx = lab.ravel()
    print("labeled px:", int((idx > 0).sum()))
    del lab, shapes, full, by_code

    lv_all, ln_all = [], []
    for y in years:
        d = json.load(open(os.path.join(OUT, f"bortle_prop_{y}.json")))
        a = np.array(Image.open(io.BytesIO(base64.b64decode(d["png"])))).ravel()
        ok = (idx > 0) & (a > 0)
        bc = np.bincount(idx[ok].astype("int64") * 10 + a[ok],
                         minlength=N * 10).reshape(N, 10)
        tot = bc.sum(axis=1)
        cum = np.cumsum(bc, axis=1)
        half = (tot + 1) // 2
        med = (cum < half[:, None]).sum(axis=1).astype("int64")   # 中位等级
        med[tot < MIN_PX] = 0                                     # 样本不足
        lv_all.append(med)
        ln_all.append(tot)
        print(f"  {y}: 有值县 {int((med > 0).sum())}, 样本不足 {int((tot < MIN_PX).sum())}")

    units = [{"a": meta[i]["a"],
              "lv": [int(lv_all[j][i + 1]) for j in range(len(years))],
              "ln": [int(ln_all[j][i + 1]) for j in range(len(years))]}
             for i in range(len(meta))]

    p = os.path.join(OUT, "district_bortle.json")
    json.dump({"years": years, "min_px": MIN_PX, "units": units},
              open(p, "w"), ensure_ascii=False)
    print("wrote", p, round(os.path.getsize(p) / 1024, 1), "KB")

    d = np.bincount(lv_all[-1], minlength=10)
    print(f"\n{years[-1]} 年县级等级分布（共 {len(meta)} 个县市区）：")
    for l in range(1, 10):
        if d[l]:
            print(f"  {l} 级 {int(d[l]):>5d} 县")
    print(f"  样本不足 {int(d[0]):>5d} 县（<{MIN_PX} 个 120″ 像元）")

    # 等级退化速览：首年→末年
    if len(years) >= 2:
        a0, a1 = lv_all[0], lv_all[-1]
        both = (a0 > 0) & (a1 > 0)
        print(f"\n{years[0]} → {years[-1]} 县级等级变化（{int(both.sum())} 个可比县）：")
        print(f"  退化(等级升高) {int(((a1 > a0) & both).sum()):>5d} 县")
        print(f"  不变          {int(((a1 == a0) & both).sum()):>5d} 县")
        print(f"  改善(等级降低) {int(((a1 < a0) & both).sum()):>5d} 县")


if __name__ == "__main__":
    main()
