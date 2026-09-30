#!/usr/bin/env python3
"""分省数据质量诊断：年度间"极暗像元"进出的空间分布是否均匀。

背景
----
2024→2025 全国 >0 像元减少 7.29%（5,686,855 → 5,272,133），但 ≥1.5 nW 口径反而 +1.40%。
已查明消失的 53.5 万个像元 99.91% 亮度 <1.5 nW、中位 0.252 nW，是检测限噪声而非熄灯。

还剩一个问题没回答：**这些噪声像元的消失是全局均匀的，还是集中在某几个省？**
- 若各省消失率接近 → 全局性的年度合成差异（检测限/掩膜），不影响任何省级结论。
- 若集中在少数省 → 区域性数据问题（云量、积雪、有效观测数），那几省的趋势需标注存疑。

本脚本按省统计"消失率"（上一年 >0、当年 =0 的像元占上一年 >0 像元的比例），
并与对照年比较，输出 output/data_quality.csv + 控制台摘要。

注：只统计省界内像元（省标签 >0），境外与海域不计。
"""
import os, json, csv, time
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.geometry import shape as shp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "proc")
OUT = os.path.join(ROOT, "output")
RAW = os.path.join(ROOT, "data", "raw")

WEST, EAST, SOUTH, NORTH = 73.0, 136.0, 3.0, 54.0
H, W = 6120, 7560
TR = from_bounds(WEST, SOUTH, EAST, NORTH, W, H)
MIN_SIG = 1.5


def main():
    t0 = time.time()
    years = sorted(int(f[3:7]) for f in os.listdir(PROC)
                   if f.startswith("cn_") and f.endswith(".tif") and "cogref" not in f)
    print("years:", years, flush=True)

    bd = json.load(open(os.path.join(RAW, "china_boundary.json")))
    pnames, pgeoms = [], []
    for f in bd["features"]:
        name = f["properties"].get("name", "")
        if not name:
            continue
        pnames.append(name)
        pgeoms.append(shp(f["geometry"]))
    NP = len(pnames)
    plab = rasterize([(g, i + 1) for i, g in enumerate(pgeoms)], out_shape=(H, W),
                     transform=TR, fill=0, dtype="uint16", all_touched=False).ravel()
    print(f"provinces: {NP}, rasterize {time.time()-t0:.1f}s", flush=True)

    rows = []
    cache = {}
    for y1, y2 in zip(years[:-1], years[1:]):
        for y in (y1, y2):
            if y not in cache:
                with rasterio.open(os.path.join(PROC, f"cn_{y}.tif")) as ds:
                    cache[y] = ds.read(1).ravel().astype("float32")
        a, b = cache[y1], cache[y2]
        a0, b0 = a > 0, b > 0
        lost = a0 & ~b0            # 上一年 >0、当年 =0
        new = (~a0) & b0           # 上一年 =0、当年 >0
        lit1 = a >= MIN_SIG
        lit2 = b >= MIN_SIG

        base = np.bincount(plab[a0], minlength=NP + 1)
        lost_n = np.bincount(plab[lost], minlength=NP + 1)
        new_n = np.bincount(plab[new], minlength=NP + 1)
        lit1_n = np.bincount(plab[lit1], minlength=NP + 1)
        lit2_n = np.bincount(plab[lit2], minlength=NP + 1)

        for i, name in enumerate(pnames):
            k = i + 1
            if base[k] == 0:
                continue
            rows.append({
                "province": name, "pair": f"{y1}->{y2}",
                "px_gt0_base": int(base[k]),
                "lost": int(lost_n[k]), "lost_rate": lost_n[k] / base[k],
                "new": int(new_n[k]), "new_rate": new_n[k] / base[k],
                "net_rate": (new_n[k] - lost_n[k]) / base[k],
                "lit_base": int(lit1_n[k]), "lit_cmp": int(lit2_n[k]),
                "lit_rate": (lit2_n[k] / lit1_n[k] - 1) if lit1_n[k] else float("nan"),
            })
        print(f"  {y1}->{y2} done {time.time()-t0:.0f}s", flush=True)
        if len(cache) > 2:
            cache.clear()

    p_csv = os.path.join(OUT, "data_quality.csv")
    with open(p_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", p_csv, len(rows), "rows")

    # ---- 摘要：只看目标年对 2024->2025 ----
    tgt = [r for r in rows if r["pair"] == f"{years[-2]}->{years[-1]}"]
    lr = np.array([r["lost_rate"] for r in tgt])
    print(f"\n=== {years[-2]}->{years[-1]} 各省「极暗像元消失率」分布（n={len(tgt)}）===")
    print(f"  中位 {np.median(lr):.2%}   四分位 [{np.percentile(lr,25):.2%}, {np.percentile(lr,75):.2%}]"
          f"   最小 {lr.min():.2%}   最大 {lr.max():.2%}")
    print(f"  极差 {lr.max()-lr.min():.2%}   标准差 {lr.std():.2%}")

    print("\n  消失率最高 6 省:")
    for r in sorted(tgt, key=lambda r: -r["lost_rate"])[:6]:
        ctrl = next((x for x in rows if x["province"] == r["province"]
                     and x["pair"] == f"{years[-3]}->{years[-2]}"), None)
        c = f"{ctrl['lost_rate']:.2%}" if ctrl else "—"
        print(f"    {r['province']:<12s} 消失 {r['lost']:>7,} / {r['px_gt0_base']:>7,} = {r['lost_rate']:>6.2%}"
              f"   对照年 {c}   亮区变化 {r['lit_rate']:+.2%}")
    print("\n  消失率最低 4 省:")
    for r in sorted(tgt, key=lambda r: r["lost_rate"])[:4]:
        print(f"    {r['province']:<12s} {r['lost_rate']:>6.2%}   亮区变化 {r['lit_rate']:+.2%}")

    # ---- 关键：与对照年基线比较，识别"异常升高"的省 ----
    # 绝对消失率高的省（西藏/新疆/青海）往往常年如此，只有相对基线升高才说明当年有区域性问题。
    ctrl_pair = f"{years[-3]}->{years[-2]}"
    ctrl = {r["province"]: r for r in rows if r["pair"] == ctrl_pair}
    deltas = []
    for r in tgt:
        c = ctrl.get(r["province"])
        if c:
            deltas.append((r["province"], r["lost_rate"] - c["lost_rate"], r, c))
    deltas.sort(key=lambda x: -x[1])

    print(f"\n=== 相对对照年（{ctrl_pair}）的变化：正值 = 当年消失率异常升高 ===")
    for name, d, r, c in deltas[:8]:
        print(f"  {name:<12s} {c['lost_rate']:>6.2%} → {r['lost_rate']:>6.2%}  ({d:+.2%})"
              f"   该省亮区(≥1.5)变化 {r['lit_rate']:+.2%}")

    print(f"\n=== 判读 ===")
    hi = [x for x in deltas if x[1] >= 0.05]
    flat = [x for x in deltas if abs(x[1]) < 0.05]
    print(f"  相对基线升高 ≥5 个百分点的省：{len(hi)} 个" +
          (f"（{'、'.join(x[0] for x in hi)}）" if hi else ""))
    print(f"  与基线基本持平的省：{len(flat)} / {len(deltas)}")
    if hi:
        # 不能只看"亮区像元数减少"就判存疑：像元数退出的多是边缘暗区，总亮度可能仍在增长。
        # 必须联合省级总亮度比(sum_ratio)一起判断。
        import csv as _csv
        pr = {}
        for r in _csv.DictReader(open(os.path.join(OUT, "province_stats.csv"))):
            if r["base_year"] == str(years[-2]) and r["cmp_year"] == str(years[-1]):
                pr[r["province"]] = float(r["sum_ratio"])
        print("  → 上列省份需联合省级总亮度比判断（像元数减少 ≠ 亮度下降）：")
        for name, d, r, c in hi:
            sr = pr.get(name)
            if sr is None:
                continue
            if sr >= 1.03:
                flag = "总亮度明确增长，结论不受影响"
            elif sr >= 0.98:
                flag = "总亮度接近停滞，当年变化在噪声量级，不宜单独解读"
            else:
                flag = "总亮度下降，需存疑"
            print(f"     {name:<12s} 亮区像元 {r['lit_rate']:+.2%}  总亮度比 ×{sr:.3f} — {flag}")
    else:
        print("  → 各省消失率与自身基线一致，属**全局性**年度合成差异（检测限/掩膜），")
        print("    不构成区域性数据问题，省级趋势结论不受影响。")
    print("\ntotal", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
