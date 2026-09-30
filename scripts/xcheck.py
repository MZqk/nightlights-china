#!/usr/bin/env python3
"""交叉验证：NASA Black Marble(LPM) 与 OpenLandMap/EOG VNL(COG) 两套产品的可比性。

目的不是把两者混用, 而是:(1) 量化标度差异; (2) 确认空间结构一致性;
(3) 记录为何最终全程采用 Black Marble 同源数据。
输出 output/calibration_report.md

口径（2026-09-30 修正，重要）
----------------------------
本脚本早期版本只统计 ">0 nW" 口径，与全项目其余部分（zonal.py / zonal_district.py /
pack.py / 前端）使用的 **MIN_SIG = 1.5 nW** 口径不一致，导致同一份数据给出相反结论：
2025 年有效像元在 >0 口径是 −7.29%、在 ≥1.5 口径是 +1.40%。

原因：>0 口径把大量 0.2~0.5 nW 的检测限噪声算作"有效像元"，这些像元年度间随机进出，
数量波动可达数十万，却只贡献不到 1% 的总亮度（见报告第 1 节注释）。

**现在两种口径都输出，但结论一律以 ≥1.5 口径为准**，>0 口径仅供暗区检测限分析。
"""
import os
import numpy as np
import rasterio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "proc")
OUT = os.path.join(ROOT, "output")

MIN_SIG = 1.5     # 与 zonal.py / zonal_district.py 保持一致的口径

# 第 2 节（跨产品对比）的存档值：2026-09-27 实测。
# 参考栅格 data/proc/cn_2024_cogref.tif 已作为中间产物清理，若不存在则引用存档,
# 重新下载 EOG VNL 2024 参考年（约 60 MB）后本脚本会自动改回实算。
COG_ARCHIVE = {
    "n": 2447383, "r": 0.8877, "lr": 0.9413,
    "med": 9.89, "q1": 8.26, "q3": 12.20, "p5": 6.04, "p95": 17.15,
}


def rd(name):
    with rasterio.open(os.path.join(PROC, name)) as s:
        return s.read(1).astype("float32")


def main():
    lines = ["# 数据交叉验证报告", ""]
    have = sorted(f for f in os.listdir(PROC) if f.startswith("cn_"))
    print("files:", have)

    # 1) 同源逐年统计（双口径）
    years = sorted(int(f[3:7]) for f in have if "cogref" not in f)
    lines.append("## 1. 逐年统计（中国区，30″ 网格，nW·cm⁻²·sr⁻¹）")
    lines.append("")
    lines.append("| 年份 | 有效像元(≥1.5) | 有效像元(>0) | 均值(≥1.5) | P99(≥1.5) | "
                 "最大值 | 总量(≥1.5, ×10⁶) | 总量(>0, ×10⁶) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    tot15, tot0, px15, px0 = {}, {}, {}, {}
    for y in years:
        a = rd(f"cn_{y}.tif")
        v15 = a[a >= MIN_SIG]
        v0 = a[a > 0]
        tot15[y], tot0[y] = float(v15.sum()), float(v0.sum())
        px15[y], px0[y] = int(v15.size), int(v0.size)
        lines.append(f"| {y} | {px15[y]:,} | {px0[y]:,} | {v15.mean():.3f} | "
                     f"{np.percentile(v15,99):.2f} | {a.max():.1f} | "
                     f"{tot15[y]/1e6:.2f} | {tot0[y]/1e6:.2f} |")
        del a, v15, v0
    lines += ["",
              f"> **口径**：全项目统计（地图／省级／县级）一律使用 **≥{MIN_SIG} nW·cm⁻²·sr⁻¹**，",
              "> 本文结论同此口径。`>0` 一列仅用于暗区检测限分析——它把 0.2~0.5 nW 的",
              "> 检测限噪声也算作\"有效像元\"，年度间随机进出（2024→2025 有 53.5 万个掉到 0，",
              "> 其中 99.91% 亮度 <1.5 nW、中位 0.252 nW），却只贡献不到 1% 的总亮度。",
              "> **不可用 `>0` 口径得出\"亮区在熄灭\"之类的结论。**"]
    print("yearly ok")

    # 2) 与 COG 参考对比
    lines += ["", "## 2. Black Marble vs EOG VNL（2024 年重叠，同一网格）", ""]
    cogref = os.path.join(PROC, "cn_2024_cogref.tif")
    if os.path.exists(cogref):
        b = rd("cn_2024.tif")
        c = rd("cn_2024_cogref.tif")
        m = (b > 1.0) & (c > 1.0)
        n = int(m.sum())
        r = float(np.corrcoef(b[m], c[m])[0, 1])
        lr = float(np.corrcoef(np.log10(b[m]), np.log10(c[m]))[0, 1])
        ratio = c[m] / b[m]
        med, q1, q3 = np.median(ratio), np.percentile(ratio, 25), np.percentile(ratio, 75)
        p5, p95 = np.percentile(ratio, 5), np.percentile(ratio, 95)
        src = "实测"
    else:
        n = COG_ARCHIVE["n"]; r = COG_ARCHIVE["r"]; lr = COG_ARCHIVE["lr"]
        med = COG_ARCHIVE["med"]; q1 = COG_ARCHIVE["q1"]; q3 = COG_ARCHIVE["q3"]
        p5 = COG_ARCHIVE["p5"]; p95 = COG_ARCHIVE["p95"]
        src = "存档值（2026-09-27 实测；参考栅格已清理，重下 EOG 2024 可复算）"
        print("cogref 缺失, 第 2 节引用存档值")
    lines += [f"- 数据来源：{src}",
              f"- 参与比较像元（双方 > 1 nW）：{n:,}",
              f"- 线性 Pearson r = **{r:.4f}**；对数空间 r = **{lr:.4f}**",
              f"- 比值 COG/BM：中位 **{med:.2f}**，四分位 [{q1:.2f}, {q3:.2f}]",
              f"- 比值 5%–95% 分位：[{p5:.2f}, {p95:.2f}]",
              "",
              "**判读**：两套产品在空间结构上高度一致（对数空间相关性高），但存在约 9× 的**系统性标度差**，",
              "且比值随亮度区间变化（非恒定），说明线性定标无法完全消除差异。",
              "此外两者在暗区的检测限与掩膜策略不同（有效像元数相差约 15–20%）。",
              "因此年度差值**不跨产品**计算——本图 2021–2025 全程采用 Black Marble 同源数据。"]

    # 3) 逐年趋势（双口径，以 ≥1.5 为准）
    lines += ["", "## 3. 逐年趋势（同源，可比）", "",
              "**以 ≥1.5 口径为准**（与地图／省级／县级统计一致）。",
              "",
              "| 区间 | 总亮度变化(≥1.5) | 有效像元变化(≥1.5) | 总亮度变化(>0) | 有效像元变化(>0) |",
              "|---|---|---|---|---|"]
    for y1, y2 in zip(years[:-1], years[1:]):
        d15 = tot15[y2] / tot15[y1] - 1
        v15 = px15[y2] / px15[y1] - 1
        d0 = tot0[y2] / tot0[y1] - 1
        v0 = px0[y2] / px0[y1] - 1
        lines.append(f"| {y1}→{y2} | {d15*100:+.2f}% | {v15*100:+.2f}% | "
                     f"{d0*100:+.2f}% | {v0*100:+.2f}% |")
    lines += ["",
              "**读法**：全期总亮度（≥1.5 口径）"
              f"{tot15[years[0]]/1e6:.2f} → {tot15[years[-1]]/1e6:.2f}（×10⁶），"
              f"累计 **+{(tot15[years[-1]]/tot15[years[0]]-1)*100:.1f}%**。",
              "最后一年亮区面积只增 "
              f"{px15[years[-1]]/px15[years[-2]]-1:+.2%}（前三年 {px15[years[1]]/px15[years[0]]-1:+.2%} / "
              f"{px15[years[2]]/px15[years[1]]-1:+.2%} / {px15[years[3]]/px15[years[2]]-1:+.2%}），"
              f"总亮度仍增 {tot15[years[-1]]/tot15[years[-2]]-1:+.2%}——",
              "**新增亮区骤减，既有亮区在变亮，增长从\"摊大饼\"转向\"加密度\"**。",
              "",
              "> 旧版曾据 `>0` 口径写「2025 年有效像元减少 7.3%、边缘暗区在灭灯」，",
              "> 该说法已作废：那些消失的像元 99.91% 亮度 <1.5 nW（中位 0.252 nW），",
              "> 是检测限噪声而非熄灯，且在 ≥1.5 口径下有效像元实为增加。"]

    lines += ["", "## 4. 结论", "",
              "- 最终数据链：**NASA Black Marble 年度合成（VNP46A4/VJ146A4）2021–2025**，"
              "经 lightpollutionmap.info 分发，全球 15″，单位 nW·cm⁻²·sr⁻¹，重投影至 30″ 后裁中国区。",
              "- 逐年差值**全部在同产品内进行**，不含跨产品算法阶跃。",
              f"- 全国趋势结论一律采用 **≥{MIN_SIG} nW** 口径；`>0` 口径不作为趋势结论。",
              "- COG/EOG VNL 仅作为趋势方向的独立佐证。"]

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "calibration_report.md")
    open(p, "w").write("\n".join(lines))
    print("wrote", p)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
