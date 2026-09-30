#!/usr/bin/env python3
"""生成单文件交互式差异地图 HTML（零外部依赖）。

界面模板独立于 scripts/template.html，便于单独迭代样式与交互；
本脚本只负责数据准备与占位符注入。
"""
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
RAW = os.path.join(ROOT, "data", "raw")

assets = json.load(open(os.path.join(OUT, "assets.json")))
stats = json.load(open(os.path.join(OUT, "province_stats.json")))
dweb = json.load(open(os.path.join(RAW, "districts_web.json")))     # 县级边界(紧凑编码)
dstat = json.load(open(os.path.join(OUT, "district_stats.json")))   # 县级统计

# ---- 县级波特尔等级（120″ 网格中位数）按 adcode 合并进 dstat.units ----
# 供排行表「等级」列与「暗夜筛选」使用。缺失时前端该列显示「—」，不影响其余功能。
_dbp = os.path.join(OUT, "district_bortle.json")
if os.path.exists(_dbp):
    _db = json.load(open(_dbp))
    _bmap = {u["a"]: u for u in _db["units"]}
    _n = 0
    for _u in dstat["units"]:
        _b = _bmap.get(_u["a"])
        if _b:
            _u["lv"], _u["ln"] = _b["lv"], _b["ln"]
            _n += 1
    dstat["bortle_years"] = _db["years"]
    dstat["bortle_min_px"] = _db["min_px"]
    print(f"district bortle merged: {_n}/{len(dstat['units'])} units, years={_db['years']}")
else:
    print("district bortle: (缺失) 排行表等级列将显示为 —")

# 波特尔等级图层见下方(需先确定 years, 故延后到 years 定义之后再注入)

# ---- 边界简化 ----
from shapely.geometry import shape as shp, mapping

bd = json.load(open(os.path.join(RAW, "china_boundary.json")))
provinces, nine_dash, geoms = [], [], []
for f in bd["features"]:
    name = f["properties"].get("name", "")
    g = shp(f["geometry"])
    if not name:   # 南海诸岛界线(九段线), DataV 以 MultiPolygon 细长条表示
        parts = list(getattr(g, "geoms", [g]))
        for part in parts:
            geom = part.exterior if part.geom_type == "Polygon" else part
            nine_dash.append([[round(x, 3), round(y, 3)] for x, y in geom.coords])
        continue
    geoms.append(g)
    g2 = g.simplify(0.02, preserve_topology=True)
    if g2.geom_type == "MultiPolygon":
        polys = [p for p in g2.geoms if p.area > 0.02]
    elif g2.area > 0.02:
        polys = [g2]
    else:
        polys = []
    rings = []
    for p in polys:
        rings.append([[round(x, 3), round(y, 3)] for x, y in p.exterior.coords])
    c = f["properties"].get("centroid") or f["properties"].get("center")
    provinces.append({"name": name, "rings": rings,
                      "c": [round(c[0], 2), round(c[1], 2)] if c else None})
print("provinces:", len(provinces), "nine-dash parts:", len(nine_dash))

years = [int(y) for y in assets["years"]]

# ---- 波特尔等级图层（传播积分版优先，多年）----
# prop 版已计 300 km 内城市天空辉光，是主口径；缺失该年时回退查表版 bortle_{y}.json。
# 前端拿到 {years, w, h, levels, mags, method, layers:{year: png_b64}}，悬停时按年解码。
bortle, _bort_layers = {}, {}
for _y in years:
    _pp = os.path.join(OUT, f"bortle_prop_{_y}.json")
    _lp = os.path.join(OUT, f"bortle_{_y}.json")
    _src = _pp if os.path.exists(_pp) else (_lp if os.path.exists(_lp) else None)
    if _src is None:
        continue
    _d = json.load(open(_src))
    _bort_layers[_y] = _d["png"]
    if not bortle:
        bortle = {k: v for k, v in _d.items() if k not in ("png", "year")}
        bortle["method"] = _d.get("method", "prop")
        bortle["years"] = []
    bortle["years"].append(_y)
if bortle:
    bortle["layers"] = _bort_layers
print("bortle layers:", bortle.get("years"), bortle.get("method"),
      f"{bortle.get('w')}x{bortle.get('h')}" if bortle else "")


# ---- 中国境内 mask：与渲染网格严格对齐 (73-136E / 3-54N, 1/120度, 1890x1530) ----
import io as _io, base64 as _b64
from rasterio.transform import from_origin as _fo
from rasterio.features import rasterize as _ras
from PIL import Image as _Im
_tr = _fo(73.0, 54.0, 1.0/120.0, 1.0/120.0)
_m = _ras(geoms, out_shape=(6120, 7560), transform=_tr, fill=0,
          default_value=255, dtype="uint8", all_touched=False)
_m = _m.reshape(1530, 4, 1890, 4).mean(axis=(1, 3)).astype("uint8")
_buf = _io.BytesIO(); _Im.fromarray(_m, mode="L").save(_buf, format="PNG", optimize=True)
MASK_B64 = _b64.b64encode(_buf.getvalue()).decode("ascii")
print("mask png:", round(_buf.getbuffer().nbytes/1024, 1), "KB")

payload = {
    "w": assets["width"], "h": assets["height"], "bbox": assets["bbox"],
    "years": years, "img": assets["layers"], "mask": MASK_B64,
}
dmeta = [{"a": m["a"], "n": m["n"], "p": m["p"], "c": m["c"], "b": m["b"]} for m in dweb["meta"]]
ctx = {
    "payload": json.dumps(payload, ensure_ascii=False),
    "stats": json.dumps(stats, ensure_ascii=False),
    "prov": json.dumps(provinces, ensure_ascii=False),
    "ninedash": json.dumps(nine_dash, ensure_ascii=False),
    "dgeo": dweb["geo"],
    "dmeta": json.dumps(dmeta, ensure_ascii=False),
    "dstat": json.dumps(dstat, ensure_ascii=False),
    "bortle": json.dumps(bortle, ensure_ascii=False),
}
print("districts:", len(dmeta), "geo b64:", round(len(dweb["geo"]) / 1048576, 2), "MB")

# ---- 注入模板 ----
TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template.html")
with open(TPL, encoding="utf-8") as f:
    HTML = f.read()

for k, v in ctx.items():
    HTML = HTML.replace("__" + k.upper() + "__", v)

os.makedirs(OUT, exist_ok=True)         # output/ 整个被删时也能直接重建
p = os.path.join(OUT, "中国夜间灯光变化地图.html")
with open(p, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", p, round(os.path.getsize(p) / 1048576, 2), "MB")
