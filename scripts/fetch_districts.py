#!/usr/bin/env python3
"""抓取 DataV.GeoAtlas 县市区级边界（与现有省级边界同源，保证拓扑一致）。

策略: 100000_full(省级) -> {省adcode}_full(地级) -> {市adcode}_full(县级)
边界情况:
  - 直辖市/特别行政区: 省级 _full 直接返回 district 级, 不再下钻
  - 直筒子市(东莞/中山/儋州/嘉峪关等): 市级 _full 返回 404, 回退用该地级自身几何作为一个县级单元
  - 台湾省(710000): 市级 404, 回退为 1 个省级单元
  - 九段线 feature(adcode=100000_JD): 单独存为 nanhai.json, 不参与统计

输出: data/raw/districts_full.json
"""
import os, json, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT_FULL = os.path.join(RAW, "districts_full.json")
OUT_JD = os.path.join(RAW, "nanhai.json")
BASE = "https://geo.datav.aliyun.com/areas_v3/bound/{}_full.json"


def get(code, retry=3):
    for i in range(retry):
        try:
            with urllib.request.urlopen(BASE.format(code), timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == retry - 1:
                raise
            time.sleep(1.5 * (i + 1))
        except Exception:
            if i == retry - 1:
                raise
            time.sleep(1.5 * (i + 1))
    return None


def main():
    os.makedirs(RAW, exist_ok=True)
    top = get(100000)
    provs, jd = [], None
    for f in top["features"]:
        p = f["properties"]
        if str(p.get("adcode")) == "100000_JD" or not p.get("name"):
            jd = f
            continue
        provs.append(f)
    print(f"省级单元 {len(provs)}  九段线 {'有' if jd else '无'}", flush=True)

    # 第一层: 省级 -> 直接子级
    def fetch(pcode):
        return pcode, get(pcode)

    city_feats = {}      # adcode -> feature(地级)
    district_feats = {}  # adcode -> (feature, province_name, city_name)
    with ThreadPoolExecutor(max_workers=12) as ex:
        res = dict(ex.map(fetch, [f["properties"]["adcode"] for f in provs]))
    for pf in provs:
        pname = pf["properties"]["name"]
        children = res.get(pf["properties"]["adcode"])
        if not children:
            # 无子级(台湾省等): 省级自身作为一个统计单元
            district_feats[pf["properties"]["adcode"]] = (pf, pname, "")
            continue
        for cf in children["features"]:
            cp = cf["properties"]
            if cp.get("level") == "district":
                district_feats[cp["adcode"]] = (cf, pname, "")
            else:
                city_feats[cp["adcode"]] = (cf, pname)
    print(f"地级单元 {len(city_feats)}  已得县级 {len(district_feats)}", flush=True)

    # 第二层: 地级 -> 县级
    codes = list(city_feats)
    with ThreadPoolExecutor(max_workers=12) as ex:
        res2 = dict(ex.map(fetch, codes))
    fallback = []
    for code in codes:
        cf, pname = city_feats[code]
        cname = cf["properties"]["name"]
        sub = res2.get(code)
        if not sub:
            fallback.append(cname)
            district_feats[code] = (cf, pname, "")     # 直筒子市: 地级几何直接当县级用
            continue
        for df in sub["features"]:
            district_feats[df["properties"]["adcode"]] = (df, pname, cname)

    feats = []
    for adcode, (f, pname, cname) in sorted(district_feats.items()):
        p = f["properties"]
        feats.append({
            "type": "Feature",
            "properties": {"adcode": adcode, "name": p.get("name"),
                           "province": pname, "city": cname,
                           "center": p.get("center")},
            "geometry": f["geometry"],
        })
    fc = {"type": "FeatureCollection", "features": feats}
    with open(OUT_FULL, "w") as fh:
        json.dump(fc, fh, ensure_ascii=False)
    if jd:
        with open(OUT_JD, "w") as fh:
            json.dump(jd, fh, ensure_ascii=False)
    print(f"县级单元 {len(feats)}  回退(直筒子市/无下级) {len(fallback)}: {fallback}", flush=True)
    print("wrote", OUT_FULL, os.path.getsize(OUT_FULL) / 1048576, "MB", flush=True)


if __name__ == "__main__":
    main()
