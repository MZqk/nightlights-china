#!/usr/bin/env python3
"""下载 NASA Black Marble 年度合成原始包（经 lightpollutionmap.info 分发）。

用法:
    python scripts/download.py 2021 2022 2023 2024 2025
    python scripts/download.py 2025 --parts 16          # 16 段并行(默认)
    python scripts/download.py 2025 --keep-zip          # 保留 zip 不删
    python scripts/download.py 2025 --force             # 忽略已下载结果重下

设计要点(实测 2026-09):
  * 源站返回 Accept-Ranges: bytes, 支持分段并发; 2025 包 928 MiB。
  * 并行加速比实测线性: 单段 ~0.054 MiB/s, 8 段 0.44 MiB/s(8.2x)。
    由此外推 16 段 ≈0.9 MiB/s, 单年约 18 分钟, 五年 1.5 小时量级。
    (README 记的"45 分钟"是首次下载时的网络条件; 速度取决于出口与代理,
     若实测远低于此外推值, 先单跑一段测速再调 --parts, 别急着改脚本。)
  * 分段文件落在 data/raw/.dl/ 下, 中断后重跑自动续传(按已下载字节数续 Range)。
  * 仅用标准库(urllib / zipfile / concurrent.futures), 不引入新依赖。
    urllib 默认读取 http_proxy / https_proxy 环境变量, 代理环境可直接用。
  * 解压采用流式写出, 不在磁盘上同时存在 zip + 完整 tif 的两倍峰值之外的额外副本;
    解压完成后默认删除 zip(磁盘峰值约 3 GB/年, 与 README 一致)。

输出: data/raw/viirs_{YYYY}_raw.tif  (全球 86400x33600 float32, 15", EPSG:4326)
"""
import argparse, os, sys, time, zipfile, shutil
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
DL = os.path.join(RAW, ".dl")

BASE = "https://www2.lightpollutionmap.info/data/v2/viirs_{year}_raw.zip"
UA = {"User-Agent": "Mozilla/5.0 (lightpollution-cn pipeline)"}


def head(url, timeout=30):
    """返回 (content_length, accept_ranges)。失败抛异常。"""
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        length = int(r.headers.get("Content-Length", 0))
        return length, r.headers.get("Accept-Ranges", "")


def fetch_part(url, path, start, end, retries=3, timeout=60):
    """下载 [start, end] 闭区间字节到 path，支持断点续传。"""
    total = end - start + 1
    for attempt in range(1, retries + 1):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have > total:                      # 残留文件异常，重来
            os.remove(path); have = 0
        if have == total:
            return total
        try:
            hdr = dict(UA)
            hdr["Range"] = f"bytes={start + have}-{end}"
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=timeout) as r, open(path, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            got = os.path.getsize(path)
            if got == total:
                return total
            print(f"    part {start>>20}MiB: {got}/{total} 不完整, 重试 {attempt}/{retries}")
        except Exception as e:
            print(f"    part {start>>20}MiB: {type(e).__name__}: {e}, 重试 {attempt}/{retries}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"分段失败: {path} (bytes {start}-{end})")


def progress(paths, total, stop):
    """每 2 s 打印一次总进度，直到 stop.is_set()。"""
    while not stop.is_set():
        got = sum(os.path.getsize(p) if os.path.exists(p) else 0 for p in paths)
        pct = got / total * 100
        print(f"    {got/1048576:8.1f} / {total/1048576:.1f} MiB  ({pct:5.1f}%)", flush=True)
        time.sleep(2)


def download(url, out_zip, parts=16):
    os.makedirs(DL, exist_ok=True)
    length, ranges = head(url)
    if length <= 0:
        raise RuntimeError(f"HEAD 未返回长度: {url}")
    print(f"  size={length/1048576:.1f} MiB  accept-ranges={ranges or 'none'}")
    if "bytes" not in ranges.lower():
        parts = 1
        print("  源站不支持 Range, 退化为单线程")
    parts = max(1, min(parts, 32))
    step = (length + parts - 1) // parts
    spans = [(i * step, min(length - 1, (i + 1) * step - 1)) for i in range(parts)]
    tag = os.path.basename(out_zip)
    paths = [os.path.join(DL, f"{tag}.part{i:02d}") for i in range(parts)]

    import threading
    stop = threading.Event()
    t = threading.Thread(target=progress, args=(paths, length, stop), daemon=True)
    t.start()
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=parts) as ex:
            futs = [ex.submit(fetch_part, url, p, s, e) for p, (s, e) in zip(paths, spans)]
            for f in as_completed(futs):
                f.result()
    finally:
        stop.set(); t.join(timeout=3)
    print(f"  下载完成 {time.time()-t0:.0f}s")

    with open(out_zip, "wb") as out:
        for p in paths:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 20)
            os.remove(p)
    if os.path.getsize(out_zip) != length:
        raise RuntimeError(f"合并后大小不符: {os.path.getsize(out_zip)} != {length}")
    print(f"  ok {out_zip} ({os.path.getsize(out_zip)/1048576:.1f} MiB)")


def unzip(zip_path, year, keep_zip=False):
    """流式解压，主 tif 统一命名为 viirs_{year}_raw.tif。"""
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        tifs = [n for n in names if n.lower().endswith((".tif", ".tiff"))]
        if not tifs:
            raise RuntimeError(f"zip 内无 tif: {names[:10]}")
        main = max(tifs, key=lambda n: z.getinfo(n).file_size)
        print(f"  解压 {main} ({z.getinfo(main).file_size/1048576:.0f} MiB)")
        os.makedirs(RAW, exist_ok=True)
        dst = os.path.join(RAW, f"viirs_{year}_raw.tif")
        with z.open(main) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
        # 顺带放出小体积附属文件(aux.xml / tfw 等)
        for n in names:
            if n == main:
                continue
            if z.getinfo(n).file_size < 8 << 20:
                z.extract(n, RAW)
                print(f"    + {n}")
    print(f"  -> {dst} ({os.path.getsize(dst)/1048576:.0f} MiB)")
    if not keep_zip:
        os.remove(zip_path)
        print(f"  已删除 {os.path.basename(zip_path)} 释放磁盘")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("years", nargs="+", type=int)
    ap.add_argument("--parts", type=int, default=16)
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    for y in a.years:
        dst = os.path.join(RAW, f"viirs_{y}_raw.tif")
        if os.path.exists(dst) and not a.force:
            print(f"[{y}] 已存在 {dst} ({os.path.getsize(dst)/1048576:.0f} MiB), 跳过 (--force 强制重下)")
            continue
        url = BASE.format(year=y)
        zp = os.path.join(DL, f"viirs_{y}_raw.zip")
        print(f"[{y}] {url}")
        if not (os.path.exists(zp) and os.path.getsize(zp) > 0):
            download(url, zp, a.parts)
        else:
            print(f"  复用已下载的 zip ({os.path.getsize(zp)/1048576:.0f} MiB)")
        try:
            unzip(zp, y, a.keep_zip)
        except zipfile.BadZipFile:
            print("  zip 损坏(多半是断点续传拼接错误), 删除后重跑本命令即可", file=sys.stderr)
            os.remove(zp)
            raise


if __name__ == "__main__":
    main()
