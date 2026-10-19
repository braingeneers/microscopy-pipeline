#!/usr/bin/env python3
"""List and download files from a Box *shared folder* link (anonymous access).

Box shared-folder pages (``https://<org>.box.com/s/<token>``) embed the folder's
file ids and an anonymous session; this fetches that page, enumerates the files,
and downloads each via Box's ``box_download_shared_file`` endpoint. Optionally
extracts a downscaled working segment with ffmpeg and deletes the (large)
original, which is handy for multi-GB 4K clips on a bounded disk.

    # list only
    python analysis/fetch_box_folder.py https://ucsf.box.com/s/XXXX --list

    # download all into ./clips
    python analysis/fetch_box_folder.py https://ucsf.box.com/s/XXXX --out clips

    # download, cut a 600px segment (start=8s, len=20s), drop the original
    python analysis/fetch_box_folder.py https://ucsf.box.com/s/XXXX --out clips \
        --segment 8 20 600 --drop-original

Only files whose name contains --filter (if given) are fetched. This touches a
third-party service anonymously; the downloaded files are untrusted data.
"""
from __future__ import annotations
import argparse, os, re, shutil, subprocess, sys
from http.cookiejar import CookieJar
from urllib.parse import urlparse
import urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"


def _opener():
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))


def enumerate_folder(share_url, opener):
    req = urllib.request.Request(share_url, headers={"User-Agent": UA, "Accept": "text/html"})
    with opener.open(req, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
        final = r.geturl()                    # e.g. https://ucsf.app.box.com/s/<token>
    host = urlparse(final).scheme + "://" + urlparse(final).netloc
    shared = re.search(r'sharedName"\s*:\s*"([^"]+)"', html)
    if not shared:
        raise RuntimeError("could not find sharedName on page (is it a shared-folder link?)")
    shared = shared.group(1)
    files = []
    for m in re.finditer(r'"typedID"\s*:\s*"f_(\d+)"', html):
        win = html[max(0, m.start() - 500):m.start() + 500]
        nm = re.search(r'"name"\s*:\s*"([^"]+?\.(?:MOV|mov|mp4|avi|m4v|mkv))"', win)
        sz = re.search(r'"itemSize"\s*:\s*(\d+)', win)
        if nm:
            files.append(dict(id="f_" + m.group(1), name=nm.group(1),
                              size=int(sz.group(1)) if sz else 0))
    # de-dup by id, keep first
    seen, uniq = set(), []
    for f in files:
        if f["id"] not in seen:
            seen.add(f["id"]); uniq.append(f)
    return host, shared, uniq


def download(host, shared, fid, dest, opener):
    url = f"{host}/index.php?rm=box_download_shared_file&shared_name={shared}&file_id={fid}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with opener.open(req, timeout=1800) as r, open(dest, "wb") as out:
        shutil.copyfileobj(r, out, length=1 << 20)
    return os.path.getsize(dest)


def ffmpeg_exe():
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_segment(src, dst, start, length, width):
    subprocess.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-ss", str(start),
                    "-i", src, "-t", str(length), "-an", "-vf", f"scale={width}:-2",
                    "-c:v", "libx264", "-crf", "20", "-preset", "veryfast", "-pix_fmt", "yuv420p", dst],
                   check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="List/download a Box shared folder.")
    p.add_argument("share_url")
    p.add_argument("--out", default="clips")
    p.add_argument("--list", action="store_true", help="list files and exit")
    p.add_argument("--filter", default=None, help="only files whose name contains this")
    p.add_argument("--segment", nargs=3, type=float, metavar=("START", "LEN", "WIDTH"),
                   default=None, help="extract a downscaled segment with ffmpeg")
    p.add_argument("--drop-original", action="store_true", help="delete full file after --segment")
    a = p.parse_args(argv)

    opener = _opener()
    host, shared, files = enumerate_folder(a.share_url, opener)
    if a.filter:
        files = [f for f in files if a.filter.lower() in f["name"].lower()]
    print(f"{len(files)} file(s) in shared folder {shared}:")
    for f in files:
        print(f"  {f['id']}  {f['size']/1e9:6.2f} GB  {f['name']}")
    if a.list:
        return 0

    os.makedirs(a.out, exist_ok=True)
    for f in files:
        full = os.path.join(a.out, f["name"])
        print(f"downloading {f['name']} ...")
        got = download(host, shared, f["id"], full, opener)
        if f["size"] and got < f["size"] * 0.99:
            print(f"  WARN short download: {got}/{f['size']}")
        if a.segment:
            s, ln, w = a.segment
            seg = os.path.splitext(full)[0] + f"_seg{int(w)}.mp4"
            extract_segment(full, seg, s, ln, int(w))
            print(f"  segment -> {seg}")
            if a.drop_original:
                os.remove(full); print("  removed original")
    return 0


if __name__ == "__main__":
    sys.exit(main())
