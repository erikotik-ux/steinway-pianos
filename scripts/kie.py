#!/usr/bin/env python
"""
kie.ai image / video generation helper — shared across brand-site builds.

API key resolution order:
  1. KIE_API_KEY environment variable
  2. global key file  C:\\Users\\eriko\\.kie\\key
  3. ./.env.local      (line `KIE_API_KEY=...`)

kie.ai is async: createTask -> poll recordInfo -> download result into --out.
Generated media on kie.ai is deleted after 14 days, so always save + commit it.

Usage:
  python kie.py gen   --prompt "..." --ratio 2:3  --out public/story.jpg [--pro]
  python kie.py video --prompt "..." --ratio 16:9 --res 1080p --dur 5 --out public/hero.mp4
  python kie.py edit  --images public/product.jpg public/logo.png --prompt "..." --ratio 1:1 --out public/product.jpg

Notes:
  - Supported aspect ratios: 1:1, 16:9, 4:3, 2:3, 5:4 (use object-fit: cover in CSS).
  - `edit` uploads each --images file as a reference; reference them in the prompt
    ("the first image is the product, the second image is the logo ...").
"""
import argparse
import json
import os
import sys
import time

import requests

CREATE = "https://api.kie.ai/api/v1/jobs/createTask"
RECORD = "https://api.kie.ai/api/v1/jobs/recordInfo"
UPLOAD = "https://kieai.redpandaai.co/api/file-stream-upload"
GLOBAL_KEY = os.path.expanduser(r"~\.kie\key")


def load_key():
    if os.environ.get("KIE_API_KEY"):
        return os.environ["KIE_API_KEY"].strip()
    if os.path.exists(GLOBAL_KEY):
        with open(GLOBAL_KEY, encoding="utf-8") as f:
            k = f.read().strip()
            if k:
                return k
    if os.path.exists(".env.local"):
        with open(".env.local", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("KIE_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise SystemExit("No kie.ai key found (env / ~/.kie/key / .env.local).")


KEY = load_key()
AUTH = {"Authorization": "Bearer " + KEY}
JSON_HEADERS = {**AUTH, "Content-Type": "application/json"}


def upload(path):
    with open(path, "rb") as fh:
        r = requests.post(UPLOAD, headers=AUTH,
                          files={"file": (os.path.basename(path), fh, "application/octet-stream")},
                          data={"uploadPath": "brand-site", "fileName": os.path.basename(path)},
                          timeout=120)
    r.raise_for_status()
    d = r.json().get("data") or {}
    url = d.get("downloadUrl") or d.get("fileUrl")
    if not url:
        raise SystemExit("upload failed for %s: %s" % (path, r.text[:200]))
    print("  uploaded", os.path.basename(path), "->", url)
    return url


def create(model, inp):
    r = requests.post(CREATE, headers=JSON_HEADERS,
                      json={"model": model, "input": inp}, timeout=60)
    r.raise_for_status()
    return r.json()["data"]["taskId"]


def download(url, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    content = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=180).content
    with open(out, "wb") as f:
        f.write(content)
    print("  saved", out, "(%d bytes)" % len(content))


def run(model, inp, out, timeout):
    print("=== %s -> %s ===" % (model, out))
    tid = create(model, inp)
    print("  taskId:", tid)
    started = time.time()
    while time.time() - started < timeout:
        time.sleep(6)
        d = requests.get(RECORD, headers=AUTH, params={"taskId": tid}, timeout=60).json().get("data", {})
        state = str(d.get("state") or "").lower()
        print("  state: %s (%ds)" % (state or "?", int(time.time() - started)))
        if state in ("success", "succeeded", "completed", "done"):
            rj = d.get("resultJson")
            rj = json.loads(rj) if isinstance(rj, str) else (rj or {})
            urls = rj.get("resultUrls") or rj.get("urls") or []
            if not urls:
                raise SystemExit("success but no result urls: " + json.dumps(d)[:300])
            download(urls[0], out)
            return True
        if state in ("fail", "failed", "error"):
            raise SystemExit("generation failed: " + json.dumps(d)[:300])
    raise SystemExit("timed out")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--prompt", required=True)
    g.add_argument("--ratio", default="1:1")
    g.add_argument("--out", required=True)
    g.add_argument("--model", default=None)
    g.add_argument("--pro", action="store_true")

    s = sub.add_parser("sd")   # seedream v4: named sizes + 4K
    s.add_argument("--prompt", required=True)
    s.add_argument("--size", default="landscape_21_9")
    s.add_argument("--res", default="4K")
    s.add_argument("--out", required=True)
    s.add_argument("--images", nargs="*", default=None)

    v = sub.add_parser("video")
    v.add_argument("--prompt", required=True)
    v.add_argument("--ratio", default="16:9")
    v.add_argument("--res", default="1080p")
    v.add_argument("--dur", type=int, default=5)
    v.add_argument("--out", required=True)
    v.add_argument("--image", default=None)
    v.add_argument("--model", default="bytedance/seedance-2")

    e = sub.add_parser("edit")
    e.add_argument("--images", nargs="+", required=True)
    e.add_argument("--prompt", required=True)
    e.add_argument("--ratio", default="1:1")
    e.add_argument("--out", required=True)
    e.add_argument("--model", default="google/nano-banana-edit")

    a = ap.parse_args()
    if a.cmd == "gen":
        model = a.model or ("google/nano-banana-pro" if a.pro else "google/nano-banana")
        run(model, {"prompt": a.prompt, "output_format": "jpeg", "aspect_ratio": a.ratio},
            a.out, timeout=420)
    elif a.cmd == "sd":
        model = "bytedance/seedream-v4-edit" if a.images else "bytedance/seedream-v4-text-to-image"
        inp = {"prompt": a.prompt, "image_size": a.size, "image_resolution": a.res}
        if a.images:
            inp["image_urls"] = [upload(p) for p in a.images]
        run(model, inp, a.out, timeout=600)
    elif a.cmd == "video":
        inp = {"prompt": a.prompt, "resolution": a.res,
               "aspect_ratio": a.ratio, "duration": a.dur}
        if a.image:
            inp["image_urls"] = [upload(a.image)]
        run(a.model, inp, a.out, timeout=1200)
    elif a.cmd == "edit":
        urls = [upload(p) for p in a.images]
        run(a.model, {"image_urls": urls, "prompt": a.prompt,
                      "output_format": "jpeg", "image_size": a.ratio}, a.out, timeout=420)


if __name__ == "__main__":
    main()
