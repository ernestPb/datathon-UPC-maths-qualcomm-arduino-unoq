#!/usr/bin/env python3
"""Prepare M3Kang for the Arduino UNO Q, on your laptop.

M3Kang stores each problem as text plus an image. The text has the question
but not the five answer options: those only appear in the image. This script:

1. downloads the parquet shards you ask for from Hugging Face,
2. keeps the languages you ask for,
3. reads the options A to E from each image with OCR (RapidOCR, offline),
4. splits the problems into train and validation by problem ID, so a
   problem and its translations never land on both sides,
5. writes small JSONL files (no images) that you copy to the board.

Changes against the original (better option reading):
  - column-based parser that fixes stacked fractions (1/2 no longer becomes 12)
  - OCR retries on a cropped / upscaled / contrast-boosted image when the first read fails
  - options must be 5 non-empty, distinct values, or they stay empty
  - raw OCR boxes are cached, so you can improve the parser without rerunning OCR
  - options made only of numbers/symbols are copied between translations of the same id
    (use --extra-languages to read more translations just to borrow from them)

Usage:
    python3 laptop/prepare_m3kang.py                      # English and Catalan
    python3 laptop/prepare_m3kang.py --languages eng,cat --extra-languages esp,fra
    python3 laptop/prepare_m3kang.py --val-years 2024     # hold out a whole year
"""
import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

DATASET = "qualcomm/M3Kang"
SHARD = "data/{split}-{index:05d}-of-{total:05d}.parquet"
SHARDS = {"high_quality": 4, "standard": 10}
LETTERS = "ABCDE"
CACHE_NAME = ".ocr_cache_v2.jsonl"  # new format: raw boxes + final options


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="high_quality", choices=sorted(SHARDS))
    p.add_argument("--shards", default="0",
                   help="comma-separated shard numbers, or 'all'. Shard 0 of high_quality has "
                        "eng, cat, fra, esp, ind, deu, swh, vie, tur, zho, kor, arb, jpn, lit")
    p.add_argument("--parquet", nargs="*", help="use local parquet files instead of downloading")
    p.add_argument("--languages", default="eng,cat", help="M3Kang language codes, e.g. eng,cat,esp")
    p.add_argument("--extra-languages", default="",
                   help="read these languages too, only to borrow options from them; not written to the output")
    p.add_argument("--val-fraction", type=float, default=0.15, help="share of problem IDs held out")
    p.add_argument("--val-years", default="", help="hold out these years instead, e.g. 2023,2024")
    p.add_argument("--out", default="data", help="output folder")
    p.add_argument("--borrow-only", default="",
                   help="languages written to the output but NOT read with OCR: their options are copied "
                        "from the other translations of the same id (numbers and symbols only). "
                        "Halves the time, e.g. --languages eng,cat --borrow-only cat")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="OCR processes in parallel (default: half of the CPU cores)")
    p.add_argument("--no-ocr", action="store_true", help="skip OCR; options will be empty")
    p.add_argument("--no-retry", action="store_true", help="one OCR pass per image (faster, reads fewer options)")
    p.add_argument("--save-images", action="store_true", help="also save problem images as PNG")
    p.add_argument("--limit", type=int, default=0, help="only process the first N rows (for testing)")
    return p.parse_args()


def shard_paths(args):
    if args.parquet:
        return args.parquet
    from huggingface_hub import hf_hub_download
    total = SHARDS[args.split]
    indexes = range(total) if args.shards == "all" else [int(s) for s in args.shards.split(",")]
    paths = []
    for i in indexes:
        name = SHARD.format(split=args.split, index=i, total=total)
        print(f"Downloading {name} (cached after the first time)...", flush=True)
        paths.append(hf_hub_download(DATASET, name, repo_type="dataset"))
    return paths


# --- Reading the options from the image -------------------------------------
#
# OCR returns text boxes. The options are the last A..E sequence in reading
# order, usually on one row at the bottom. OCR noise we handle, seen on real
# M3Kang images: "A)21" (no space), "E16" (lost parenthesis), "C" and "13" in
# separate boxes, and ")768351" (parenthesis attached to the value).

MARKER = re.compile(r"^\s*([A-E])\s*([\)）\]:.]?)\s*(.*)$")
# Options made only of digits and math symbols are identical in every language.
LANG_FREE = re.compile(r"^[\d\s.,/+\-−×x*÷:()%^√π=<>]+$")


def is_valid(options):
    """Five options, none empty, all different."""
    if len(options) != 5 or set(options) != set(LETTERS):
        return False
    values = [v.strip() for v in options.values()]
    return all(values) and len(set(values)) == 5


def reading_order(boxes):
    """Sort OCR boxes into rows (top to bottom), then left to right."""
    items = []
    for box, text, _score in boxes:
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        items.append({"x": min(xs), "y": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys), "text": text})
    items.sort(key=lambda it: it["y"])
    rows, current = [], []
    for it in items:
        if current and abs(it["y"] - current[-1]["y"]) > 0.6 * max(it["h"], current[-1]["h"]):
            rows.append(current)
            current = []
        current.append(it)
    if current:
        rows.append(current)
    return [it for row in rows for it in sorted(row, key=lambda it: it["x"])]


def is_marker(text, letter, strict):
    m = MARKER.match(text)
    if not m or m.group(1) != letter:
        return None
    paren, rest = m.group(2), m.group(3)
    if paren:
        return rest
    if strict:
        return None
    # Without a parenthesis, only accept "C" alone or "E16"-style values,
    # so words such as "ADA" or figure labels with text are not markers.
    if rest == "" or re.match(r"^[-+−\d]", rest):
        return rest
    return None


def parse_options(boxes):
    """Row-based parser (original). Returns {"A": "12", ..., "E": "24"}, or {} when the options are not text."""
    items = reading_order(boxes)
    starts = [i for i, it in enumerate(items) if is_marker(it["text"], "A", strict=True) is not None]
    for start in reversed(starts):  # the options are the last A..E sequence
        options, current, i = {}, None, start
        expected = iter(LETTERS)
        letter = next(expected)
        while i < len(items):
            text = items[i]["text"]
            rest = is_marker(text, letter, strict=(letter == "A")) if letter else None
            if rest is not None:
                current = letter
                options[current] = rest
                letter = next(expected, None)
            elif current:
                options[current] = (options[current] + " " + text).strip()
            i += 1
        if len(options) == 5:
            cleaned = {k: v.lstrip(")）] ").strip() for k, v in options.items()}
            if all(cleaned.values()):
                return cleaned
            return {}  # five markers, but the options are pictures
    return {}


def _items(boxes):
    out = []
    for box, text, score in boxes:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append({"x": min(xs), "xr": max(xs), "y": (min(ys) + max(ys)) / 2,
                    "h": max(ys) - min(ys), "text": text.strip(), "score": float(score)})
    return out


def parse_options_columns(boxes):
    """Column-based parser: every box goes to the nearest marker on its left.

    Works when the five markers sit on one row (the common layout). Boxes stacked
    above/below a marker (fractions) stay in the same option and are joined as
    'numerator/denominator'. Returns {} when it does not apply, so the row-based
    parse_options runs as the fallback.
    """
    items = _items(boxes)
    markers = {}
    for it in items:
        m = MARKER.match(it["text"])
        if not m:
            continue
        letter, paren, rest = m.group(1), m.group(2), m.group(3)
        if not paren and rest and not re.match(r"^[-+−\d]", rest):
            continue
        # keep the candidate lowest in the image: the options are at the bottom
        if letter not in markers or it["y"] > markers[letter]["it"]["y"]:
            markers[letter] = {"it": it, "rest": rest}
    if set(markers) != set(LETTERS):
        return {}
    h = max(markers[l]["it"]["h"] for l in LETTERS) or 1
    ys = [markers[l]["it"]["y"] for l in LETTERS]
    if max(ys) - min(ys) > h:           # markers on two rows: let the row parser handle it
        return {}
    xs = [markers[l]["it"]["x"] for l in LETTERS]
    if xs != sorted(xs):                # A..E must run left to right
        return {}
    y0 = sum(ys) / 5
    marker_ids = {id(markers[l]["it"]) for l in LETTERS}
    parts = {l: [] for l in LETTERS}
    for l in LETTERS:
        rest = markers[l]["rest"].lstrip(")）] ")
        if rest:
            parts[l].append((markers[l]["it"]["y"], markers[l]["it"]["x"], rest))
    for it in items:
        if id(it) in marker_ids or abs(it["y"] - y0) > 2.5 * h:
            continue
        owner = None
        for l in LETTERS:               # nearest marker to the left (small tolerance)
            if markers[l]["it"]["x"] <= it["x"] + 0.5 * h:
                owner = l
        if owner:
            parts[owner].append((it["y"], it["x"], it["text"].lstrip(")）] ")))
    options = {}
    for l, ps in parts.items():
        if not ps:
            return {}
        levels = sorted({round(p[0] / (0.6 * h)) for p in ps})
        if len(levels) >= 2 and all(len(p[2]) <= 6 for p in ps):
            ps.sort(key=lambda p: (p[0], p[1]))       # stacked: top to bottom
            options[l] = "/".join(p[2] for p in ps)
        else:
            ps.sort(key=lambda p: p[1])               # one line: left to right
            options[l] = " ".join(p[2] for p in ps)
    return options if is_valid(options) else {}


PARSERS = [parse_options_columns, parse_options]


def best_parse(boxes):
    """First valid result from the parsers, or {}."""
    for parse in PARSERS:
        options = parse(boxes)
        if is_valid(options):
            return options
    return {}


def _variants(image_bytes):
    """Yield numpy images: original, bottom crop x2, full x2 with contrast."""
    import numpy as np
    from PIL import Image, ImageOps
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    yield np.array(img)
    w, h = img.size
    crop = img.crop((0, int(h * 0.55), w, h))
    yield np.array(crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS))
    gray = ImageOps.autocontrast(ImageOps.grayscale(img))
    yield np.array(gray.resize((w * 2, h * 2), Image.LANCZOS).convert("RGB"))


def ocr_with_retries(ocr, image_bytes):
    """Try each image variant; return (options, boxes of the first try).

    The first-try boxes are what we cache, so parser improvements can be
    re-applied without running the OCR again.
    """
    first_boxes = None
    for variant in _variants(image_bytes):
        boxes, _ = ocr(variant)
        boxes = boxes or []
        if first_boxes is None:
            first_boxes = boxes
        options = best_parse(boxes)
        if options:
            return options, first_boxes
    return {}, first_boxes or []


def serialize_boxes(boxes):
    return [[[[float(p[0]), float(p[1])] for p in box], str(text), float(score)]
            for box, text, score in boxes]


def fill_from_other_languages(rows):
    """Copy options between translations of the same problem id.

    Only when every option is language-free (numbers and symbols) and the label
    matches. Mutates rows; returns how many problems were filled.
    """
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r)
    filled = 0
    for group in by_id.values():
        donors = [r for r in group
                  if is_valid(r["options"]) and all(LANG_FREE.match(v) for v in r["options"].values())]
        if not donors:
            continue
        for r in group:
            if is_valid(r["options"]):
                continue
            for d in donors:
                if d["label"] == r["label"]:
                    r["options"] = dict(d["options"])
                    filled += 1
                    break
    return filled


_OCR = None


def _init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the main process


def ocr_task(task):
    """Read one image. Runs in a worker process (or in this one when --workers 1)."""
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    key, image, retry = task
    if retry:
        options, boxes = ocr_with_retries(_OCR, image)
    else:
        boxes, _ = _OCR(image)
        boxes = boxes or []
        options = best_parse(boxes)
    return {"key": key, "options": options, "boxes": serialize_boxes(boxes)}


def main():
    args = parse_args()
    import pyarrow.parquet as pq

    wanted_languages = set(args.languages.split(","))
    borrow_only = set(l for l in args.borrow_only.split(",") if l)
    languages = wanted_languages | set(l for l in args.extra_languages.split(",") if l)
    val_years = set(y for y in args.val_years.split(",") if y)
    os.makedirs(args.out, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

    cache_path = os.path.join(args.out, CACHE_NAME)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                row = json.loads(line)
                cache[row["key"]] = row

    # 1. Read the problems (and keep the images that still need OCR).
    records = []
    for path in shard_paths(args):
        table = pq.read_table(path, columns=["language"])
        wanted = [i for i, lang in enumerate(table.column(0).to_pylist()) if lang in languages]
        if not wanted:
            continue
        full = pq.read_table(path).take(wanted)
        for i in range(full.num_rows):
            if args.limit and len(records) >= args.limit:
                break
            r = {c: full.column(c)[i].as_py() for c in ("id", "language", "level", "text_only", "label", "text")}
            r["key"] = f"{r['id']}|{r['language']}"
            image = full.column("image")[i].as_py()["bytes"]
            if args.save_images:
                with open(os.path.join(args.out, "images", r["key"].replace("|", "_") + ".png"), "wb") as f:
                    f.write(image)
            needs_ocr = not args.no_ocr and r["language"] not in borrow_only and r["key"] not in cache
            r["image"] = image if needs_ocr else None
            records.append(r)

    # 2. OCR, in a fixed random order of problem ids, so that if you stop it
    #    (Ctrl+C) the problems already read are a fair sample of all levels.
    order = sorted({r["id"] for r in records})
    random.Random(0).shuffle(order)
    rank = {pid: n for n, pid in enumerate(order)}
    tasks = [(r["key"], r["image"], not args.no_retry)
             for r in sorted(records, key=lambda r: rank[r["id"]]) if r["image"] is not None]
    print(f"{len(records)} images: {len(tasks)} to read with OCR, "
          f"{sum(1 for r in records if r['key'] in cache)} already in the cache, "
          f"{sum(1 for r in records if r['language'] in borrow_only)} copied from other languages "
          f"({args.workers} worker{'s' if args.workers != 1 else ''}). Ctrl+C keeps what is read so far.",
          flush=True)

    t0 = time.time()
    cache_file = open(cache_path, "a")

    def done(n, res):
        cache[res["key"]] = res
        cache_file.write(json.dumps(res, ensure_ascii=False) + "\n")
        cache_file.flush()
        if n % 20 == 0 or n == len(tasks):
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(tasks)} images, {rate:.1f}/s, about {(len(tasks) - n) / rate / 60:.0f} min left",
                  flush=True)

    interrupted = False
    try:
        if args.workers > 1 and tasks:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
                try:
                    for n, res in enumerate(pool.map(ocr_task, tasks, chunksize=2), 1):
                        done(n, res)
                except KeyboardInterrupt:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
        else:
            for n, task in enumerate(tasks, 1):
                done(n, ocr_task(task))
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopped. Writing the problems read so far...", flush=True)
    cache_file.close()

    # 3. Build the rows. Problems not read yet (only after Ctrl+C) are left out entirely.
    rows, unread = [], set()
    for r in records:
        entry = cache.get(r["key"])
        if entry is not None:
            options = entry["options"]
            if entry.get("boxes"):  # re-apply the current parsers to the cached boxes
                options = best_parse(entry["boxes"]) or options
        elif r["language"] in borrow_only or args.no_ocr:
            options = {}
        else:
            unread.add(r["id"])
            continue
        year = r["id"].split("_")[1]
        if val_years:
            split = "val" if year in val_years else "train"
        else:
            bucket = int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 1000
            split = "val" if bucket < args.val_fraction * 1000 else "train"
        rows.append({
            "id": r["id"], "language": r["language"], "level": r["level"], "year": int(year),
            "text_only": r["text_only"], "text": r["text"].strip(), "options": options,
            "label": r["label"], "split": split,
        })
    if unread:
        rows = [r for r in rows if r["id"] not in unread]
        print(f"Left out {len(unread)} problems that were not read yet. Run it again to continue from the cache.")

    before = sum(1 for r in rows if r["options"])
    filled = fill_from_other_languages(rows)
    print(f"Copied options between translations for {filled} problems "
          f"({before} -> {before + filled} with options, all languages read)")

    rows = [r for r in rows if r["language"] in wanted_languages]  # drop borrow-only extra languages
    for split in ("train", "val"):
        out = os.path.join(args.out, f"{split}.jsonl")
        with open(out, "w") as f:
            for r in rows:
                if r["split"] == split:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "split"}, ensure_ascii=False) + "\n")

    report(rows, args.out)


def report(rows, out):
    if not rows:
        sys.exit("No rows matched. Check --languages and --shards.")
    by = Counter((r["split"], r["language"]) for r in rows)
    with_options = sum(1 for r in rows if r["options"])
    text_only = [r for r in rows if r["text_only"]]
    labels = Counter(r["label"] for r in rows if r["split"] == "train")
    top, count = labels.most_common(1)[0]
    print()
    print(f"Wrote {out}/train.jsonl and {out}/val.jsonl")
    for (split, lang), n in sorted(by.items()):
        print(f"  {split:5} {lang}: {n}")
    print(f"Options read from the image: {with_options}/{len(rows)} ({100 * with_options / len(rows):.0f}%)")
    if text_only:
        ok = sum(1 for r in text_only if r["options"])
        print(f"  of which text-only problems: {ok}/{len(text_only)} ({100 * ok / len(text_only):.0f}%)")
    print(f"Most common answer in train: {top} ({100 * count / sum(labels.values()):.1f}%). "
          f"Always answering {top} is the floor to beat.")


if __name__ == "__main__":
    main()
