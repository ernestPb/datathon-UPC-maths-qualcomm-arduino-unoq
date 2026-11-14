#!/usr/bin/env python3
"""Quick before/after test of the option reading on a small sample of M3Kang.

Runs the OCR once per image and compares, on the SAME images:
  original parser  vs  new parsers  vs  new + retries  vs  new + copy between languages.
Then prints examples to check by hand. Does not write train/val files.

Put it next to prepare_m3kang.py (laptop/) and run, from the repository folder:
    python3 laptop/compare_ocr.py                 # 80 problems x 2 languages
    python3 laptop/compare_ocr.py --n 40          # even quicker
    python3 laptop/compare_ocr.py --languages eng,cat,esp --n 60
"""
import argparse
import os
import sys
import time
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare_m3kang as p


def read_new(ocr, image, boxes):
    """New pipeline on one image. Returns (options, how) with how in columns/rows/retry/none."""
    if p.is_valid(p.parse_options_columns(boxes)):
        return p.parse_options_columns(boxes), "columns"
    if p.is_valid(p.parse_options(boxes)):
        return p.parse_options(boxes), "rows"
    variants = p._variants(image)
    next(variants)  # the original image was already read
    for variant in variants:
        b, _ = ocr(variant)
        options = p.best_parse(b or [])
        if options:
            return options, "retry"
    return {}, "none"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=80, help="problems (ids) to sample, spread over the dataset")
    ap.add_argument("--languages", default="eng,cat")
    ap.add_argument("--shards", default="0")
    ap.add_argument("--show", type=int, default=8, help="examples to print per category")
    ap.add_argument("--out", default="data/compare_images", help="where to save the images of the examples")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from rapidocr_onnxruntime import RapidOCR

    languages = args.languages.split(",")
    path = p.shard_paths(Namespace(parquet=None, split="high_quality", shards=args.shards))[0]
    meta = pq.read_table(path, columns=["id", "language"])
    ids, langs = meta.column("id").to_pylist(), meta.column("language").to_pylist()
    pool = sorted({i for i, l in zip(ids, langs) if l == languages[0]})
    step = max(1, len(pool) // args.n)
    chosen = set(pool[::step][:args.n])
    wanted = [k for k, (i, l) in enumerate(zip(ids, langs)) if l in languages and i in chosen]
    full = pq.read_table(path).take(wanted)
    print(f"{len(chosen)} problems x {len(languages)} languages = {full.num_rows} images")

    ocr = RapidOCR()
    rows, t0 = [], time.time()
    for k in range(full.num_rows):
        pid = full.column("id")[k].as_py()
        lang = full.column("language")[k].as_py()
        label = full.column("label")[k].as_py()
        image = full.column("image")[k].as_py()["bytes"]
        boxes, _ = ocr(image)
        boxes = boxes or []
        old = p.parse_options(boxes)                 # what the original script produced
        first = p.best_parse(boxes)                  # new parsers, no retries
        new, how = read_new(ocr, image, boxes)       # new parsers + retries
        rows.append({"id": pid, "language": lang, "label": label, "old": old, "first": first,
                     "options": new, "how": how, "image": image})
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{full.num_rows}  ({(time.time() - t0) / (k + 1):.1f} s per image)", flush=True)

    total = len(rows)
    n_old = sum(1 for r in rows if r["old"])
    n_old_valid = sum(1 for r in rows if p.is_valid(r["old"]))
    n_first = sum(1 for r in rows if r["first"])
    n_new = sum(1 for r in rows if r["options"])
    for r in rows:
        r["before_fill"] = bool(r["options"])
    filled = p.fill_from_other_languages(rows) if len(languages) > 1 else 0
    n_fill = sum(1 for r in rows if r["options"])

    def pct(x):
        return f"{x:>4}/{total}  ({100 * x / total:4.1f}%)"

    print("\n=== Options read ===")
    print(f"Original script:                {pct(n_old)}")
    print(f"  of which valid (5, distinct): {pct(n_old_valid)}")
    print(f"New parsers, no retries:        {pct(n_first)}")
    print(f"New + retries:                  {pct(n_new)}")
    if len(languages) > 1:
        print(f"New + copy between languages:   {pct(n_fill)}   ({filled} copied)")
    hows = {h: sum(1 for r in rows if r['how'] == h) for h in ("columns", "rows", "retry", "none")}
    print(f"Read by: columns {hows['columns']}, rows {hows['rows']}, retry {hows['retry']}, not read {hows['none']}")
    print(f"Time: {time.time() - t0:.0f} s for {total} images")

    # Examples to check by hand. A wrong reading is worse than none.
    newly = [r for r in rows if not r["old"] and r["options"]]
    changed = [r for r in rows if r["old"] and r["options"] and r["old"] != r["options"]]
    lost = [r for r in rows if r["old"] and p.is_valid(r["old"]) and not r["options"]]
    os.makedirs(args.out, exist_ok=True)

    def show(title, items):
        print(f"\n=== {title}: {len(items)} ===")
        for r in items[:args.show]:
            name = f"{r['id']}_{r['language']}.png"
            with open(os.path.join(args.out, name), "wb") as f:
                f.write(r["image"])
            print(f"- {name}  [{r['how']}]")
            if r["old"]:
                print(f"    before: {r['old']}")
            print(f"    now:    {r['options']}")

    show("NEW: not read before, read now", newly)
    show("CHANGED: read differently (is the new one right?)", changed)
    show("LOST: valid before, empty now (should be 0)", lost)
    print(f"\nImages of the examples are in {args.out}/. Open them and check the values by eye.")


if __name__ == "__main__":
    main()
