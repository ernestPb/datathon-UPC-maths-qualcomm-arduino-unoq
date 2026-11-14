#!/usr/bin/env python3
"""Build trainer.db for the adaptive trainer, on your laptop.

Reads the JSONL files from prepare_m3kang.py and builds a map of the problems:
  - a topic for each problem (k-means on the embeddings of the English text)
  - a difficulty score and band (level + position of the problem in its contest)
  - the nearest neighbours of each problem, and some easier ones, precomputed
  - empty tables for the student profile, the log and the hint cache

Only the question text is needed, so it works with --no-ocr data. Run it again
when the OCR finishes: it fills the options and keeps the same topics.

Usage:
    python3 laptop/build_trainer_db.py --input data_text/train.jsonl data_text/val.jsonl
    python3 laptop/build_trainer_db.py --input data_text/*.jsonl --try-k 6,8,10,12,14
    python3 laptop/build_trainer_db.py --input data/*.jsonl --topics 10 --topic-names data/topics.json

Embeddings: bge-small-en-v1.5 via fastembed (pip install fastembed) is the default;
--embedder tfidf needs no model download and is lower quality.
"""
import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict

import numpy as np

SCHEMA = """
CREATE TABLE problems (
  id TEXT PRIMARY KEY, level INTEGER, year INTEGER, num INTEGER,
  topic INTEGER, difficulty REAL, band INTEGER, label TEXT, split TEXT);
CREATE TABLE problem_texts (
  id TEXT, language TEXT, text TEXT, options TEXT, has_options INTEGER,
  PRIMARY KEY (id, language));
CREATE TABLE neighbors (
  id TEXT, neighbor_id TEXT, sim REAL, delta REAL, same_topic INTEGER);
CREATE TABLE topics (topic INTEGER PRIMARY KEY, name TEXT, keywords TEXT, size INTEGER);
CREATE TABLE student_profile (student TEXT, topic INTEGER, est_band REAL, seen INTEGER, correct INTEGER,
  PRIMARY KEY (student, topic));
CREATE TABLE student_log (student TEXT, problem_id TEXT, correct INTEGER, hints INTEGER, seconds REAL, ts TEXT);
CREATE TABLE hints (problem_id TEXT, language TEXT, rung INTEGER, text TEXT, created TEXT,
  PRIMARY KEY (problem_id, language, rung));
CREATE INDEX idx_problems_cell ON problems (topic, band);
CREATE INDEX idx_neighbors_id ON neighbors (id, sim DESC);
CREATE INDEX idx_neighbors_easier ON neighbors (id, delta, sim DESC);
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", required=True, help="train.jsonl and val.jsonl (a file with 'val' in its name is the val split)")
    p.add_argument("--out", default="data/trainer.db")
    p.add_argument("--embedder", default="auto", choices=["auto", "bge", "tfidf"])
    p.add_argument("--topics", type=int, default=10, help="number of topics (k-means clusters)")
    p.add_argument("--try-k", default="", help="e.g. 6,8,10,12: print the silhouette score for each k and stop")
    p.add_argument("--bands", type=int, default=4, help="number of difficulty bands")
    p.add_argument("--neighbors", type=int, default=15, help="nearest neighbours kept per problem")
    p.add_argument("--easier", type=int, default=5, help="extra easier neighbours kept per problem")
    p.add_argument("--topic-names", default="", help="JSON file {topic id: name}; created with automatic names if missing")
    return p.parse_args()


def load_problems(paths):
    problems = {}
    for path in paths:
        split = "val" if "val" in os.path.basename(path) else "train"
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                p = problems.setdefault(r["id"], {
                    "id": r["id"], "level": r["level"], "year": r["year"],
                    "num": int(r["id"].rsplit("_", 1)[1]), "label": r["label"], "split": split, "texts": {}})
                p["texts"][r["language"]] = {"text": r["text"], "options": r.get("options") or {}}
    return [problems[k] for k in sorted(problems)]


def embedding_text(p):
    t = (p["texts"].get("eng") or next(iter(p["texts"].values())))["text"]
    return re.sub(r"^\s*\d+\.\s*", "", t)       # drop the problem number


def embed(texts, method):
    if method in ("auto", "bge"):
        try:
            from fastembed import TextEmbedding
            model = TextEmbedding("BAAI/bge-small-en-v1.5")
            E = np.array(list(model.embed(texts)), dtype=np.float32)
            print("Embeddings: bge-small-en-v1.5")
        except ImportError:
            if method == "bge":
                sys.exit("fastembed is not installed: pip install fastembed (or use --embedder tfidf)")
            print("WARNING: fastembed not installed, using TF-IDF (lower quality). pip install fastembed for bge-small.")
            method = "tfidf"
    if method == "tfidf":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        X = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit_transform(texts)
        E = TruncatedSVD(min(64, X.shape[1] - 1), random_state=0).fit_transform(X)
        print("Embeddings: TF-IDF + SVD")
    return E / np.linalg.norm(E, axis=1, keepdims=True).clip(1e-9)


def difficulty(problems, bands):
    """level + position in the contest (0 to 1). ASSUMPTION: later problems of a contest are harder."""
    maxnum = defaultdict(int)
    for p in problems:
        maxnum[p["level"]] = max(maxnum[p["level"]], p["num"])
    d = np.array([p["level"] + (p["num"] - 1) / max(1, maxnum[p["level"]] - 1) for p in problems])
    cuts = np.quantile(d, np.linspace(0, 1, bands + 1)[1:-1])
    return d, np.searchsorted(cuts, d, side="right")


def topic_keywords(texts, labels, k, top=6):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=3)
    X = vec.fit_transform(texts).toarray()
    terms, overall, out = np.array(vec.get_feature_names_out()), X.mean(axis=0), {}
    for t in range(k):
        mask = labels == t
        out[t] = ", ".join(terms[np.argsort(-(X[mask].mean(axis=0) - overall))[:top]]) if mask.any() else ""
    return out


def main():
    args = parse_args()
    problems = load_problems(args.input)
    if not problems:
        sys.exit("No problems found in the input files.")
    texts = [embedding_text(p) for p in problems]
    print(f"{len(problems)} problems, languages: {sorted({l for p in problems for l in p['texts']})}")
    E = embed(texts, args.embedder)
    from sklearn.cluster import KMeans

    if args.try_k:
        from sklearn.metrics import silhouette_score
        print("k   silhouette (higher is cleaner, but pick the k whose topics make sense)")
        for k in [int(x) for x in args.try_k.split(",")]:
            lab = KMeans(k, n_init=10, random_state=0).fit_predict(E)
            print(f"{k:<3} {silhouette_score(E, lab):.3f}")
        return

    k = args.topics
    labels = KMeans(k, n_init=10, random_state=0).fit_predict(E)
    keywords = topic_keywords(texts, labels, k)
    names = {}
    if args.topic_names and os.path.exists(args.topic_names):
        names = {int(a): b for a, b in json.load(open(args.topic_names, encoding="utf-8")).items()}
    elif args.topic_names:
        json.dump({str(t): keywords[t] for t in range(k)}, open(args.topic_names, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"Wrote {args.topic_names} with automatic names: edit them and run again.")
    d, band = difficulty(problems, args.bands)

    S = E @ E.T
    np.fill_diagonal(S, -1)
    ids = [p["id"] for p in problems]
    neighbors = []
    for i in range(len(problems)):
        order = np.argsort(-S[i])
        keep = {int(j) for j in order[:args.neighbors]}
        easier = np.where(d < d[i] - 0.25)[0]
        keep |= {int(j) for j in easier[np.argsort(-S[i][easier])][:args.easier]}
        for j in keep:
            neighbors.append((ids[i], ids[j], float(S[i][j]), float(d[j] - d[i]), int(labels[i] == labels[j])))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)
    db = sqlite3.connect(args.out)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO problems VALUES (?,?,?,?,?,?,?,?,?)", [
        (p["id"], p["level"], p["year"], p["num"], int(labels[i]), float(d[i]), int(band[i]), p["label"], p["split"])
        for i, p in enumerate(problems)])
    db.executemany("INSERT INTO problem_texts VALUES (?,?,?,?,?)", [
        (p["id"], lang, v["text"], json.dumps(v["options"], ensure_ascii=False), int(bool(v["options"])))
        for p in problems for lang, v in p["texts"].items()])
    db.executemany("INSERT INTO neighbors VALUES (?,?,?,?,?)", neighbors)
    sizes = Counter(labels.tolist())
    db.executemany("INSERT INTO topics VALUES (?,?,?,?)",
                   [(t, names.get(t, keywords[t]), keywords[t], sizes[t]) for t in range(k)])
    db.commit()
    report(db, k, args.bands)
    db.close()
    print(f"\nWrote {args.out}. Copy it to the board; do not commit it (it holds M3Kang text).")


def report(db, k, bands):
    print("\nTopics (check them by hand: do the problems in each one really go together?)")
    for t, name, size in db.execute("SELECT topic, name, size FROM topics"):
        ex = [r[0][:70].replace("\n", " ") for r in db.execute(
            "SELECT text FROM problem_texts pt JOIN problems p ON p.id = pt.id "
            "WHERE p.topic = ? AND pt.language = (SELECT MIN(language) FROM problem_texts WHERE id = p.id) LIMIT 2", (t,))]
        print(f"  {t:>2}  {size:>4} problems  {name}")
        for e in ex:
            print(f"        e.g. {e}")
    print("\nProblems per cell (rows = topic, columns = difficulty band; 0 = empty cell)")
    grid = {(t, b): n for t, b, n in db.execute("SELECT topic, band, COUNT(*) FROM problems GROUP BY topic, band")}
    print("      " + "".join(f"band {b:<3}" for b in range(bands)))
    for t in range(k):
        print(f"  {t:>2}  " + "".join(f"{grid.get((t, b), 0):<8}" for b in range(bands)))
    empty = sum(1 for t in range(k) for b in range(bands) if not grid.get((t, b)))
    print(f"Empty cells: {empty}/{k * bands}")
    total, same = db.execute("SELECT COUNT(*), SUM(same_topic) FROM neighbors").fetchone()
    no_easier = db.execute(
        "SELECT COUNT(*) FROM problems p WHERE NOT EXISTS "
        "(SELECT 1 FROM neighbors n WHERE n.id = p.id AND n.delta < 0)").fetchone()[0]
    print(f"Neighbours in the same topic: {100 * same / total:.0f}%")
    print(f"Problems with no easier neighbour: {no_easier} (expected for the easiest band)")
    with_opt = db.execute("SELECT COUNT(*) FROM problem_texts WHERE has_options = 1").fetchone()[0]
    all_texts = db.execute("SELECT COUNT(*) FROM problem_texts").fetchone()[0]
    print(f"Texts with options read: {with_opt}/{all_texts}")


if __name__ == "__main__":
    main()