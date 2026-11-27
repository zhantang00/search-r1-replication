
from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
from glob import glob
from pathlib import Path

from config import (
    ASSETS_DIR, BAMBOOGLE_CSV, BAMBOOGLE_PATH, BASE_MODEL, CORPUS_FILE, CORPUS_PATH,
    CORPUS_REPO, DATA_DIR, DEBUG_MODEL, INDEX_PARTS, INDEX_PATH, INDEX_REPO,
    RETRIEVER_MODEL, SEARCH_TEMPLATE_PATH, TRAIN_DATA_REPO, TRAIN_PATH,
)

# Paper Table 1. Used only if the authors' exact template can't be recovered
# from their training data (see derive_search_template).
PAPER_SEARCH_TEMPLATE = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search>, and it will return "
    "the top searched results between <information> and </information>. You can search as many "
    "times as you want. If you find no further external knowledge needed, you can directly "
    "provide the answer inside <answer> and </answer> without detailed illustrations. For "
    "example, <answer> xxx </answer>. Question: {question}\n"
)


# --------------------------------------------------------------------------- #
# Small JSONL helpers (used by every other file)
# --------------------------------------------------------------------------- #
def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def normalize_question(question: str) -> str:
    """Match the authors' preprocessing: strip, and end with a question mark."""
    question = question.strip()
    if question and not question.endswith("?"):
        question += "?"
    return question


# --------------------------------------------------------------------------- #
# Bamboogle
# --------------------------------------------------------------------------- #
def read_bamboogle_csv(path: str = BAMBOOGLE_CSV) -> list[dict]:
    """Columns used: 'Question', 'Answer', and the optional 'Answer 2'."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            question = (row.get("Question") or "").strip()
            answers = [(row.get(k) or "").strip() for k in ("Answer", "Answer 2")]
            answers = [a for a in answers if a]
            if not question or not answers:
                continue
            rows.append({
                "id": f"bamboogle_{len(rows)}",
                "question": normalize_question(question),
                "golden_answers": answers,
                "data_source": "bamboogle",
            })
    return rows


# --------------------------------------------------------------------------- #
# Official NQ + HotpotQA data
# --------------------------------------------------------------------------- #
def _golden_answers(row: dict) -> list[str]:
    answers = row.get("golden_answers")
    if answers is None:
        answers = ((row.get("reward_model") or {}).get("ground_truth") or {}).get("target")
    if isinstance(answers, str):
        answers = [answers]
    return [str(a) for a in (answers or [])]


def _load_split_rows(root: str, split: str) -> list[dict]:
    """Read one split from a snapshot, whether stored as parquet or via save_to_disk."""
    files = sorted(p for p in glob(f"{root}/**/*.parquet", recursive=True)
                   if split in Path(p).name)
    if files:
        import pyarrow.parquet as pq
        rows: list[dict] = []
        for file in files:
            rows.extend(pq.read_table(file).to_pylist())
        return rows
    split_dir = Path(root) / split
    if (split_dir / "dataset_info.json").exists():
        from datasets import load_from_disk
        return load_from_disk(str(split_dir)).to_list()
    return []


def derive_search_template(rows: list[dict], n_check: int = 200) -> tuple[str, bool]:
    """Recover the authors' exact prompt template from their training rows."""
    try:
        first = rows[0]
        content = first["prompt"][0]["content"]
        template = content.replace(first["question"], "{question}", 1)
        if "{question}" not in template:
            return PAPER_SEARCH_TEMPLATE, False
        for row in rows[:n_check]:
            if template.replace("{question}", row["question"]) != row["prompt"][0]["content"]:
                return PAPER_SEARCH_TEMPLATE, False
        return template, True
    except (KeyError, IndexError, TypeError):
        return PAPER_SEARCH_TEMPLATE, False


def _to_record(row: dict, idx: int, split: str) -> dict:
    return {
        "id": str(row.get("id") or f"{split}_{idx}"),
        "question": normalize_question(str(row["question"])),
        "golden_answers": _golden_answers(row),
        "data_source": str(row.get("data_source", "")),
    }


def prepare_qa_data() -> dict:
    """Write train.jsonl, bamboogle.jsonl and search_r1_template.txt. Returns stats."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(TRAIN_DATA_REPO, repo_type="dataset",
                              local_dir=f"{ASSETS_DIR}/nq_hotpotqa_train")
    train_rows = _load_split_rows(local, "train")
    test_rows = _load_split_rows(local, "test")
    if not train_rows:
        raise RuntimeError(f"No training split found under {local}; inspect its layout.")

    train = [_to_record(r, i, "train") for i, r in enumerate(train_rows)]
    train = [r for r in train if r["golden_answers"]]
    write_jsonl(TRAIN_PATH, train)

    template, recovered = derive_search_template(train_rows)
    os.makedirs(DATA_DIR, exist_ok=True)
    Path(SEARCH_TEMPLATE_PATH).write_text(template, encoding="utf-8")

    bamboogle = read_bamboogle_csv()
    comparison = _compare_bamboogle(bamboogle, test_rows)
    bamboogle, n_replaced = _use_official_golds(bamboogle, test_rows)
    write_jsonl(BAMBOOGLE_PATH, bamboogle)

    stats = {
        "train_rows": len(train),
        "train_sources": _count(r["data_source"] for r in train),
        "bamboogle_rows": len(bamboogle),
        "template_recovered_from_official_data": recovered,
        "official_test_rows": len(test_rows),
        "bamboogle_vs_official": comparison,
        "bamboogle_golds_taken_from_official": n_replaced,
    }
    print(json.dumps(stats, indent=2))
    print("Search-R1 template:\n" + template)
    return stats


def _count(values) -> dict:
    counts: dict = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


def _compare_bamboogle(ours: list[dict], official_test_rows: list[dict]) -> dict:
    """Check our CSV-derived Bamboogle against the authors' test split, if it has one."""
    from scoring import normalize_answer

    official = {normalize_answer(normalize_question(str(r["question"]))): _golden_answers(r)
                for r in official_test_rows if r.get("data_source") == "bamboogle"}
    if not official:
        return {"official_bamboogle_found": False}
    missing, answer_diffs = [], []
    for row in ours:
        key = normalize_answer(row["question"])
        if key not in official:
            missing.append(row["question"])
        elif sorted(map(normalize_answer, official[key])) != sorted(map(normalize_answer, row["golden_answers"])):
            answer_diffs.append({"question": row["question"], "ours": row["golden_answers"],
                                 "official": official[key]})
    return {"official_bamboogle_found": True, "official_count": len(official),
            "not_in_official": len(missing), "answer_differences": answer_diffs[:10]}


def _use_official_golds(ours: list[dict], official_test_rows: list[dict]) -> tuple[list[dict], int]:
    """Keep the CSV's questions, but score against the authors' gold answers."""
    from scoring import normalize_answer

    official = {normalize_answer(normalize_question(str(r["question"]))): _golden_answers(r)
                for r in official_test_rows if r.get("data_source") == "bamboogle"}
    rows, replaced = [], 0
    for row in ours:
        golds = official.get(normalize_answer(row["question"]))
        if golds:
            replaced += 1
            row = {**row, "golden_answers": golds}
        rows.append(row)
    return rows, replaced


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def download_models(models: tuple[str, ...] = (BASE_MODEL, DEBUG_MODEL, RETRIEVER_MODEL)) -> None:
    """Cache model weights under HF_HOME (on the Volume) so later runs skip the download."""
    from huggingface_hub import snapshot_download

    for repo in models:
        path = snapshot_download(
            repo, ignore_patterns=["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot"])
        print(f"cached {repo} -> {path}")


# --------------------------------------------------------------------------- #
# Retrieval assets (corpus + E5 flat index)
# --------------------------------------------------------------------------- #
def download_retrieval_assets() -> None:
    """Download, join and decompress the authors' index and corpus. Safe to re-run."""
    from huggingface_hub import hf_hub_download

    os.makedirs(ASSETS_DIR, exist_ok=True)
    index_done = Path(ASSETS_DIR) / ".index_done"
    corpus_done = Path(ASSETS_DIR) / ".corpus_done"

    if not index_done.exists():
        parts_dir = f"{ASSETS_DIR}/index_parts"
        parts = [hf_hub_download(INDEX_REPO, part, repo_type="dataset", local_dir=parts_dir)
                 for part in INDEX_PARTS]
        _concatenate(parts, INDEX_PATH)
        shutil.rmtree(parts_dir, ignore_errors=True)
        index_done.touch()
    print(f"index ready: {INDEX_PATH} ({_gb(INDEX_PATH):.1f} GB)")

    if not corpus_done.exists():
        gz_dir = f"{ASSETS_DIR}/corpus_gz"
        gz_path = hf_hub_download(CORPUS_REPO, CORPUS_FILE, repo_type="dataset", local_dir=gz_dir)
        tmp = CORPUS_PATH + ".tmp"
        with gzip.open(gz_path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=64 << 20)
        os.replace(tmp, CORPUS_PATH)
        shutil.rmtree(gz_dir, ignore_errors=True)
        corpus_done.touch()
    _unwrap_tar(CORPUS_PATH)  # also repairs a copy decompressed by an earlier version
    print(f"corpus ready: {CORPUS_PATH} ({_gb(CORPUS_PATH):.1f} GB)")


def _unwrap_tar(path: str) -> bool:
    """If `path` is a tar archive, replace it with the single file inside it."""
    import tarfile

    if not tarfile.is_tarfile(path):  # reads only the first header block
        return False
    tmp = path + ".tmp"
    with tarfile.open(path) as archive:
        member = archive.next()
        with archive.extractfile(member) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, length=64 << 20)
    os.replace(tmp, path)
    # Any offset table built from the wrapped file is wrong; remove it so it is rebuilt.
    for stale in glob(f"{ASSETS_DIR}/wiki-18.offsets*.npy"):
        os.remove(stale)
    print(f"corpus was a tar archive; extracted {member.name} ({member.size / 1e9:.1f} GB)")
    return True


def _concatenate(parts: list[str], out_path: str) -> None:
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as out:
        for part in parts:
            with open(part, "rb") as src:
                shutil.copyfileobj(src, out, length=64 << 20)
    os.replace(tmp, out_path)


def _gb(path: str) -> float:
    return os.path.getsize(path) / 1e9


if __name__ == "__main__":
    # Local check of the CSV parser only (no downloads).
    rows = read_bamboogle_csv("Bamboogle_Prerelease_-_Sheet1.csv")
    print(len(rows), rows[0])