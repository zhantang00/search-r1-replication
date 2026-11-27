
from __future__ import annotations

import json
import os
import time
from array import array
from collections import OrderedDict

from config import CORPUS_OFFSETS_PATH, CORPUS_PATH, INDEX_PATH, RETRIEVER_MODEL


def format_docs(contents_list: list[str]) -> str:
    """Official formatting: 'Doc {i}(Title: {title}) {text}\\n' per passage."""
    out = ""
    for i, contents in enumerate(contents_list):
        lines = contents.split("\n")
        out += f"Doc {i + 1}(Title: {lines[0]}) {chr(10).join(lines[1:])}\n"
    return out


def ensure_offsets(corpus_path: str = CORPUS_PATH,
                   offsets_path: str = CORPUS_OFFSETS_PATH) -> bool:
    """Build the byte-offset table for the corpus once. Returns True if it was built now."""
    if os.path.exists(offsets_path):
        return False
    import tarfile

    import numpy as np

    if tarfile.is_tarfile(corpus_path):
        raise RuntimeError(f"{corpus_path} is still a tar archive; run "
                           "`modal run modal_app.py::download_assets` to extract the JSONL")
    t0 = time.time()
    offsets = array("q")
    blank_lines = []
    position = 0
    with open(corpus_path, "rb", buffering=64 << 20) as f:
        for line_no, line in enumerate(f):
            if line.strip():
                offsets.append(position)
            else:
                blank_lines.append(line_no)
            position += len(line)
    tmp = offsets_path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, np.frombuffer(offsets, dtype=np.int64))
    os.replace(tmp, offsets_path)
    print(f"built corpus offsets: {len(offsets):,} passages in {time.time() - t0:.0f}s; "
          f"skipped {len(blank_lines)} blank line(s) at line numbers {blank_lines[:10]}")
    return True


class DenseRetriever:
    def __init__(self, topk: int = 3, n_threads: int = 8, index_path: str = INDEX_PATH,
                 corpus_path: str = CORPUS_PATH, offsets_path: str = CORPUS_OFFSETS_PATH,
                 model_name: str = RETRIEVER_MODEL, max_length: int = 256,
                 batch_size: int = 256, cache_size: int = 200_000):
        import faiss
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(n_threads)
        faiss.omp_set_num_threads(n_threads)
        self.topk, self.max_length, self.batch_size = topk, max_length, batch_size
        self.cache: OrderedDict[str, str] = OrderedDict()
        self.cache_size = cache_size

        t0 = time.time()
        self.offsets = np.load(offsets_path)
        with open(corpus_path, "rb") as f:
            self.corpus = f.read()                         # ~14 GB; lookups become memory slices
        print(f"loaded corpus into RAM in {time.time() - t0:.0f}s")
        self._check_ids_follow_rows()  # cheap; runs before the slow index load

        t0 = time.time()
        self.index = faiss.read_index(index_path)
        if self.index.ntotal != len(self.offsets):
            raise RuntimeError(
                f"index has {self.index.ntotal:,} vectors but the offsets table has "
                f"{len(self.offsets):,} passages ({offsets_path}); first passage id "
                f"{self._doc(0).get('id')!r}, last {self._doc(len(self.offsets) - 1).get('id')!r}")
        # Zero-copy view of the flat index's vectors as a [N, d] float32 tensor.
        xb = faiss.rev_swig_ptr(self.index.get_xb(), self.index.ntotal * self.index.d)
        self.vectors = torch.from_numpy(xb).view(self.index.ntotal, self.index.d)
        print(f"loaded index ({self.index.ntotal:,} vectors) in {time.time() - t0:.0f}s")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()

    def _doc(self, row: int) -> dict:
        start = int(self.offsets[row])
        end = self.corpus.find(b"\n", start)
        return json.loads(self.corpus[start:end if end >= 0 else len(self.corpus)])

    def _topk(self, emb, chunk: int = 1 << 20):
        """Exact top-k inner product over all vectors, in chunks, on every core.

        Returns ids [nq, k], best first (same ordering as faiss IndexFlatIP.search).
        """
        import torch

        queries = torch.from_numpy(emb)
        best_scores = torch.full((len(emb), self.topk), float("-inf"))
        best_ids = torch.zeros((len(emb), self.topk), dtype=torch.long)
        for start in range(0, self.vectors.size(0), chunk):
            scores = queries @ self.vectors[start:start + chunk].T     # [nq, chunk]
            top_scores, top_ids = scores.topk(self.topk, dim=1)
            merged_scores = torch.cat([best_scores, top_scores], dim=1)
            merged_ids = torch.cat([best_ids, top_ids + start], dim=1)
            best_scores, pick = merged_scores.topk(self.topk, dim=1)
            best_ids = merged_ids.gather(1, pick)
        return best_ids.numpy()

    def agreement(self, queries: list[str]) -> float:
        """Share of queries for which our search and faiss's return the same top-k ids."""
        emb = self.encode(queries)
        _, faiss_ids = self.index.search(emb, self.topk)
        ours = self._topk(emb)
        return sum(set(a) == set(b) for a, b in zip(ours, faiss_ids)) / len(queries)

    def _check_ids_follow_rows(self) -> None:
        """If passage ids are numeric, (id - row) must be the same everywhere.

        A blank or extra line in the middle of the file would shift every later
        row, making retrieved passages silently wrong; this catches that.
        """
        n = len(self.offsets)
        rows = sorted({0, 1, n // 4, n // 2, 3 * n // 4, n - 1})
        docs = [self._doc(r) for r in rows]  # an unreadable line raises here
        try:
            deltas = {int(doc["id"]) - r for doc, r in zip(docs, rows)}
        except (KeyError, ValueError):
            print("corpus ids are not numeric; skipping the id/row alignment check")
            return
        if len(deltas) != 1:
            raise RuntimeError(f"corpus ids drift relative to row numbers (id - row = {deltas})")
        print(f"corpus ids line up with row numbers (id - row = {deltas.pop()})")

    def encode(self, queries: list[str]):
        import numpy as np
        import torch

        inputs = self.tokenizer([f"query: {q}" for q in queries], max_length=self.max_length,
                                padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"]
            hidden = hidden.masked_fill(~mask[..., None].bool(), 0.0)
            emb = hidden.sum(dim=1) / mask.sum(dim=1)[..., None]
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.float().numpy().astype(np.float32, order="C")

    def passage(self, idx: int) -> str:
        return self._doc(idx)["contents"]

    def search(self, queries: list[str]) -> list[str]:
        missing = [q for q in dict.fromkeys(queries) if q not in self.cache]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start:start + self.batch_size]
            t0 = time.time()
            emb = self.encode(batch)
            t1 = time.time()
            ids = self._topk(emb)
            t2 = time.time()
            for query, row in zip(batch, ids):
                self.cache[query] = format_docs([self.passage(i) for i in row])
            print(f"[retriever] {len(batch)} queries: encode {t1 - t0:.1f}s, "
                  f"search {t2 - t1:.1f}s, fetch {time.time() - t2:.2f}s")
        results = [self.cache[q] for q in queries]
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return results


class FakeRetriever:
    """Canned passages for plumbing tests; no index or corpus needed."""

    def search(self, queries: list[str]) -> list[str]:
        return [format_docs([f'"Placeholder"\nThis is a placeholder passage for the query: {q}.'])
                for q in queries]



CASE_STUDY_QUERIES = {
    "Curious fragrance information": "Curious (fragrance)",
    "Britney Spears birthplace": "Britney Spears",
    "Teide National Park location": "Teide National Park",
    "Empire Distribution location": "Empire Distribution",
    "who was Richard C. Sarafian": "Richard C. Sarafian",
    "Gary Barlow profession": "Gary Barlow",
}