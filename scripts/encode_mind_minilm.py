#!/usr/bin/env python3
"""Encode the MIND article corpus with a sentence-transformer (Q3 alternative to LSA).

    PYTHONPATH=. .venv/bin/python -u scripts/encode_mind_minilm.py

`all-MiniLM-L6-v2` is a 6-layer, 384-dimensional English sentence encoder. It is chosen over
a larger multilingual model for two reasons: MIND is English, so multilingual capacity buys
nothing here, and this machine encodes on CPU where a 6-layer model is the difference between
minutes and hours.

Vectors are L2-normalised and cached to `data/processed/mind_minilm.npz`, so the expensive
pass runs once and every downstream experiment reuses it. The cache stores article ids
alongside the matrix — a matrix whose row order cannot be reconstructed is not a cache.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.pipeline.mind import read_news, split_dir
from src.semantic.embeddings import l2_normalise

ROOT, OUT = Path("data/interim/mind"), Path("data/processed")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH = 256


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "mind_minilm.npz"

    ids, texts = [], []
    seen = set()
    for name in ("MINDsmall_train", "MINDsmall_dev", "MINDlarge_test"):
        n = read_news(split_dir(ROOT, name))
        for a, ti, ab in zip(n["news_id"], n["title"], n["abstract"]):
            if a in seen:
                continue
            seen.add(a); ids.append(a)
            # Title first: it is the text the user actually saw in the slate, and truncation
            # at the model's 256-token limit should drop abstract tail rather than headline.
            texts.append(f"{ti or ''}. {ab or ''}".strip())
    log(f"{len(ids):,} unique articles to encode")

    from sentence_transformers import SentenceTransformer
    log(f"loading {MODEL}")
    model = SentenceTransformer(MODEL, device="cpu")
    log(f"  max_seq_length={model.max_seq_length}  dim={model.get_sentence_embedding_dimension()}")

    t0 = time.perf_counter()
    vectors = model.encode(texts, batch_size=BATCH, show_progress_bar=False,
                           convert_to_numpy=True, normalize_embeddings=False)
    elapsed = time.perf_counter() - t0
    log(f"encoded in {elapsed:.0f}s ({len(ids)/elapsed:.0f} articles/s)")

    matrix = l2_normalise(np.asarray(vectors, dtype=np.float32))
    np.savez_compressed(cache, ids=np.array(ids, dtype=object), matrix=matrix)
    log(f"cached {matrix.shape} -> {cache} ({cache.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
