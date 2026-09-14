"""The serving state: everything one request needs, loaded once, with its memory measured
(SPEC.md §16.1–16.2).

Built from exactly the objects P2 trains against — `scripts.rerank_<dataset>_a2.Stage1` (article
text, inverted index + BM25, flat FAISS index, `RollingCounts` over the training split) — plus
three stores a live system keeps per user or per session and P2 computes in batch:

* `UserStore`   — last-n clicked ids (the query / user vector) and the user's click log with
                  categories (the recency-weighted profile, h = ∞);
* `SessionStore` — the pre-t position of an impression in its session (`session_pos`), as a
                  per-session counter would give it;
* article meta   — category and first-known time (publish time on EB-NeRD; earliest sighting on
                  MIND) for `freshness_hours`.

`memory[component] = {disk_bytes, ram_bytes}`: disk = the files loaded; RAM = RSS delta while
building that component (FAISS flat: `ntotal × d × 4`, exact). RSS deltas include Python
object overhead, which is the honest number for this implementation.
"""
from __future__ import annotations

import gc
import os
import resource
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import polars as pl


def _rss() -> int:
    with open(f"/proc/{os.getpid()}/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _disk(*paths: Path) -> int:
    return sum(Path(p).stat().st_size for p in paths if Path(p).exists())


@dataclass
class SessionStore:
    """imp_row -> session_pos, from `session_features` (1 + impressions of the session strictly
    before t). A live system keeps this as a counter per (user, session)."""
    pos: dict[int, int]

    @classmethod
    def from_behaviors(cls, beh: pl.DataFrame) -> "SessionStore":
        from src.features.behavioural import session_features
        sf = session_features(beh.select("imp_row", "user_id", "session_id", "t"))
        return cls(dict(zip(sf["imp_row"].to_list(), sf["session_pos"].to_list())))

    def position(self, imp_row: int) -> int:
        return self.pos.get(imp_row, 1)


@dataclass
class UserStore:
    """Per user: the last-n clicked ids (oldest first) and the full click log with categories."""
    recent_ids: dict
    logs: dict

    @classmethod
    def from_click_log(cls, log: pl.DataFrame, n_recent: int) -> "UserStore":
        log = log.sort("user_id", "ts")
        recent = {u: ids[-n_recent:] for u, ids in
                  log.group_by("user_id", maintain_order=True).agg("article_id").iter_rows()}
        logs = {u: part for u, part in log.partition_by("user_id", as_dict=True, maintain_order=True).items()}
        logs = {(k[0] if isinstance(k, tuple) else k): v for k, v in logs.items()}
        return cls(recent, logs)

    def recent(self, user_id) -> list:
        return list(self.recent_ids.get(user_id, []))

    def log(self, user_id) -> pl.DataFrame:
        return self.logs.get(user_id, _EMPTY_LOG)


_EMPTY_LOG = pl.DataFrame({"user_id": [], "ts": [], "article_id": [], "category": []},
                          schema={"user_id": pl.Int64, "ts": pl.Datetime("us"), "article_id": pl.Int64, "category": pl.Utf8})


@dataclass
class ServingState:
    dataset: str
    stage1: object                     # scripts.rerank_<dataset>_a2.Stage1: text, bm, ann, rc
    features: list[str]                # config.FINAL[dataset]["features"], in model order
    article_cat: dict
    article_time: dict                 # article_id -> first-known time (publish / first sighting)
    users: UserStore
    sessions: SessionStore
    untimed_ts: datetime | None
    memory: dict = field(default_factory=dict)
    build_seconds: dict = field(default_factory=dict)

    @classmethod
    def build(cls, dataset: str) -> "ServingState":
        return {"ebnerd": cls._build_ebnerd, "mind": cls._build_mind}[dataset]()

    @classmethod
    def _build_ebnerd(cls) -> "ServingState":
        from scripts.rerank_ebnerd_a2 import EMB, N_RECENT, Stage1
        from src.rerank.config import FINAL
        from src.rerank.ebnerd import ARTICLES, SMALL, load_articles, load_behaviors, load_history
        mem, secs = {}, {}
        gc.collect(); r0 = _rss(); t0 = time.perf_counter()
        articles = load_articles()
        article_cat = dict(zip(articles["article_id"].to_list(), articles["category"].to_list()))
        article_time = dict(zip(articles["article_id"].to_list(), articles["published_time"].to_list()))
        gc.collect(); r1 = _rss(); mem["articles"] = {"disk_bytes": _disk(ARTICLES), "ram_bytes": max(r1 - r0, 1)}
        secs["articles"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        train_beh = load_behaviors(SMALL / "train/behaviors.parquet")
        val_beh = load_behaviors(SMALL / "validation/behaviors.parquet")
        gc.collect(); r2 = _rss()
        st = Stage1(train_beh)                                   # text + BM25, ANN, counts — as P2
        gc.collect(); r3 = _rss()
        # split Stage1's RSS by rebuilding-free estimates: ANN exact, the rest by construction order
        ann_bytes = int(st.ann.matrix.nbytes)
        mem["ann_index"] = {"disk_bytes": _disk(EMB), "ram_bytes": ann_bytes}
        idx = st.bm.index
        mem["bm25_index"] = {"disk_bytes": 0, "ram_bytes": max(r3 - r2 - ann_bytes - _counts_bytes(st.rc), 1),
                             "n_docs": idx.n_docs, "n_terms": len(idx.postings) if hasattr(idx, "postings") else None}
        mem["counts"] = {"disk_bytes": _disk(SMALL / "train/behaviors.parquet"), "ram_bytes": _counts_bytes(st.rc),
                         "n_articles": st.rc.n_articles}
        secs["stage1"] = time.perf_counter() - t0

        t0 = time.perf_counter(); gc.collect(); r4 = _rss()
        # The split's own history snapshot (EB-NeRD ships one per split): what the system knows
        # about each user at the start of the served period, and exactly what the batch path
        # reads for these impressions (`build` -> load_history / recent_history on the same file).
        log = load_history(SMALL / "validation/history.parquet", articles, val_beh["user_id"])
        ustore = UserStore.from_click_log(log.select("user_id", "ts", "article_id", "category"), N_RECENT)
        del log; gc.collect(); r5 = _rss()
        mem["user_store"] = {"disk_bytes": _disk(SMALL / "validation/history.parquet"),
                             "ram_bytes": max(r5 - r4, 1), "n_users": len(ustore.recent_ids)}
        secs["user_store"] = time.perf_counter() - t0

        t0 = time.perf_counter(); gc.collect(); r6 = _rss()
        # keyed by the validation split's imp_row (what the bench requests carry); a live system
        # would keep a counter per (user, session) instead
        sessions = SessionStore.from_behaviors(val_beh.select("imp_row", "user_id", "session_id", "t"))
        gc.collect(); r7 = _rss()
        mem["session_store"] = {"disk_bytes": _disk(SMALL / "validation/behaviors.parquet"), "ram_bytes": max(r7 - r6, 1),
                                "n_impressions": len(sessions.pos)}
        secs["session_store"] = time.perf_counter() - t0
        mem["process_rss_after_build"] = {"ram_bytes": _rss()}
        return cls("ebnerd", st, FINAL["ebnerd"]["features"], article_cat, article_time, ustore, sessions, None, mem, secs)

    @classmethod
    def _build_mind(cls) -> "ServingState":
        """MIND: as `scripts/rerank_mind_a2.py` — BM25 over the small-set news (train + dev), the
        MiniLM ANN over every id in `mind_minilm.npz`, counts over MINDsmall_train. History is
        inline per impression in MIND, so the user store is the dev split's per-user snapshot and
        each request may also carry its own `history` (what the batch path sees)."""
        from scripts.rerank_mind_a2 import DEV, OUT, TRAIN, Stage1
        from src.rerank.config import FINAL
        from src.rerank.mind import ROOT, load_behaviors, split_dir
        mem, secs = {}, {}
        gc.collect(); r0 = _rss(); t0 = time.perf_counter()
        train, dev = load_behaviors(TRAIN), load_behaviors(DEV)
        gc.collect(); r1 = _rss()
        st = Stage1([TRAIN, DEV], train)
        gc.collect(); r2 = _rss()
        ann_bytes = int(st.ann.matrix.nbytes)
        news_files = [split_dir(ROOT, n) / "news.tsv" for n in (TRAIN, DEV)]
        mem["articles"] = {"disk_bytes": _disk(*news_files), "ram_bytes": 1, "n_articles": len(st.text)}
        mem["ann_index"] = {"disk_bytes": _disk(OUT / "mind_minilm.npz"), "ram_bytes": ann_bytes}
        mem["bm25_index"] = {"disk_bytes": 0, "ram_bytes": max(r2 - r1 - ann_bytes - _counts_bytes(st.rc), 1), "n_docs": st.bm.index.n_docs}
        mem["counts"] = {"disk_bytes": _disk(split_dir(ROOT, TRAIN) / "behaviors.tsv"), "ram_bytes": _counts_bytes(st.rc), "n_articles": st.rc.n_articles}
        secs["stage1"] = time.perf_counter() - t0

        t0 = time.perf_counter(); gc.collect(); r3 = _rss()
        recent = dict(zip(dev["user_id"].to_list(), [list(h) for h in dev["history_ids"].to_list()]))
        ustore = UserStore(recent, {})
        gc.collect(); r4 = _rss()
        mem["user_store"] = {"disk_bytes": _disk(split_dir(ROOT, DEV) / "behaviors.tsv"), "ram_bytes": max(r4 - r3, 1), "n_users": len(recent)}
        secs["user_store"] = time.perf_counter() - t0
        sessions = SessionStore({})                       # no session feature in config.FINAL["mind"]
        mem["session_store"] = {"disk_bytes": 0, "ram_bytes": 1, "n_impressions": 0}
        mem["process_rss_after_build"] = {"ram_bytes": _rss()}
        return cls("mind", st, FINAL["mind"]["features"], dict(st.cat), {}, ustore, sessions, None, mem, secs)


def _counts_bytes(rc) -> int:
    """RollingCounts: per-article Python lists of timestamps. Sum of container + element sizes."""
    import sys
    total = 0
    for store in (rc._clicks, rc._views):
        total += sys.getsizeof(store)
        for k, v in store.items():
            total += sys.getsizeof(k) + sys.getsizeof(v) + sum(sys.getsizeof(x) for x in v[:1]) * len(v)
    return total


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
