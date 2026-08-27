"""Article embeddings for semantic retrieval (A1 Q3.1).

Two sources, because the two datasets genuinely differ in what they ship:

* **EB-NeRD provides them.** `document_vector.parquet` (Ekstra Bladet's own word2vec) and a
  multilingual BERT file, both covering all 125,541 articles. We use the word2vec one: it was
  trained by the publisher on Danish news, where the multilingual BERT is a general-purpose
  model that happens to include Danish. Using the provided vectors is also what the brief
  suggests, and it removes a multi-hour embedding job from the critical path.

* **MIND provides none.** It ships TransE *entity* embeddings, not article vectors, so
  article vectors have to be computed. We use TF-IDF followed by truncated SVD -- classical
  latent semantic analysis. The alternative was a sentence-transformer, which means a ~2.5 GB
  download and an encoding pass over 125K articles on a machine with ~2 GB free (SPEC.md
  §11). LSA is a defensible semantic representation, runs in seconds, and -- being a linear
  factorisation of the same term-document matrix BM25 scores -- makes the lexical-vs-semantic
  comparison a genuine comparison of *representation*, not of model scale.

Every vector is L2-normalised, so an inner-product index computes cosine similarity.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Zero rows stay zero instead of becoming NaN.

    A zero row is a real case -- an article whose text tokenises to nothing -- and turning it
    into NaN would silently poison every similarity it participates in.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def load_provided(path, id_column: str = "article_id", vector_column: str | None = None):
    """Load EB-NeRD's provided article vectors. Returns (ids, matrix)."""
    df = pl.read_parquet(path)
    if vector_column is None:
        vector_column = next(c for c in df.columns if c != id_column)
    ids = df[id_column].to_list()
    matrix = np.asarray(df[vector_column].to_list(), dtype=np.float32)
    return ids, l2_normalise(matrix)


def compute_lsa(texts: list[str], n_components: int = 128, seed: int = 0,
                min_df: int = 2, max_features: int = 200_000):
    """TF-IDF -> truncated SVD, i.e. latent semantic analysis. Returns (matrix, vectorizer, svd).

    `n_components=128` is a choice, not a constant: it trades reconstruction fidelity against
    index size and query cost, and is ablated in RESULTS.md. `min_df=2` drops terms occurring
    in a single document -- for a 125K-article corpus those are overwhelmingly typos and
    identifiers, and they inflate the matrix without adding recoverable structure.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(min_df=min_df, max_features=max_features,
                                 sublinear_tf=True, dtype=np.float32)
    tfidf = vectorizer.fit_transform(texts)
    # n_components must stay below the rank of the matrix; clamp rather than let SVD raise
    # on a small corpus, which is what the toy fixtures in the tests are.
    k = min(n_components, min(tfidf.shape) - 1)
    svd = TruncatedSVD(n_components=max(k, 1), random_state=seed)
    return l2_normalise(svd.fit_transform(tfidf)), vectorizer, svd
