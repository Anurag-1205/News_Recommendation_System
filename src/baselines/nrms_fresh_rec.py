"""NRMS + freshness term for MIND, on top of Microsoft Recommenders' NRMS (SPEC.md §15.1).

Same change as `nrms_fresh_ebrec`: `score = user · news + g(x, unknown)`. The encoders are the
package's own; `NRMSFreshModel` overrides only the graph assembly, the batch → inputs mapping
and the fast-evaluation scoring. `MINDFreshIterator` is the package's `MINDIterator` with one
extra per-candidate array, `candidate_fresh_batch`, looked up by (behaviors row, news id) from a
table built by `nrms_fresh_features` on the same TSVs.

Runs under tf-keras (`TF_USE_LEGACY_KERAS=1`, TF1 graph mode) on Kaggle only.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import layers
from recommenders.models.newsrec.io.mind_iterator import MINDIterator
from recommenders.models.newsrec.models.nrms import NRMSModel
from recommenders.models.newsrec.newsrec_utils import newsample

FRESH_DIM = 2
UNKNOWN = (0.0, 1.0)


class MINDFreshIterator(MINDIterator):
    """`MINDIterator` plus `candidate_fresh_batch`.

    `fresh_by_row`: list indexed by behaviors.tsv row, each a dict news_id (e.g. "N28682") →
    (x, unknown). Set it before the first `load_*` call; `mask=True` serves (0, 1) everywhere."""

    fresh_by_row: list[dict[str, tuple[float, float]]] | None = None
    mask: bool = False

    def _fresh(self, line: int, news_index: int) -> tuple[float, float]:
        if self.mask:
            return UNKNOWN
        nid = self.index2nid.get(int(news_index))     # 0 = the sampler's padding id, not an article
        return self.fresh_by_row[line].get(nid, UNKNOWN) if nid is not None else UNKNOWN

    def init_news(self, news_file):
        super().init_news(news_file)
        self.index2nid = {v: k for k, v in self.nid2index.items()}

    def parser_one_line(self, line):
        # The package's parser, with the freshness pairs added to each yield (its structure is
        # kept line for line so the sampling sequence — and hence determinism — is unchanged).
        if self.npratio > 0:
            impr_label, impr = self.labels[line], self.imprs[line]
            poss, negs = [], []
            for news, click in zip(impr, impr_label):
                (poss if click == 1 else negs).append(news)
            for p in poss:
                label = [1] + [0] * self.npratio
                n = newsample(negs, self.npratio)
                cands = [p] + n
                candidate_title_index = self.news_title_index[cands]
                click_title_index = self.news_title_index[self.histories[line]]
                fresh = np.array([self._fresh(line, c) for c in cands], dtype=np.float32)
                yield (label, [self.impr_indexes[line]], [self.uindexes[line]], candidate_title_index, click_title_index, fresh)
        else:
            impr_label, impr = self.labels[line], self.imprs[line]
            for news, label in zip(impr, impr_label):
                candidate_title_index = [self.news_title_index[news]]
                click_title_index = self.news_title_index[self.histories[line]]
                fresh = np.array([self._fresh(line, news)], dtype=np.float32)
                yield ([label], [self.impr_indexes[line]], [self.uindexes[line]], candidate_title_index, click_title_index, fresh)

    def load_data_from_file(self, news_file, behavior_file):
        if not hasattr(self, "news_title_index"):
            self.init_news(news_file)
        if not hasattr(self, "impr_indexes"):
            self.init_behaviors(behavior_file)
        assert self.fresh_by_row is not None or self.mask, "set fresh_by_row first"
        buf = {k: [] for k in ("label", "imp", "user", "cand", "click", "fresh")}
        indexes = np.arange(len(self.labels))
        if self.npratio > 0:
            np.random.shuffle(indexes)
        for index in indexes:
            for label, imp, user, cand, click, fresh in self.parser_one_line(index):
                for k, v in zip(buf, (label, imp, user, cand, click, fresh)):
                    buf[k].append(v)
                if len(buf["label"]) >= self.batch_size:
                    yield self._convert_data_fresh(buf)
                    buf = {k: [] for k in buf}
        if buf["label"]:
            yield self._convert_data_fresh(buf)

    def _convert_data_fresh(self, buf):
        d = self._convert_data(buf["label"], buf["imp"], buf["user"], buf["cand"], buf["click"])
        d["candidate_fresh_batch"] = np.asarray(buf["fresh"], dtype=np.float32)
        return d


class NRMSFreshModel(NRMSModel):
    """Recommenders' NRMS with the additive freshness term."""

    def _build_nrms(self):
        hp = self.hparams
        his_input_title = keras.Input(shape=(hp.his_size, hp.title_size), dtype="int32")
        pred_input_title = keras.Input(shape=(hp.npratio + 1, hp.title_size), dtype="int32")
        pred_input_fresh = keras.Input(shape=(hp.npratio + 1, FRESH_DIM), dtype="float32", name="pred_input_fresh")
        pred_input_title_one = keras.Input(shape=(1, hp.title_size), dtype="int32")
        pred_input_fresh_one = keras.Input(shape=(1, FRESH_DIM), dtype="float32", name="pred_input_fresh_one")
        pred_title_one_reshape = layers.Reshape((hp.title_size,))(pred_input_title_one)

        embedding_layer = layers.Embedding(self.word2vec_embedding.shape[0], hp.word_emb_dim,
                                           weights=[self.word2vec_embedding], trainable=True)
        titleencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder

        # g as a functional model with an explicit Input: in TF1 graph mode a Sequential used only
        # through TimeDistributed has no standalone input, so `g.predict` (run_fast_eval) fails.
        g_in = keras.Input(shape=(FRESH_DIM,), dtype="float32", name="fresh_in")
        g_h = layers.Dense(8, activation="relu", name="fresh_hidden", kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed))(g_in)
        g_out = layers.Dense(1, name="fresh_out", kernel_initializer=keras.initializers.glorot_uniform(seed=self.seed))(g_h)
        self.g = keras.Model(g_in, g_out, name="fresh_term")

        user_present = self.userencoder(his_input_title)
        news_present = layers.TimeDistributed(self.newsencoder)(pred_input_title)
        news_present_one = self.newsencoder(pred_title_one_reshape)

        dot = layers.Dot(axes=-1)([news_present, user_present])
        fresh = layers.Reshape((hp.npratio + 1,))(layers.TimeDistributed(self.g)(pred_input_fresh))
        preds = layers.Activation(activation="softmax")(layers.Add()([dot, fresh]))

        dot_one = layers.Dot(axes=-1)([news_present_one, user_present])
        fresh_one = layers.Reshape((1,))(self.g(layers.Reshape((FRESH_DIM,))(pred_input_fresh_one)))
        pred_one = layers.Activation(activation="sigmoid")(layers.Add()([dot_one, fresh_one]))

        model = keras.Model([his_input_title, pred_input_title, pred_input_fresh], preds)
        scorer = keras.Model([his_input_title, pred_input_title_one, pred_input_fresh_one], pred_one)
        return model, scorer

    def _get_input_label_from_iter(self, batch_data):
        return [batch_data["clicked_title_batch"], batch_data["candidate_title_batch"], batch_data["candidate_fresh_batch"]], batch_data["labels"]

    def zero_g(self):
        out = self.g.get_layer("fresh_out")
        out.set_weights([np.zeros_like(out.get_weights()[0]), np.zeros_like(out.get_weights()[1])])

    def g_values(self, pairs: np.ndarray) -> np.ndarray:
        """g on an (n, 2) array of (x, unknown) pairs, in one predict call."""
        return self.g.predict(pairs.astype(np.float32), batch_size=65536, verbose=0).reshape(-1)

    def run_fast_eval(self, news_filename, behaviors_file):
        """The package's fast path (user vectors · news vectors) plus the freshness term, looked
        up through the test iterator's `fresh_by_row` / `mask`."""
        news_vecs = self.run_news(news_filename)
        user_vecs = self.run_user(news_filename, behaviors_file)
        self.news_vecs, self.user_vecs = news_vecs, user_vecs
        it = self.test_iterator
        rows = list(it.load_impression_from_file(behaviors_file))
        pairs = np.array([it._fresh(impr_index, n) for impr_index, news_index, _, _ in rows for n in news_index], dtype=np.float32)
        gvals = self.g_values(pairs)
        out_idx, out_lab, out_pred = [], [], []
        k = 0
        for impr_index, news_index, user_index, label in rows:
            m = len(news_index)
            pred = np.dot(np.stack([news_vecs[i] for i in news_index], axis=0), user_vecs[impr_index]) + gvals[k:k + m]
            k += m
            out_idx.append(impr_index); out_lab.append(label); out_pred.append(pred)
        return out_idx, out_lab, out_pred
