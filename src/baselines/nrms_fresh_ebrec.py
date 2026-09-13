"""NRMS + freshness term for EB-NeRD, on top of ebnerd-benchmark's NRMS (SPEC.md §15.1).

    score = user · news + g(x, unknown),   g = Dense(8, relu) → Dense(1)

`NRMSFreshModel` subclasses the benchmark's `NRMSModel` and overrides only the graph assembly:
the news and user encoders are the benchmark's own (`_build_newsencoder`, `_build_userencoder`),
built exactly as before; the model gains a third input `pred_input_fresh` of shape
(n_candidates, 2) and the scorer one of shape (1, 2). With `g ≡ 0` (`zero_g()`) the outputs
equal the baseline's for the same encoder weights — the additive identity `tests/test_nrms_fresh_model.py`
asserts.

`NRMSFreshLoader` extends the benchmark's pretransform loader: the behaviors frame carries a
`fresh_inview` list column (one [x, unknown] pair per candidate, built **after** wu2019 sampling
because that sampler reshuffles the slate) and `__getitem__` emits it as the third input array.
`mask=True` emits (0, 1) for every candidate: ablation row 3 (§15.3), the term switched off at
inference without retraining.

Requires TensorFlow and `ebrec` on sys.path (Kaggle); nothing here runs on the laptop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from ebrec.models.newsrec.dataloader import NRMSDataLoaderPretransform
from ebrec.models.newsrec.nrms import NRMSModel

FRESH_DIM = 2   # [x, unknown]


class NRMSFreshModel(NRMSModel):
    """The benchmark's NRMS with an additive freshness term in the scoring head."""

    def _build_nrms(self):
        hp = self.hparams
        his_input_title = tf.keras.Input(shape=(hp.history_size, hp.title_size), dtype="int32")
        pred_input_title = tf.keras.Input(shape=(None, hp.title_size), dtype="int32")
        pred_input_fresh = tf.keras.Input(shape=(None, FRESH_DIM), dtype="float32", name="pred_input_fresh")
        pred_input_title_one = tf.keras.Input(shape=(1, hp.title_size), dtype="int32")
        pred_input_fresh_one = tf.keras.Input(shape=(1, FRESH_DIM), dtype="float32", name="pred_input_fresh_one")
        pred_title_one_reshape = tf.keras.layers.Reshape((hp.title_size,))(pred_input_title_one)

        titleencoder = self._build_newsencoder(units_per_layer=hp.newsencoder_units_per_layer)
        self.userencoder = self._build_userencoder(titleencoder)
        self.newsencoder = titleencoder

        # g: one small MLP applied per candidate; its layers are named so the identity test and
        # the ablation can find them.
        self.g = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation="relu", name="fresh_hidden",
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
            tf.keras.layers.Dense(1, name="fresh_out",
                                  kernel_initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)),
        ], name="fresh_term")

        user_present = self.userencoder(his_input_title)
        news_present = tf.keras.layers.TimeDistributed(self.newsencoder)(pred_input_title)
        news_present_one = self.newsencoder(pred_title_one_reshape)

        dot = tf.keras.layers.Dot(axes=-1)([news_present, user_present])                  # (batch, n)
        fresh = tf.keras.layers.Reshape((-1,))(tf.keras.layers.TimeDistributed(self.g)(pred_input_fresh))  # (batch, n)
        preds = tf.keras.layers.Activation("softmax")(tf.keras.layers.Add()([dot, fresh]))

        dot_one = tf.keras.layers.Dot(axes=-1)([news_present_one, user_present])           # (batch, 1)
        fresh_one = tf.keras.layers.Reshape((1,))(self.g(tf.keras.layers.Reshape((FRESH_DIM,))(pred_input_fresh_one)))
        pred_one = tf.keras.layers.Activation("sigmoid")(tf.keras.layers.Add()([dot_one, fresh_one]))

        model = tf.keras.Model([his_input_title, pred_input_title, pred_input_fresh], preds)
        scorer = tf.keras.Model([his_input_title, pred_input_title_one, pred_input_fresh_one], pred_one)
        return model, scorer

    def zero_g(self) -> None:
        """Make the freshness term identically zero (the additive-identity check)."""
        out = self.g.get_layer("fresh_out")
        out.set_weights([np.zeros_like(out.get_weights()[0]), np.zeros_like(out.get_weights()[1])])

    def copy_encoders_from(self, baseline: NRMSModel) -> None:
        """Take the news and user encoder weights from a baseline NRMS (same hparams)."""
        self.newsencoder.set_weights(baseline.newsencoder.get_weights())
        self.userencoder.set_weights(baseline.userencoder.get_weights())


@dataclass
class NRMSFreshLoader(NRMSDataLoaderPretransform):
    """The benchmark's pretransform loader plus the freshness input.

    `behaviors` must carry `fresh_inview`: per impression, a list of [x, unknown] pairs aligned
    with `article_ids_inview` (from `nrms_fresh_features.as_lists`, built on the *sampled* frame
    for training). `mask=True` replaces every pair with (0, 1)."""
    fresh_col: str = "fresh_inview"
    mask: bool = False

    def __getitem__(self, idx):
        (his, pred), y = super().__getitem__(idx)
        batch_X = self.X[idx * self.batch_size: (idx + 1) * self.batch_size]
        rows = batch_X[self.fresh_col].to_list()
        if self.eval_mode:
            fresh = np.asarray([pair for slate in rows for pair in slate], dtype="float32").reshape(-1, FRESH_DIM)
            assert fresh.shape[0] == pred.shape[0], (fresh.shape, pred.shape)
            fresh = fresh.reshape(-1, 1, FRESH_DIM)                     # scorer input: (rows, 1, 2)
        else:
            fresh = np.asarray(rows, dtype="float32")                    # (batch, npratio+1, 2)
            assert fresh.shape[:2] == pred.shape[:2], (fresh.shape, pred.shape)
        if self.mask:
            fresh = np.tile(np.array([0.0, 1.0], dtype="float32"), fresh.shape[:-1] + (1,))
        return (his, pred, fresh), y
