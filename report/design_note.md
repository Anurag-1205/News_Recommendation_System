# Lexical and Semantic Retrieval on EB-NeRD and MIND

**CS4.406 Information Retrieval and Extraction — Assignment 1, Component 1**
Anurag Kaushal · 22 August 2026

**Scope.** This note covers Q6 of the assignment brief: what was built, the alternatives
considered, observations from the experiments, and where the pipeline breaks at ten times the
current scale. Supporting material is cross-referenced as follows — component specifications
and verification strategy in `SPEC.md`, every measured figure with its originating command in
`RESULTS.md`, and the tooling disclosure in `AI_USAGE.md`.

**Statistical convention.** Every reported metric carries a bootstrap 95% confidence interval
computed by resampling impressions. Where two intervals overlap, the result is reported as
showing no significant difference; the term "beats" is reserved for disjoint intervals.

---

## 1. What I built

A pipeline that ranks the candidate articles of an impression by click likelihood, on two
datasets that share almost nothing structurally — MIND (English, TSV, history inline) and
EB-NeRD (Danish, Parquet, history in a separate file, 13.5M test impressions).

**One scorer, two harnesses.** Settling this before writing any retrieval code was the single
highest-leverage decision. Every model exposes `score(user, candidates) -> list[float]`, wrapped
by either *candidate generation* (score the whole corpus, report recall@K) or *in-impression
re-ranking* (score only `article_ids_inview`, report AUC/MRR/nDCG). The two leaderboards measure
the second; Q2.4 and Q3.4 ask for the first. Retrofitting one onto the other would have cost a
day I did not have.

```
raw archives ─► reader ─► unified schema ─► temporal split ─► feature store (as-of cutoff)
                                                        │
                                          point-in-time counts (strictly before t)
                                                                      │
   ┌──────────────────────────────────────────────────────────────────┤
   ▼                          ▼                        ▼              ▼
inverted index + BM25   TF-IDF→SVD + FAISS      popularity      fusion (RRF / weighted)
   └──────────────────────────┴────────────────────────┴──────────────┘
                                     │
                      ┌──────────────┴───────────────┐
              recall@K (corpus)            AUC/MRR/nDCG + beyond-accuracy
                                           + slices + bootstrap CIs
                                                     │
                                          Codabench submission + offline validator
```

**Order of construction: Q1 → Q4 → Q2 → Q3.** The evaluation harness was built *before* the
retrievers, because it is their oracle. Building BM25 first would have meant tuning it blind and
re-running everything once metrics existed.

## 2. Choices and alternatives

**Own BM25, `rank_bm25` as oracle only.** The library is in `requirements.txt` purely to disagree
with my implementation. It earned its place: my Okapi IDF floor averaged only the *positive* IDFs,
where `rank_bm25` averages all of them including negatives. Exact score comparison caught a wrong
floor I would never have found by inspection.

**Lucene IDF over Okapi.** Okapi's `ln((N-df+0.5)/(df+0.5))` goes *negative* for terms in more
than half the corpus, so a common word subtracts from a document's score. On six-word news
headlines that is a real misbehaviour, not a curiosity. Lucene's `+1` inside the log keeps IDF
positive. Both are implemented; Okapi exists so the external check can run.

**TF-IDF + truncated SVD first, a sentence-transformer once it was affordable.** MIND ships no
article vectors. LSA runs in 7 s and, being a linear factorisation of the same term-document
matrix BM25 scores, isolates *representation* from model scale in the lexical-vs-semantic
comparison. Its cost is that 128 components recover only 12.2% of the variance. When network and
memory conditions allowed, `all-MiniLM-L6-v2` was encoded over all 125,590 articles in 2,255 s on
CPU (56 articles/s, 384 dimensions) and measured against LSA under the gap-aware protocol — where
it wins by 0.040 AUC, disjoint. The lesson is that the representation, not the ranker, was the
binding constraint on content signal.

**EB-NeRD's provided word2vec over multilingual BERT.** Publisher-trained on Danish news beats a
general multilingual model that happens to include Danish.

**Flat FAISS index, not IVF/HNSW.** At 125K articles a flat index answers in ~1 ms and is *exact*,
so recall@K measures the retrieval model rather than the index's approximation error. IVF is
implemented to measure the trade, not to serve results.

**Polars lazy scans over Dask.** Offered Dask by the TAs; both reference notebooks use Polars, and
matching them keeps the code cross-referenceable.

## 3. Observations

**Popularity behaves oppositely on the two datasets.** On MIND it helps a little (AUC 0.5318) and
fuses usefully with BM25. On EB-NeRD it is *actively harmful* — **AUC 0.4429 [0.4424, 0.4435]**,
well below chance — and every fusion containing it does worse than BM25 alone. This was verified independently of the main
pipeline before being accepted. The mechanism is popularity bias in the logging
policy: EB-NeRD's `article_ids_inview` lists are what Ekstra Bladet's own recommender chose to
show, so globally popular articles appear in nearly every impression and are clicked in few.
*Conditioned on being in view*, train-popularity predicts non-clicks. This is not a coverage
problem — EB-NeRD coverage is 43%, against MIND's 6.5%.

**Consequence: the winning model differs by dataset.** MIND ships weighted-sum fusion
(0.5565 [0.5543, 0.5586], beating both components with disjoint intervals); EB-NeRD ships BM25
alone (0.5030 [0.5019, 0.5041], beating both fusions). A single model selected across both datasets would
therefore have been the wrong choice.

**A correction to an earlier conclusion.** The anti-predictive popularity result was
initially read as evidence that EB-NeRD is intrinsically difficult to rank. That inference was
incorrect. Adding *article age at impression* — computable
from the `published_time` column, which had not previously been used — moves EB-NeRD from
0.5030 to **0.7084 [0.7054, 0.7112]**. The limiting factor was not the text model but an
unused column. EB-NeRD's slates are pre-filtered to
fresh articles, and how fresh is what separates them. The anti-predictive popularity measurement stands on its own terms; the conclusion drawn from
it did not.

**The point-in-time result does not replicate across datasets, and the reason is legible.** On
MIND, rolling popularity beats frozen by 0.039 with disjoint intervals. On EB-NeRD the intervals
overlap — no difference. Permutation importance explains it: on EB-NeRD `age_hours` carries
+0.125 while every popularity feature sits near +0.005. Rolling versus frozen cannot matter when
popularity itself barely matters. A finding that holds on one dataset and not the other, with a
mechanism for the difference, is worth more than one asserted to be universal.

**Semantic beats lexical overall, but not where the textbook says.** The expectation is that
semantic retrieval rescues cold-start users. It does not:

| Slice | BM25 | LSA | Verdict |
|---|---|---|---|
| all | 0.5468 | 0.5588 | semantic, disjoint |
| cold-start (≤5 clicks) | 0.5308 | 0.5379 | **overlap — no difference** |
| warm (>5 clicks) | 0.5501 | 0.5632 | semantic, disjoint |
| head articles | 0.5159 | 0.5604 | semantic, disjoint |
| tail articles | 0.5575 | 0.5582 | **overlap — no difference** |

Semantic wins where there is history to pool and on *head* articles — popular stories covered by
many near-synonymous headlines, exactly where term overlap fails. On cold-start it has too little
to pool to help at all.

**Recency-weighted pooling lost to mean pooling**, at every K, by ~40% — contradicting the
hypothesis it was built to test. News decays fast, so recent clicks *should* describe a user
better. The likely reason: averaging more history suppresses noise. The decay hypothesis was not
wrong about news, it was wrong about which quantity is being estimated.

**Coverage is worse than any accuracy metric reveals.** Both models surface under **4%** of the
catalogue in any top-10, with Gini > 0.90. This is a popularity amplifier, and no AUC would show it.

**Serving-time honesty is not a formality.** Adding EB-NeRD's `next_read_time` — absent from test
by construction — moves AUC from **0.5029 to 0.9629**, an inflation of **+0.46**. A chance-level
model becomes apparently near-perfect via a column that cannot exist at request time.

**Point-in-time popularity beats frozen popularity — and the direction is the lesson.** Counting
clicks strictly before each impression gives AUC 0.6447 [0.6426, 0.6467]; freezing the count over
the whole training window gives 0.6058 [0.6035, 0.6079], disjoint. The frozen variant *leaks* —
an early impression is scored using clicks that came after it — and still loses, because a count
averaged over six days is stale against a dev day outside that window, and news popularity turns
over in hours. Staleness costs more than the leak gains. That is a stronger argument for
point-in-time features than "leakage is bad".

**Behavioural signal dominates content signal, quantitatively.** Permutation importance puts
`pop_total` (+0.041) and `ctr_total` (+0.041) at roughly 6× `bm25` (+0.006) and 5× `semantic`
(+0.007). This is the same finding the weak recall@K numbers imply, measured a second way.
(`slate_size` ranks first in that table and should be ignored. It is constant within an
impression, so shuffling it globally manufactures variation the real feature never has.
Leave-one-out refitting confirms this: 0.6417 without it against 0.6454 with it, intervals
overlapping. Permutation importance is not group-aware, and feature attribution needs the same
interval discipline as every other number.)

**Offline-to-leaderboard calibration held for two submissions and then broke — which was the
most useful result of the project.** Submissions 1 and 2 came back 0.028 and 0.031 below their
offline figures, and I concluded the harness was calibrated. Submission 3, built on
point-in-time behavioural features, came back **0.090** below. Two points from one model family
were not evidence about a different one.

The cause is measured, not inferred. Training click events end 14 Nov; dev is 15 Nov; the test
split is 16–22 Nov. A 24-hour lookback from a dev impression still reaches training data. The
same lookback from a test impression reaches a window with no events in it. Across 787,307
sampled test candidates, `pop_24h`, `pop_1h` and `ctr_24h` are **zero for 100% of them**, while
on dev two of the three carried signal 42% of the time and the trees split on them.

This is not leakage — nothing from the future was used. It is **train/serve skew**, and the
validation split was structurally incapable of detecting it, because dev is adjacent to training
and the test set never is. The fix was to rebuild the validation rather than the model: fit
counts to 12 Nov, train on 14 Nov, evaluate on 15 Nov, so both sides sit in the degenerate
regime the test split imposes. Under that protocol a 384-dimensional sentence encoder beats the
128-component LSA by 0.040 AUC with disjoint intervals, while dropping the dead features changes
nothing — a tree cannot split on a constant, so it had already ignored them.

## 4. Where it breaks at 10×

Measured, not projected.

| Operation | Measured | At 10× |
|---|---|---|
| Index build, 125K articles | 4 s, ~500 MB | fine — linear, still seconds |
| BM25 corpus query (110 terms) | 172 ms | **breaks** — 1.7 s/query is not servable |
| In-impression re-rank | 0.41 ms | 4 ms — acceptable |
| MIND test pass (2.37M) | 33 min, 287 MB peak | ~5.5 h |
| EB-NeRD test pass (13.5M) | 2.7 h, 654 MB peak | **~27 h — breaks** |

**Memory is the first constraint to bind, and it did so during this work.** The EB-NeRD validation run was killed
holding four rankers' per-impression score lists for 244,647 impressions. Reducing each impression
to four floats as it is scored fixed it. Two more traps were waiting at 13.5M rows: a
`{impression_id: n_candidates}` dict (~1.2 GB) and a `seen` set in the submission validator
(~600 MB). Both became fixed-width structures — an inline assertion and an 8-byte-per-id typed
array with a single `np.unique` pass. **All three were object-per-row structures that were
invisible at MIND's scale and fatal at EB-NeRD's.** At 10×, the same class of bug reappears in the
history dictionary (807,677 users today).

**The second is per-query cost.** BM25 query time grows at ~1.4 ms per unique query term, since
each term means a postings traversal. At 10× corpus this needs WAND/block-max pruning or a
service-side shard — the flat postings walk is already the bottleneck at 172 ms.

**Sequential layout is doing the heavy lifting.** Reading EB-NeRD's test file one Parquet row
group at a time — its own physical chunking — bounds peak memory at one group regardless of file
size. Materialising the frame, as the reference notebooks do, does not survive this machine.

**Limitation, stated explicitly.** `ebnerd_large.zip` (3 GB) failed its CRC after a duplicated
download and was never re-fetched, so EB-NeRD popularity is fitted on `ebnerd_small` — 1,892
distinct clicked articles. This is a material limitation on the EB-NeRD figures and is noted wherever they appear.

## 5. Submissions

| # | Dataset | Model | Leaderboard AUC |
|---|---|---|---|
| 1 | MIND | popularity | 0.5036 |
| 2 | MIND | weighted-sum fusion (BM25 + popularity) | 0.5258 |
| 3 | MIND | point-in-time GBDT, LSA semantics | 0.5554 |
| 4 | MIND | **MiniLM semantics, skew-resistant features** | **0.5714** |
| 5 | EB-NeRD | BM25 | 0.5110 |
| 6 | EB-NeRD | point-in-time GBDT + article recency | submitted, result pending |

Four MIND submissions, improving monotonically on all four metrics, **+0.0678 AUC** in total.
Every file was validated offline before upload — line count, permutation validity, duplicate
policy — and none was rejected.

**The offset is a property of the dataset, not the harness.** Every MIND submission returned
*below* its offline figure. EB-NeRD returned *above* it — 0.5110 against an offline 0.5030. That
settles what the earlier "calibration" claim really was: not a constant of the evaluation code,
but a symptom of how far each test window sits from its training data.

**A fact I should have measured before building the EB-NeRD models.** The scorer's per-day output
reveals the test window, confirmed against the file: training ends **25 May 2023** and the test
set runs **1–8 June 2023** — a gap of seven to fourteen days, against MIND's two to eight. This
single fact explains the whole EB-NeRD picture: why frozen popularity scored 0.4429, below chance;
why article recency dominated the importance table; and why rolling-versus-frozen made no
measurable difference, since with a two-week gap no trailing window reaches live data at all. The
diagnosis was reached the long way round, by measuring feature degeneracy, when reading two
timestamps would have predicted it.

**Whether the gap-aware protocol earned its place.** It predicted MiniLM would beat LSA, and it
did. It predicted +0.0397 and the leaderboard returned +0.0160 — roughly 40% of the forecast. So
it is a better guide than the dev split, which had overstated by 0.090, but it is not calibrated
either. The residual has a stated cause: its evaluation day sits two to three days past the last
counted event where the test split reaches eight, and `MINDsmall_train` spans only six days in
total, so a fully matched window cannot be constructed from it. The defensible summary is that
offline figures here predict the *direction* of a change reliably and overstate its *size* by an
amount that grows with the test window's distance from training.

## 6. Further work

1. Fuse BM25 + LSA + popularity **per dataset**, with the weight chosen on validation — the two
   datasets already demand different mixtures.
2. Give recall@K bootstrap CIs. A 2,000-impression run produced a confident crossover claim that
   reversed at 20,000; without intervals, small gaps in that table are noise.
3. Attack coverage directly. Sub-4% catalogue exposure is the most serious defect here and no
   accuracy metric penalises it.
