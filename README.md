# reelrank

A two-stage hybrid movie recommender. Given a user, a few liked titles, or a
free-text request like "a slow-burn sci-fi like Arrival but funnier", it returns
a ranked list of movies (including current and trending releases) with a short
reason for each pick.

It is built the way production recommenders are: a fast retrieval stage that
narrows tens of thousands of movies down to a few hundred candidates, then a
ranking stage that orders them carefully.

```
                 ┌─────────────────────────── stage 1: retrieval ───────────────────────────┐
  user / liked   │   user encoder ─┐                                                          │
  titles / text ─┼──▶ or text   ──▶│ query vector ──▶  Proxima HNSW index  ──▶ ~200 candidate │
                 │   encoder       │  (cosine / ip)    (SQ8: int8 + re-rank)     movies       │
                 │   item encoder ─┘                                                          │
                 └───────────────────────────────────────────────────────────────────────────┘
                                                                          │
                 ┌─────────────────── stage 2: ranking ───────────────────▼─────┐
                 │   neural ranker over user/item/cross features ──▶ final order │
                 └───────────────────────────────────────────────────────────────┘
```

Retrieval uses [Proxima](https://github.com/sushantlokhande14/proxima), an HNSW
vector search engine written from scratch in C++, as the ANN index. Its SQ8 mode
(int8 graph traversal with a float32 re-rank) keeps the serving memory small,
which is what makes a live demo affordable on a free tier.

## Status

This is built in the order a real system would be: get a measured MVP working,
then layer on quality and features. Honest checklist of where it is:

- [x] MovieLens pipeline with a leakage-free temporal train/val/test split
- [x] Ranking metrics (Recall@K, NDCG@K, MAP) with unit tests
- [x] Popularity baseline
- [x] Two-tower retrieval (in-batch-negative softmax, logQ correction) over Proxima
- [x] Content embeddings, content-similarity baseline, hybrid item tower, cold-start eval
- [ ] Neural ranker (stage 2)
- [ ] TMDB live catalog + daily index refresh
- [ ] Natural-language vibe search (optionally via the Relay LLM gateway)
- [ ] FastAPI backend + React/TypeScript frontend
- [ ] Dockerized, deployed live demo

## Results so far

Numbers below are from the development dataset (`ml-latest-small`, ~33k positive
interactions, per-user temporal holdout, 595 eval users). They establish that the
two-tower retrieval beats both baselines, and that adding content helps. The
headline run on `ml-25m` with a strict global temporal split is pending and will
be reported here when it lands; no number is claimed that has not been measured.

| model | Recall@10 | NDCG@10 | MAP@10 | Recall@100 |
|---|---|---|---|---|
| popularity | 0.0410 | 0.0367 | 0.0192 | 0.2311 |
| content similarity | 0.0349 | 0.0261 | 0.0132 | 0.1799 |
| two-tower (collaborative) | 0.0570 | 0.0450 | 0.0219 | 0.3465 |
| two-tower + content (hybrid) | **0.0598** | **0.0499** | **0.0246** | **0.3673** |

Full K-sweep is written to `artifacts/<dataset>/metrics.md` by the training run.

**Cold start.** To test whether the system can recommend movies it has no ratings
for, a fifth of the items are withheld from training entirely and then represented
from content alone (the path a fresh TMDB title with zero ratings takes), versus a
control that gives those items a random vector. Measured on the never-trained
items only:

| representation | Recall@50 | Recall@100 |
|---|---|---|
| content (hybrid) | 0.0973 | 0.1854 |
| no content (control) | 0.0112 | 0.0175 |

Content gives roughly ten times the recall on cold items. `python
scripts/evaluate_cold_start.py` reproduces it.

## Methodology

**Temporal split (no future leakage).** Two protocols, both leakage-free:

- `global`: one timestamp cut across all interactions. The latest interactions
  become the test set, so the model is always asked to predict the future. This
  is the strict protocol used for the headline `ml-25m` numbers.
- `user`: a per-user temporal holdout (each user's most recent interactions are
  held out). A user's own future never trains their past, and every active user
  contributes to evaluation, which keeps metrics stable on small data. Used for
  the dev dataset.

**Vocabulary from training only.** User and item ids are built from the training
split alone. A purely collaborative model cannot embed something it never saw, so
val/test rows referencing unseen users or items are dropped from this stage and
handled later as the cold-start case (content embeddings).

**Evaluation.** For each user seen in training we hold out their positive
interactions in the eval split that were not already in their training history,
exclude training-seen items from the recommendations, and score the top-K ranking
with Recall@K, NDCG@K, and MAP@K.

**Baselines to beat.** Popularity (recommend the globally most-watched titles)
and, coming next, plain content-based similarity.

## Run it

Requires Python 3.10+. Proxima is the ANN index and is a hard dependency; install
it from the sibling repo.

```bash
pip install -e ".[train,dev]"     # core + training + test deps
pip install ../proxima            # the Proxima HNSW extension

pytest                            # unit tests for metrics, split, retrieval

python scripts/evaluate_baselines.py  --config config/default.yaml   # popularity + content
python scripts/train_two_tower.py     --config config/default.yaml   # train + evaluate
python scripts/evaluate_cold_start.py --config config/default.yaml   # cold-start eval
python scripts/train_two_tower.py     --config config/ml25m.yaml     # headline run
```

The first run downloads MovieLens into `data/raw/` and caches the processed split
under `data/processed/`. Trained embeddings and the Proxima index land in
`artifacts/<dataset>/`.

## Layout

```
config/                 default (dev) and ml25m (headline) configs
src/reelrank/
  config.py             typed, YAML-backed configuration
  data/                 MovieLens download, filtering, temporal split
  models/               two-tower encoders + content-aware hybrid item tower
  features/             content documents + sentence embeddings
  training/             in-batch-negative training loop
  retrieval/            Proxima index wrapper + stage-1 recommender
  baselines/            popularity + content similarity
  eval/                 metrics + leakage-free harness
scripts/                runnable entry points
tests/                  pytest
```

## Credits

- [Proxima](https://github.com/sushantlokhande14/proxima): the C++ HNSW vector
  search engine used for retrieval.
- [Relay](https://github.com/sushantlokhande14/Relay): LLM gateway, planned for
  natural-language query understanding and explanations.
- [MovieLens](https://grouplens.org/datasets/movielens/) for interactions and
  [TMDB](https://www.themoviedb.org/) for the live catalog (planned).

MIT licensed.
