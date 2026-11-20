# Experiment: all cohort users and padding instead of filler tokens (`--train-pool cohort --pad-tokens`, `runs/relgt_poolpad_ho0927.ps1`)

[relgt_pool](relgt_pool.md) trains on every cohort user. This adds a second change to the token sampler, for users
that have fewer than K = 100 tokens: stop adding nodes when no neighbour is left, instead of filling the slots.

## What the sampler did before

`sample_walk` (chain graph) and `sample_tokens` (fk graph) in `experiments/relgt/run.py` always return K tokens:
- a user with some tokens but fewer than K: the remaining slots are **random repeats** of the user's own tokens
  (`pick = rng.random(...) * cnt`);
- a user with no neighbour at all: the K - 1 slots are **random visible nodes of the whole graph**, hop = the
  fallback hop.

The ~3,000 cohort users that `--train-pool cohort` adds are mostly of the second kind or close to it.

## What `--pad-tokens` does

- Sampler: slots after the user's real tokens are padding. The slot keeps the fallback hop as its marker
  (`max_hops + 1` on the chain graph, `MAX_HOP` on the fk graph), `paths = N_PATHS + 1`, and any valid node id
  (the seed), so the row encoder still gets a valid index. Padding is always a suffix.
- Subgraph edges: `subgraph_edges` and `action_chain` take a `pad` mask; a padding slot has no key and no edges.
- Model (`RelGT`, `pad=True`): `pad = hops == max_hop`; attention gets a `-inf` bias on padding keys; the readout in
  `LocalModule` gives padding slots zero weight (finite fill value and a zero multiplier, so an example whose
  neighbours are all padding reads out its seed token and gets no NaN). The global codebook attention uses only
  the seed query, so it is unchanged.
- Cache: a `("pad",)` entry joins the token cache key, so the tokens are sampled again for every week (~10 min once).
  Default runs keep their key and their numbers.
- Check (scratch script, 600 users, week 09-19): 144 of 600 users had padding slots (mean 10.1 over all, so ~42 for
  those that have it); padding is a suffix of every example; no subgraph edge touches a padding slot; a forward and
  backward pass on 64 examples, one of them with only padding neighbours, gave finite outputs and gradients.

## Setup

- `relgt_chain` of `runs/ensemble_ho0927.sh` plus `--train-pool cohort --pad-tokens`; holdout week 09-26 16:00 UTC,
  seeds 0-2, run names `relgt_chain_poolpad_ens0926_s<seed>`.
- Comparison: `experiments/ensemble/run.py` with the reference runs `relgt_chain_ens0926_s0-2` and
  `relgt_chain_pool_ens0926_s0` (all users, default fill) when it exists. Compare `bag:relgt_chain_poolpad` with
  `bag:relgt_chain_pool` for the effect of padding alone, and with `bag:relgt_chain` for both changes.

```powershell
.\runs\relgt_poolpad_ho0927.ps1              # seeds 0-2 (missing ones) on GPUs 0 and 1, then the comparison
.\runs\relgt_poolpad_ho0927.ps1 -Smoke       # pipeline check
```

## Limits

- Padding changes the BatchNorm statistics of the transformer's feed-forward layers, which are taken over all token
  positions, so padding slots enter them; they are not removed from the batch.
- One holdout week; 3 seeds per config, but only 1 for `relgt_chain_pool`; seeds span 1.4-2.1 points.
- The model is told by the hop id only; no separate learned padding embedding, because the attention and readout
  masks make the slot invisible to the seed.

## Results

Run `.\runs\relgt_poolpad_ho0927.ps1`, seeds 0-2 (`relgt_chain_poolpad_ens0926_s0-2`, best epochs 11, 12, 14).
Holdout week 09-26 16:00 UTC, 6,064 users; tune week 09-19, 6,412 users. Official 6-activity score. `relgt_chain_pool`
has only seed 0 (its seeds 1 and 2 were not run), so every comparison with it rests on that one seed.

### The runs' own decisions

| | Score, mean (min-max) | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|---|
| `relgt_chain_poolpad`, validation model (3 seeds) | 51.65 (51.21-52.13) | 0.355 | 0.432 | 0.410 | 0.470 | 0.415 | 0.673 |
| `relgt_chain_pool`, validation model (1 seed) | 51.17 | 0.342 | 0.391 | 0.409 | 0.476 | 0.433 | 0.675 |
| `relgt_chain`, validation model (3 seeds) | 52.05 (52.00-52.14) | 0.404 | 0.426 | 0.391 | 0.451 | 0.446 | 0.675 |
| `relgt_chain_poolpad`, final model (3 seeds) | **53.34** (52.89-53.82) | 0.405 | 0.464 | 0.439 | 0.486 | 0.412 | 0.683 |
| `relgt_chain_pool`, final model (1 seed) | 52.63 | 0.400 | 0.454 | 0.425 | 0.471 | 0.436 | 0.676 |
| `relgt_chain`, final model (3 seeds) | 52.59 (51.72-53.04) | 0.402 | 0.462 | 0.407 | 0.447 | 0.443 | 0.681 |

### Same decision rule on the same users (`experiments/ensemble/run.py relgt_chain_poolpad relgt_chain_pool relgt_chain`)

Shares from the tune week of the combined probabilities, applied to the holdout week (final-model probabilities). The
reference for the bootstrap is `bag:relgt_chain_poolpad` (best tune-week score).

| Variant | Score | Tune | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | vs reference [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| seed:relgt_chain_poolpad_s0 | 53.32 | 52.17 | 0.409 | 0.472 | 0.439 | 0.481 | 0.391 | 0.685 | |
| seed:relgt_chain_poolpad_s1 | 52.67 | 52.25 | 0.416 | 0.447 | 0.424 | 0.483 | 0.405 | 0.680 | |
| seed:relgt_chain_poolpad_s2 | 54.10 | 52.06 | 0.464 | 0.472 | 0.454 | 0.494 | 0.436 | 0.677 | |
| seed:relgt_chain_pool_s0 | 52.50 | 51.94 | 0.441 | 0.454 | 0.425 | 0.471 | 0.425 | 0.673 | |
| seed:relgt_chain_s0 | 53.85 | 51.01 | 0.444 | 0.469 | 0.437 | 0.495 | 0.456 | 0.682 | |
| seed:relgt_chain_s1 | 53.01 | 51.53 | 0.384 | 0.481 | 0.412 | 0.482 | 0.442 | 0.683 | |
| seed:relgt_chain_s2 | 51.72 | 51.42 | 0.407 | 0.434 | 0.421 | 0.411 | 0.433 | 0.679 | |
| bag:relgt_chain_poolpad (3 seeds) | 53.82 | 52.75 | 0.409 | 0.467 | 0.443 | 0.488 | 0.444 | 0.686 | 0 |
| bag:relgt_chain_pool (1 seed) | 52.50 | 51.94 | 0.441 | 0.454 | 0.425 | 0.471 | 0.425 | 0.673 | -1.32 [-2.13, -0.52] |
| bag:relgt_chain (3 seeds) | **54.42** | 51.63 | 0.434 | 0.481 | 0.461 | 0.502 | 0.454 | 0.684 | +0.60 [-0.15, +1.30] |
| mean (the three bags) | 53.84 | 52.91 | 0.414 | 0.482 | 0.439 | 0.464 | 0.449 | 0.686 | +0.01 [-0.54, +0.55] |
| select | 53.98 | 52.66 | 0.441 | 0.467 | 0.443 | 0.471 | 0.454 | 0.686 | +0.16 [-0.39, +0.74] |

The official scorer reproduces `mean` (53.84). `select` took open_pr and pr_review from `relgt_chain_pool`, open_issue,
issue_comment and push from `relgt_chain_poolpad`, and pr_review_comment from `relgt_chain`.

### Reading

- **Padding helps the all-users model.** Single seeds on the common rule: `poolpad` 53.32, 52.67, 54.10 (mean 53.36)
  against 52.50 for the one `pool` seed (seed 0 against seed 0: +0.82). The two runs differ only in `--pad-tokens`. With
  one `pool` seed, and a spread of 1.4 points between the `poolpad` seeds, this is suggestive, not shown. The `pool`
  seeds 1 and 2 would be needed.
- **It does not beat the default model.** Mean of single seeds: `poolpad` 53.36 (52.67-54.10) against `relgt_chain`
  52.86 (51.72-53.85), +0.50 with overlapping ranges. As 3-seed bags `poolpad` is 53.82 against 54.42 for `relgt_chain`:
  0.60 lower (the bootstrap gives `bag:relgt_chain` +0.60 [-0.15, +1.30] over `bag:relgt_chain_poolpad`; the interval
  includes 0). The final model's own decisions are 53.34
  against 52.59 (+0.75, the validation models: 51.65 against 52.05, -0.40). Nothing separates the two on this week.
- **Bagging helps `poolpad` less.** Bag against its seed mean: +0.46 (53.82 against 53.36), `relgt_chain` +1.56
  (54.42 against 52.86). The `poolpad` seeds also agree more (spread 1.4 against 2.1 points). The 3-seed bag of
  `relgt_chain` is still the best number, as in [ensemble](ensemble.md) and [relgt_pool](relgt_pool.md).
- **Per activity** (bags, `poolpad` against `relgt_chain`): open_pr 0.409 / 0.434, open_issue 0.467 / 0.481,
  issue_comment 0.443 / 0.461, pr_review 0.488 / 0.502, pr_review_comment 0.444 / 0.454 are lower; push 0.686 / 0.684
  is level. Only push is not lower.
- **The tune week ranks the models wrongly once more.** `poolpad` has the best tune-week score of every run (52.06-52.25
  per seed, 52.75 as a bag, against 51.0-51.9 for the rest), yet its holdout bag (53.82) is below `bag:relgt_chain`
  (54.42, tune 51.63). `select` and `mean` give 53.98 and 53.84, so mixing the bags adds nothing.
- **Cost:** the padding mask makes attention slower per step (about twice in the 30-step smoke run), and tokens are
  sampled again (~8 min a seed).
- One holdout week; `pool` has one seed; the tokens differ from the reference runs by the random stream as well as
  by the padding.

**Conclusion:** padding makes up for the all-users run (+0.8 on seed 0) but gives no gain over the default eligible-users
model, and the default's 3-seed bag is still the best number. Not adopted; the WRITEUP.md table and the submission
pipeline are unchanged. What is not known: whether padding alone helps the eligible-users model (not tried: the
`--pad-tokens` option works without `--train-pool cohort`), and the `pool` seeds 1 and 2.
