# RefRoute release bundle

This bundle contains the shareable artifacts promised by the paper and the
measurements needed to verify its reported runtime table. Two sentences define
the core release scope.

From the Limitations:

> We instead provide the generation code and shareable evaluation metadata.

From the Appendix:

> We instead provide the per-configuration metric values, anonymized human
> evaluation ratings and evaluation codebook, and the analysis scripts used to
> compute the reported aggregate results.

```
release/
  validate_release.py one-command data and consistency audit
  requirements.txt   pinned versions the reported runs used
  benchmark/         the evaluation metadata: 60 configurations, image inventory
  code/              the generation code
  results/           metrics, ratings, runtime records, codebook, analysis scripts
```

Run the complete data-only audit from this directory with:

```bash
python validate_release.py
```

The audit requires only Python and NumPy. It checks the benchmark composition,
all metric-run coverage, the anonymized human records, the reconstructed plan,
and the reported human and runtime aggregates.

## Generation code

`code/src/run.py` is RefRoute: reference encoding through FLUX.1-Redux-dev,
foreground token selection on the Redux grid, the additive routing bias, and
structure-conditioned sampling with FLUX.1-Depth-dev or FLUX.1-Canny-dev.
`code/src/prepare.py` derives the control images and masks each run consumes.
`code/experiments/generate_variants.py` is the driver that produced every
reported run, over a plan file rebuilt from the metadata below;
`code/src/run.py` is imported by it as a module. `code/CODE_MAP.md` says what
every file does.

The reported protocol is 1024x1024 output, 20 sampling steps, seed 0, guidance
10 for Depth and 30 for Canny, reference scale lambda = 3, foreground-token
threshold tau = 0.3, and no reference-token cap. It is recorded per
configuration in the manifest, and once under its `reported_protocol` key.

## Evaluation metadata

`benchmark/manifest_60.json` gives, for each of the 60 configurations, the case
identifier, domain, scene, the exact global prompt, the structure image and its
SHA-256, the part-to-reference assignment with each reference image and its
SHA-256, the routing order, the seed, and every sampling setting.
`benchmark/image_inventory.csv` lists all 42 images with their digests and the
roles each one plays.

The metadata is complete rather than summary.
`code/experiments/manifest_to_plan.py` turns it back into the plan file the
generation code consumes:

```bash
python code/experiments/manifest_to_plan.py --out plan.json
# wrote plan.json (60 configurations)
```

The emitted plan carries the reported protocol, so running the generation code
against it reproduces those settings without further flags; Canny runs add
`--guidance 30`.

The same script has a `--verify` mode that compares such a reconstruction
against an existing plan field by field. That is the check we ran against the
plan used for the reported runs, and it reported no difference; it needs that
original plan, so it is a statement about the released metadata rather than
something this bundle lets you re-run.

`results/analyze_metrics.py` also reads `benchmark/manifest_60.json`, for the
structure image behind each case, which is the unit its bootstrap resamples.

`benchmark/BENCHMARK.md` describes the composition, the canonical asset layout,
and how the masks and control images were produced.

## Metric values, ratings, and analysis

| File | Contents |
| --- | --- |
| `results/per_case_metrics.csv` | Per-configuration values for the six metrics reported in the manuscript: 60 complete run groups, keyed by case ID and output name. |
| `results/reported_run_map.json` | Mapping from manuscript comparisons to the corresponding run prefixes in the metric table. |
| `results/human_ratings_public.csv` | 2,640 rows: one per (condition, rater, case, method). Raters appear only as `R1`--`R11`; method labels are the post-study decoded labels. |
| `results/human_summary_public.json` | Aggregate endpoints of the human evaluation. |
| `results/study_codebook_public.json` | Task definitions and the post-study A/B mapping, without private paths or identifiers. |
| `results/analyze_metrics.py` | Means, paired per-configuration differences, and cluster-bootstrap intervals over `per_case_metrics.csv`. |
| `results/verify_human_eval.py` | Recomputes every reported human endpoint from `human_ratings_public.csv` alone and diffs the result against `human_summary_public.json`. |
| `results/runtime_matched.json`, `results/runtime_dreamrenderer.json` | Measurements underlying the reported runtime table. |
| `results/verify_runtime.py` | Recomputes the runtime means and speedups and checks them against the manuscript values. |
| `results/HUMAN_EVAL.md` | The task, the blinding and randomization, the rater panel, and the bootstrap settings. |

The analysis and verification scripts read only released files:

```bash
python results/analyze_metrics.py --compare depth_refroute depth_dreamrenderer_fg
python results/verify_human_eval.py --check
python results/verify_runtime.py --check
```

`results/study_codebook_public.json` and `results/HUMAN_EVAL.md` are the
evaluation codebook: the label definitions and the post-study A/B mapping for
all 120 configuration--condition pairs, and the account of what the raters were
shown and asked, how the blinding and the A/B side assignment were derived, and
which bootstrap the intervals use.

## Not included

**The benchmark images and their derived masks**, as the paper states: the
structure and reference images, the target-part and reference foreground masks,
the annotation overlays, and the depth and Canny control images. Reproducing the
benchmark requires obtaining the 42 images. Their collection URLs were not
recorded and cannot be recovered, and the SHA-256 digests in
`benchmark/image_inventory.csv` do not stand in for them: a digest cannot be
searched back to an image, and re-downloading the same photograph will not
reproduce the bytes of our hand-prepared file. What the digests do is pin the
benchmark to 42 specific, immutable files, so it cannot be altered after the
fact. `benchmark/BENCHMARK.md` sets out exactly what they can and cannot
establish.

**Rater login names**, the salted credential file, the raw per-rater sheets, and
the private A/B codebooks. The released tables contain the decoded response
fields needed for analysis, with raters replaced by `R1`--`R11`; they contain no
login name, item hash, free-text note, credential, or private file path.

**Model weights**, which are downloaded from their published sources.

The metric implementations, baseline drivers, rating interface, benchmark mask
parser, runtime measurement drivers, and other experiment runners are not part
of this bundle. Their released outputs and the procedures needed to interpret
them are provided here and in the manuscript.

## Environment

Python 3.12 and CUDA 12.8; the reported runs used a single RTX PRO 6000
Blackwell. `requirements.txt` pins the versions those runs used; install the
CUDA build of `torch` and `torchvision` from the PyTorch index. The data-only
audit and analysis scripts require only NumPy and run without a GPU or model
weights.

## Consent

Raters were informed that their de-identified responses would be shared. The
public human-study table contains anonymous rater indices, configurations, and
the response fields used in the reported analysis, with no login name,
free-text note, or other identifying field. Optional exploratory preference
reasons are not released.
