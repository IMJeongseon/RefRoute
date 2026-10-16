# What each file implements

Paths are as they sit in the repository, so `experiments/generate_variants.py` resolves
`from src import run` unchanged. Local absolute paths have been rewritten;
nothing else is modified.

## The method

| File | Role |
| --- | --- |
| `src/run.py` | RefRoute itself. Builds the reference-token sets (`redux_tokens`, `foreground_tokens`), maps target parts to references (`target_ownership`), and applies the regional attention bias (`regional_bias`) over the sequence `[text \| R_1 .. R_K \| image]`: every reference column is closed, then the image rows owned by region k are reopened on the columns of reference k. The same bias is applied in all 57 blocks and at every sampling step. Also carries the token-budget paths (`allocate_equal_budget`, `farthest_point_tokens`) used only in the appendix sweep. |
| `src/prepare.py` | Derives the inputs each run consumes from the collected images: Depth-Anything-V2-Small depth maps, Canny edges, and BiRefNet reference foreground masks. `-Large-hf` is selectable but was not used for any reported run. Its `extract_target_masks` turns a colour overlay into target-part masks by the dominant channel of (overlay - source); this is the path for preparing new inputs, and is not the parser that produced the frozen benchmark masks --- see `../benchmark/BENCHMARK.md`. |

## Running it

Every reported run was produced by `experiments/generate_variants.py` over the
plan file, never by `src/run.py`'s single-case entry point:

```
python code/experiments/generate_variants.py \
    --variant refroute --plan <plan>.json --n-values 3 \
    --control-type depth                       # Canny adds --guidance 30
```

A plan rebuilt by `experiments/manifest_to_plan.py` already carries the
reported 20 steps, so `--steps` is only needed to depart from it.

`src/run.py` is imported by that driver as a module; its own `--config` flag is a
convenience for one configuration at a time and was not used for any reported
result. Rebuild the plan file from the released metadata with
`experiments/manifest_to_plan.py` below.

| File | Role |
| --- | --- |
| `experiments/generate_variants.py` | Driver for RefRoute and its ablations over a plan file. `--variant` selects `refroute`, `depth_only`, `global_redux`, `all_reference_tokens` (no foreground filtering), `full_image_block` (condition and image state both isolated), and the diagnostic variants `one_wrong_reference`, `random_tokens`, `first_tokens`, `mismatched_depth`. `--reference-query-scope own_region` selects the diagnostic variant that restricts reference-token queries to their assigned region. A CLI `--steps` or `--guidance` overrides the plan; Canny runs used `--guidance 30`. |
| `experiments/control_conditions.py` | Shared depth and Canny control handling, and the control-image cache. |
| `experiments/prompts.py` | The domain nouns and the four scene settings that compose every global prompt. |

## Turning the released metadata back into an input

| File | Role |
| --- | --- |
| `experiments/manifest_to_plan.py` | Rebuilds the plan file from `../benchmark/manifest_60.json` with `--out`. Its `--verify` mode diffs a reconstruction against an existing plan; run against the plan behind the reported runs it reported no difference, which is how the metadata was shown to be complete. That plan is not in this bundle, so `--verify` is a property of the metadata rather than a check you can repeat here. |

## The analysis scripts

They live in `../results/`, beside the tables they read:
`../results/analyze_metrics.py` for the automatic-metric aggregates and
`../results/verify_human_eval.py` for the human endpoints. The runtime
measurements and their table verifier are also in `../results/`.

## What is deliberately absent

The metric implementations, baseline drivers, rating interface, benchmark mask
parser, runtime measurement drivers, and other experiment runners are outside
the stated scope and are not included. Their shareable outputs are provided in
`../results/`: per-configuration metrics, anonymized human responses, and the
runtime measurements underlying the reported table. The masks remain withheld
because they derive from non-redistributable images. DreamRenderer* is defined
in the manuscript rather than shipped as baseline code.

## Model weights

Downloaded from their published sources and not redistributed:
`black-forest-labs/FLUX.1-Depth-dev`, `FLUX.1-Canny-dev`, `FLUX.1-dev` for the
shared text encoders and VAE, `FLUX.1-Redux-dev` for reference features,
`ZhengPeng7/BiRefNet` for reference foreground masks, and
`depth-anything/Depth-Anything-V2-Small-hf` for the depth controls. All
pretrained weights remain frozen; RefRoute trains nothing.
