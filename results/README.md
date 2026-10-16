# Released results

## Automatic metrics

`per_case_metrics.csv` contains 60 run groups with 60 configurations per run.
It includes only runs that contribute to quantitative results in the
manuscript and only the six reported metrics: Region DINO, DINO Assignment
Accuracy, VGG Gram Distance, FG IoU, Scene CLIP, and CLIP-IQA. Metric families
computed in separate passes use the suffixes `_appearance` and `_spatial`;
empty cells indicate that a metric belongs to the other family.
`reported_run_map.json` maps every retained run prefix to its manuscript
comparison. Use `analyze_metrics.py` to reproduce means, paired differences,
and cluster-bootstrap intervals that resample source structure images.

## Human evaluation

`human_ratings_public.csv` contains 2,640 per-output ratings from 1,320 paired
trials and the response fields used in the reported analysis. Optional
preference-reason flags were exploratory, were not analyzed, and are not
released. Label definitions and missing-value rules are in
`study_codebook_public.json`; the full protocol is in `HUMAN_EVAL.md`.

## Runtime

`runtime_matched.json` and `runtime_dreamrenderer.json` contain the measurements
underlying the runtime table. `verify_runtime.py --check` recomputes the reported
means and speedups.

Run all release-level checks from the bundle root:

```bash
python validate_release.py
```
