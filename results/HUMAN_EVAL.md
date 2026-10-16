# Human evaluation codebook

What the raters were shown and asked, how the blinding worked, and how their
answers became the reported numbers. The files it describes sit beside it in
this directory.

| File | Role |
| --- | --- |
| `study_codebook_public.json` | Label definitions and the post-study A/B mapping for every configuration--condition pair. |
| `human_ratings_public.csv` | One row per (condition, rater, case, method): the verdict, the four observation flags, and the pair's preference. |
| `human_summary_public.json` | The reported aggregate endpoints, which `verify_human_eval.py --check` diffs its recomputation against. |
| `verify_human_eval.py` | Recomputes every reported endpoint from the ratings table alone. |

## Task

Raters saw the global prompt, the structure image, a color-coded panel showing
each target region and its assigned reference, and the two outputs labeled A and
B. For each output they gave one overall verdict --- *intact* (one natural
object), *mild* (awkward but still one object), or *severe* (not one natural
object) --- and could additionally mark four defect types: part duplication, a
missing or malformed part, an unnatural region boundary, and a foreign patch.
They then chose which output better transferred the assigned references, with a
no-preference option. All raters received identical instructions through the
interface.

After the preference response, raters could optionally select reasons for that
choice. These exploratory flags were not used in any reported endpoint and are
not included in the release.

Defect Rate, the primary endpoint, is the fraction of ratings at severity >= 1.
Severe Defect Rate is the fraction at severity 2.

## Blinding and randomization

Three things were randomized, all by SHA-256 keyed on a single per-package seed,
so each is reproducible from that seed alone and none depends on machine state.
The main study used seed 20260819 for both structural conditions, with all 60
configurations entering each package and no subset selection.

**Blinded item ID.** `sha256("{seed}:{case_id}")[:12]`. The rater never saw a
case ID, a method name, or a file path.

**A/B side assignment.** Items were ranked by `sha256("{seed}:{item_id}:order")`;
the first half received the method order as given and the second half the
reversed order. A per-item coin flip produced a 48/72 split in an early
120-item package. Ranking and splitting keeps the order unpredictable to the
rater while making the left--right balance exactly 50/50.

**Presentation order.** Items were sorted by the same hash key. In plan order
the two structural conditions sit in separate blocks 60 items apart, so a rater
would meet every Depth item before any Canny item and could anchor the second
judgment on the first. Hashing mixes the conditions and separates repeats of the
same structure image.

Method identities lived only in each package's private codebook, written at
package-build time and not consulted until rating was complete.
`study_codebook_public.json` is the de-identified form of that mapping, with
file paths, image hashes, and login names removed.

## Raters

Eleven raters, recruited by convenience sampling from laboratory and personal
acquaintances of the authors, including people reached indirectly through those
contacts. No rater is an author or contributed to developing the method.
Participation was voluntary and uncompensated. No demographic data were
collected. Raters chose their own login names, used only to track completion;
those names are excluded from the release, as is the salted credential file the
interface stored.

All 11 completed all 60 configurations under both structural conditions: 660
paired trials and 660 ratings per method per condition, and 2,640 rows in
`human_ratings_public.csv`. One further person began the Depth evaluation
without finishing and was excluded under the completeness rule. No other sheet
and no individual rating was excluded.

## Analysis

Intervals are cluster bootstrap over the source structure images --- the cluster
is the structure image, not the rating --- with 10,000 resamples, seed 0, and
percentile 95% intervals. Paired differences are bootstrapped on the same
resamples. Raters are not resampled, so the intervals are conditional on this
panel of eleven and do not estimate uncertainty over a broader rater population.
No hypothesis-test decisions are reported.

```bash
python results/verify_human_eval.py --check
# all endpoints match the published summary
```

## Language

The study was conducted in Korean, the raters' first language. The definitions
above are English translations of the same wording. The released tables decode
the blinded A/B labels and replace login names with anonymous rater indices;
their columns and values are documented in `study_codebook_public.json`.

## Consent

Raters were informed that their de-identified responses would be shared. The
public table contains anonymous rater indices, configurations, and the response
fields used in the reported analysis, with no login name, free-text note, or
other identifying field.
