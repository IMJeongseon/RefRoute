# Evaluation set: what is released and how to rebuild it

The 60 configurations are released as metadata. The images and their derived
masks are not, because they were collected through web image search and we hold
no redistribution rights. Everything else that defines a configuration --- which
image plays which role, which part is assigned to which reference, the prompt,
the seed, and every sampling setting --- is here.

## Files

| File | Contents |
| --- | --- |
| `manifest_60.json` | The 60 configurations in full: case ID, domain, scene, prompt, structure image, per-part reference assignment, routing order, seed, and sampling settings. |
| `image_inventory.csv` | The 42 benchmark images: identifier, domain, SHA-256, and the roles each image plays. |

## Composition

60 configurations, built from 31 structure images across six domains --- animals,
birds, buildings, butterflies, cakes, cars --- with ten configurations per domain.
Twenty-nine structure images contribute two configurations each and two
contribute one. Each configuration has three disjoint target-part masks, three
same-domain reference images with whole-object foreground masks, a fixed
part-to-reference assignment, and one of four scene prompts (snow, autumn, moss,
desert), fifteen configurations per prompt. The 42 images are drawn from one
pool: all 31 structure images also serve as references in other configurations.

Every count above is recomputable from `manifest_60.json`.

Case identifiers are internal labels and two of them do not name the structure
they are built on: `cheetah_r2_132` and `goat_r2_132` both use the `kangaroo`
structure. Read `structure_image`, not the identifier.

## What the digests are, and are not

Collection URLs were not recorded when the images were gathered, so
`source_url` in `image_inventory.csv` is empty and cannot be filled in
retrospectively.

Each digest is the SHA-256 of the PNG file as it sits in our working tree, after
we cropped and converted it. Those files are heterogeneous --- 300x300 to
1563x1563, mostly but not exactly square, 36 RGB and 6 RGBA --- because each was
prepared by hand from its source.

That has a consequence worth stating plainly. **The digests do not let a third
party confirm that an image they obtained is the one we used.** Re-downloading
the same photograph yields the original file at its original resolution and
encoding; cropping and re-encoding it will not reproduce our bytes, and no
preprocessing recipe is documented that would make it do so. There is also no
way to search from a digest back to an image.

What the digests do give:

- **A commitment.** The released metadata names 42 specific, immutable files.
  The benchmark cannot be quietly altered after the fact by substituting an
  image, because every substitution changes a digest.
- **Distinctness.** All 42 digests differ, so no configuration silently reuses
  the same file under two names.
- **Integrity, if the files are ever transferred.** Should the images be shared
  directly, or released later under cleared terms, the digests confirm that what
  arrived is what the metadata describes.

Reproducing the benchmark still requires the images themselves. The digests
document what those images are; they do not substitute for them.

## Canonical layout

Every path in the manifest is an identifier relative to `data/`. Placing the
assets at these locations makes the manifest directly runnable:

```
data/<domain>/<stem>.png                                   structure and reference images
data/prepared/masking/target/<domain>/<stem>/target_0k.png  target part mask k, k = 1,2,3
data/prepared/masking/foreground/<domain>/<stem>.png        reference foreground mask
data/prepared/depth/<domain>/<stem>.png                     depth control image
```

Control images and foreground masks are derived, not collected:
`../code/src/prepare.py` regenerates them from the images. Depth maps come from
`depth-anything/Depth-Anything-V2-Small-hf`, bilinearly resized to 1024x1024,
min-max normalized per image with epsilon = 1e-8, and written as an 8-bit
three-channel PNG. Canny controls resize the structure image to 1024x1024 with
Lanczos, convert to grayscale, and apply OpenCV Canny with thresholds 50 and
200, aperture 3, and `L2gradient=False`, with no smoothing; the binary edge map
is repeated across three channels.

## Rebuilding the plan file

The generation code consumes a plan file, and
`../code/experiments/manifest_to_plan.py` reconstructs one from the manifest:

```bash
python code/experiments/manifest_to_plan.py --out plan.json
# wrote plan.json (60 configurations)
```

The same script has a `--verify` mode that diffs such a reconstruction against
an existing plan, field by field. Running it against the plan behind the
reported runs reported no difference, which is how we established that the
released metadata withholds only the pixels. It needs that original plan, so it
is a property of the metadata rather than a check this bundle lets you repeat.

## Sampling settings

`settings` in the manifest carries the values used for every reported run:
20 steps, guidance 10 for Depth and 30 for Canny, seed 0, reference scale
lambda = 3, foreground-token threshold tau = 0.3, and no reference-token cap.
`plan_file_defaults` records the stale 28-step default carried in the plan file
used for the reported runs, which every one of those runs overrode on the
command line. The manifest keeps both so the override is visible rather than
implied. `../code/experiments/manifest_to_plan.py` emits the reported 20 steps by default, so a plan
rebuilt from this metadata reproduces the protocol even if `--steps` is not
passed; `--plan-defaults` reproduces the original file instead.

## Mask creation

Target-part masks were produced by asking ChatGPT or Gemini to divide the
foreground object of each structure image into three semantically meaningful
parts and return a color-coded segmentation: part 1 red, part 2 blue, part 3
green. The returned overlays were used without manual correction and converted
to binary masks by a fixed parser. ChatGPT and Gemini were used only for this
annotation step; neither was involved in generation or evaluation. All masks
were frozen before any method was run, so every method receives identical masks.
The overlays are the authoritative specification of the target regions; they are
derived from the structure images and are withheld along with them.

The parser that produced these frozen masks is described in the manuscript and
is not part of the released scope. It is not the same as `extract_target_masks`
in `../code/src/prepare.py`, which is the simpler difference-based converter for
preparing new inputs. It downsamples the overlay with box filtering
to the source resolution, aligns it to the source with an ECC affine warp, and
scores each pixel on two independent signals: the per-channel difference against
the source, and, inside the reference foreground, the overlay's own hue and
saturation. The second signal recovers strokes whose color matches the content
underneath --- green paint on a yellow butterfly, red paint over multicolored
sprinkles --- which a pure difference test drops. Scores are median-filtered,
then thresholded by hysteresis: pixels scoring at least 50 seed a region, and
connected pixels scoring at least 16 are absorbed into it. Components smaller
than 96 pixels are discarded. The foreground gate disables itself for any source
whose strongly painted pixels it would cut by more than 0.5%, so the overlay
always remains the specification.

Reference foreground masks are not annotations: they come from the
`ZhengPeng7/BiRefNet` checkpoint. Each reference image is resized to 1024x1024,
ImageNet-normalized, and passed through BiRefNet; the final logits become
foreground probabilities through a sigmoid, are stored as an 8-bit soft mask,
and are resized back to the reference resolution. For reference-token selection
the mask is bilinearly resized to the square Redux token grid and flattened, and
token j is retained when its foreground probability satisfies m_j >= tau, with
tau = 0.3. No token-count cap is applied.
