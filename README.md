# mountpoint

**Training-free articulation estimation for functional elements in 3D scans.**

Telling a robot to *"open the bedroom door"* needs more than the handle's
location. It needs to know the handle **rotates**, about **which axis**, and
through **which point in space**. Work on 3D functionality understanding mostly
stops at the mask.

`mountpoint` estimates those parameters with no training, no per-scene
optimisation, and no constant fitted to a reported score. The name is the method:
a functional element carries almost no motion information itself, so we read the
motion off the surface it is **mounted on**. Fit a plane to a shell of
neighbourhood points around the element — deliberately excluding the element's
own points — and the mounting surface plus the scan's gravity axis determine the
type, the axis, and the origin.

Removing exactly that one idea, and nothing else, costs **26.6 motion-gate
points** and drives revolute direction error from 8.45° to 74.09°.

---

## Results at a glance

**Cross-dataset transfer.** The rules were designed on SceneFun3D and moved to
Articulate3D — different scanner, different object scale — with no retuning, and
scored by **USDNet's own released evaluator**:

| Articulate3D validation (n=389 parts, 41 scenes) | axis ≤15° | origin <0.25 m | composed |
|---|---|---|---|
| **mountpoint** (training-free) | **93.8%** [87.9, 97.7] | 77.4% [70.8, 84.0] | **83.0%** [75.8, 88.2] |
| USDNet (supervised) | 82.8% | 75.1% | 59.8% |

Brackets are 95% scene-clustered bootstrap intervals. **Read the asymmetry
before quoting these**: our figures are on ground-truth part masks, while
USDNet's are retention on its own predicted masks. Axis and the composed gate
separate clearly; **origin does not** — 75.1% falls inside our interval, so we do
not claim an origin win. The hybrid run that removes the asymmetry is listed
under *Not in this repo* below.

**Robustness to the segmenter.** Because the motion is never read off the
element, corrupting the element barely moves the result (n=734, 3 reps):

| corruption | mountpoint | control (reads the element's own normal) |
|---|---|---|
| dilate region to 2× element extent | 60.9 (−1.6) | 36.5 (−0.5) |
| discard 50% of region points | 62.8 (**+0.3**) | 37.4 (+0.3) |
| centroid drift of 2 cm | 58.5 (−4.1) | 37.1 (+0.1) |

The control's most diagnostic row is one not shown: contaminating the element
with its surroundings *improves* it by **12.9 points**, because contamination
accidentally hands it the mounting surface it was denied.

---

## Install

```bash
git clone <this repo> && cd mountpoint
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tests/test_smoke.py        # must print OK
```

`requirements.txt` covers the motion module, every evaluation script, and every
figure — all CPU, no GPU, no dataset. The segmentation front end is separate and
optional (`requirements-pipeline.txt`); it needs CUDA.

**Run the smoke test first.** Every number in the paper comes from data that
cannot be redistributed, so without it a broken install produces
plausible-looking numbers and no signal that anything is wrong. It builds a
synthetic door from geometry alone and asserts the hinge is vertical and on the
far edge from the handle.

## Data layout

Nothing here assumes a particular home directory. Every path resolves from the
environment:

```bash
export SF3D_ROOT=/path/to/scenefun3d          # required for dataset scripts
export SF3D_TOOLKIT=/path/to/scenefun3d-toolkit   # only for the official evaluator
export ARTICULATE3D_ROOT=/path/to/articulate3d    # only for the transfer scripts
```

`$SF3D_ROOT` is expected to look like:

```
$SF3D_ROOT/
├── data/<visit>/<visit>_laser_scan.ply
│                <visit>_annotations.json
│                <visit>_motions.json
├── annot_points_r11/<visit>.npz     built by scripts/build_annot_caches.py
├── annot_origins/<visit>.npz                  "
├── frames/                          posed RGB-D, front end only
└── exemplar_db.pkl                  built by scripts/build_exemplars.py
```

SceneFun3D and Articulate3D are obtained from their own authors under their own
licences; we redistribute neither. The two caches are built locally:

```bash
python scripts/build_annot_caches.py --data $SF3D_ROOT/data \
    --pts-dir $SF3D_ROOT/annot_points_r11 \
    --org-dir $SF3D_ROOT/annot_origins --radius 1.10 --cap 120000
```

The 1.10 m radius is not arbitrary: at 0.60 m, 37% of true hinge origins fall
outside the neighbourhood entirely.

## Reproducing the paper

Each row regenerates the JSON it names in `docs/`, which is the file the paper
quotes. Scripts marked **CPU** need no GPU.

| paper | script | needs | CPU |
|---|---|---|---|
| §4.1 edge-selection rule | `sf3d_edge_selection.py`, `sf3d_edge_heldout.py` | `$SF3D_ROOT` | ✓ |
| §4.1 shell geometry | `shell_geometry_study.py` | `$SF3D_ROOT` | ✓ |
| §4.2 signed direction | `sign_direction.py` | `$SF3D_ROOT` | ✓ |
| §5 Articulate3D transfer | `articulate3d_transfer.py` | `$ARTICULATE3D_ROOT` | ✓ |
| §5 official re-scoring | `articulate3d_official_metric.py` | `$ARTICULATE3D_ROOT` | ✓ |
| §5 panel degradation | `articulate3d_stress.py` | `$ARTICULATE3D_ROOT` | ✓ |
| §5 flush-door diagnosis | `sf3d_panel_headroom.py` | `$SF3D_ROOT` | ✓ |
| §6 robustness sweep | `segmenter_robustness.py` | `$SF3D_ROOT` | ✓ |
| §6 Fun3DU readout fault | `diag_fun3du_readout.py` | nothing | ✓ |
| §6 Fun3DU re-threshold | `fun3du_threshold_sweep.py` | Fun3DU stage-4 output | ✓ |
| §7.1–7.2 composed metric | `composed_metric_gt.py`, `composed_metric_predicted.py` | `$SF3D_ROOT`, `$SF3D_TOOLKIT` | ✓ |
| §7.3 detection AP/AR | `ap_confidence_and_ar.py` | predictions, `$SF3D_TOOLKIT` | ✓ |
| §7.4 failure analysis | `pick_examples.py` → `fig_failures.py` | `$SF3D_ROOT` | ✓ |
| §9 ablations | `motion_ablations.py` | `$SF3D_ROOT` | ✓ |
| all confidence intervals | `confidence_intervals.py` | `$SF3D_ROOT` | ✓ |
| Figures 2–5 | `fig_insight.py`, `fig_qualitative.py`, `fig_failures.py`, `fig_robustness.py` | `$SF3D_ROOT` | ✓ |

The segmentation front end (`sf3d/pipeline.py`) is the one GPU component, and it
is deliberately not the contribution — §6 exists to show the motion module does
not depend on it.

## Weights that cannot ship

| artefact | size | where |
|---|---|---|
| DINOv3 ViT-L/16 | 1.1 GB | Hugging Face, **gated** — request access |
| SAM ViT-B | 358 MB | Meta's Segment Anything release |
| GroundingDINO-tiny | 690 MB | Hugging Face, ungated |
| `exemplar_db.pkl` | 413 MB | build locally, `scripts/build_exemplars.py` |

## Why paths come from the environment

An earlier version of this code hardcoded a Google Drive path, and the package
lived inside a directory the repository ignored. A working copy drifted from the
repository and silently reverted two bug fixes — one of which put a sampled
point on the wrong image row and cost a full detection baseline before anyone
could diff the two copies, because one of them was not under version control.

Hence: one package, at the repository root; one data root, named by
`$SF3D_ROOT`; and `scripts/_paths.py` as the only place a path is resolved.

## Layout

```
sf3d/          the method. motion.py is the contribution.
scripts/       one script per paper claim; _paths.py resolves every path
docs/          measured results as JSON, the paper draft, and figures
notebooks/     development notebooks, outputs stripped
tests/         synthetic smoke test, no data required
```

## Not in this repo

Honest about what is outstanding rather than silent about it:

- **The USDNet hybrid** — their predicted masks, our axis and origin, all rows
  through one metric implementation. This is what turns §5 from evidence into a
  claim, and it needs a CUDA 12.1 Docker build.
- **Our split0 detection row.** The detection table is headed split0 and our row
  is an estimate on a 48-scene set.
- **Fun3DU as a second segmenter** — diagnosed (`diag_fun3du_readout.py`), not
  yet delivered.
- **Real-robot validation.** A LoCoBot demonstration is the claim's true
  discharge.
- Exploratory sweeps, diagnostics and measured-negative experiments live in the
  private working repository; everything backing a number in the paper is here.

## Citation

Paper under submission; a BibTeX entry will replace this on acceptance.

## License

MIT — see [LICENSE](LICENSE). SceneFun3D and Articulate3D carry their own
licences and are not redistributed here.
