# Reading Articulation off the Mounting Surface: Training-Free Motion Estimation for Functional Elements in 3D Scans

*Draft targeting **IEEE RA-L** (8 pages incl. references; rolling submission).
IROS 2027 (1 Mar) is the conference fallback. Every number below is measured and
reproducible from `scripts/`; anything not yet measured is marked ▢ and named as
such. Confidence intervals are scene-clustered bootstraps, B = 10,000
(`scripts/confidence_intervals.py`).*

---

## Abstract

Telling a robot to *"open the bedroom door"* requires more than finding the
handle. It requires knowing that the handle **rotates**, about **which axis**, and
through **which point in space**. Work on 3D functionality understanding stops at
segmentation; the motion parameters that make a mask actionable are either
ignored or learned from supervision that does not transfer to unscanned rooms.

We show that these parameters need no learning at all. Our claim is that
**articulation is determined by what an element is attached to, not by the
element's own shape.** We fit a plane to the mounting surface surrounding the
element — deliberately excluding the element's own points — and read motion type,
axis and origin off that surface together with the scan's gravity axis. There is
no training, no per-scene optimisation, and no constant fitted to a reported
score.

Evaluated on **Articulate3D** with **USDNet's own released evaluator**, the rules
reach axis accuracy **93.8%** (95% CI [87.9, 97.7]) and a composed axis-and-origin
gate of **83.0%** [75.8, 88.2], against the supervised USDNet's 82.8% and 59.8% —
having never seen that dataset, its scanner, or its object scale. The comparison
is not yet like-for-like: our figures are on ground-truth part masks and USDNet's
are retention on its own predictions, so we state the gap as evidence rather than
as a claimed win, and §7.1 is the experiment that settles it.

The module's independence from its segmenter is measured rather than asserted. A
parametric corruption sweep over 734 elements shows that dilating a region to
twice the element's extent costs **1.6 gate points** and discarding half its
points costs **none**, while the same corruptions leave a control that reads
geometry off the element itself flat at 37%. On SceneFun3D the rules retain
**61.6%** of detections through every motion gate where a fully supervised
baseline retains 29.7%, at **22.8 s per element on a 16 GB T4**.

---

## 1. Introduction

A functional element is a part of a scene a person touches to change its state: a
handle, a knob, a switch, a plug. SceneFun3D [1] annotates 14.8k of them in real
indoor laser scans, with masks, natural-language descriptions, and — uniquely —
**motion parameters**: type (rotation or translation), axis direction, and axis
origin.

That last piece is what separates scene *understanding* from scene *interaction*.
A perfect mask on a door handle tells a robot where to reach; it does not tell it
whether to pull, slide, or twist, nor about what axis. Yet the field's attention
has gone almost entirely to the mask. Of the methods reporting on SceneFun3D,
only the dataset's own supervised baseline reports motion at all.

**The elements are small.** The median annotated element measures **1.7 cm**
across. In a 1920×1440 frame of a room that is a few dozen pixels — one or two
patches of a vision transformer. This single fact shapes every design decision
that follows, and explains where the pipeline succeeds and where it does not.

### 1.1 Why training-free

The obvious approach is supervision, and SceneFun3D provides a baseline for it.
We deliberately do not, for a reason specific to the target application rather
than ideological preference: **a robot operates in rooms nobody scanned.** A model
trained on 545 annotated scenes learns what handles look like *in those scenes*.
Our motion rules are fixed geometry — a plane fit, a gravity vector, a panel edge
— which are properties of how furniture is built, not of how a dataset was
collected. §5 tests exactly that distinction by moving the rules, unchanged, to a
different dataset with a different scanner.

This imposes a discipline we hold throughout: **no constant in this work was
chosen by maximising a reported score.** Every adopted change was selected on
development scenes and reported on held-out scenes with a paired bootstrap.

### 1.2 Contributions

1. **A geometric motion module** (§4.1) that estimates type, axis and origin from
   a single static scan with no learned component. Removing its central insight
   costs **26.6 gate points** and drives revolute direction error from 8.45° to
   74.09° (§6.1).
2. **Cross-dataset evidence that the rules are properties of furniture, not of a
   dataset** (§5). Transferred unchanged to Articulate3D and scored by USDNet's
   released evaluator, they reach a composed gate of **83.0%** against that
   supervised method's 59.8%.
3. **Segmenter-independence measured parametrically rather than at three
   arbitrary points** (§6), including a mechanistic control: contaminating an
   element with its surroundings *improves* the element-normal baseline by 12.9
   points, because it accidentally supplies the mounting surface.
4. **A signed-direction rule that costs nothing** (§4.2). Plane fits give an axis,
   not a vector, so the deployed direction is sign-correct 40.7% of the time — a
   coin flip. Orienting the parent normal outward raises this to **76.9%** against
   an unsigned ceiling of 84.3%, with no new model and no new constant.

▢ **Figure 1 (teaser).** Instruction → predicted 3D region → predicted axis and
origin against ground truth. *Blocked: requires our predicted masks, which are on
the Colab run and not local. Figures 2–5 do not depend on it.*

---

## 2. Related work

**Functional 3D scene understanding.** SceneFun3D [1] introduced the benchmark
with three tasks: functionality segmentation, task-driven affordance grounding,
and motion estimation. Its supervised baseline **Mask3D-FM** reaches AP25 26.6 on
the held-out test split, trained on a 545-scene regime never publicly released
(the public release is 315 scenes), with an evaluation implementation that has
also not been released. Its numbers therefore cannot share a table column with
methods evaluated on public data, and we report them only as a cited reference.

**Training-free 3D grounding.** Open-vocabulary 3D methods transfer poorly to
this task. OpenMask3D [3], OpenIns3D [4] and LERF [5] reach AP25 of 0.4, 0.0 and
0.0 respectively [2]: they localise broadly (AR25 27–52) but cannot delineate
centimetre-scale parts. **Fun3DU** [2] is the current training-free leader
(AP25 33.3), using OWLv2, SAM and the 7B-parameter Molmo VLM on an A100.

**Articulation estimation.** Prior articulation work (OPD, MultiScan,
PartNet-Mobility) largely operates on isolated objects or single views, where the
notion of a *mounting surface* is unavailable. Our method depends on scene
context — the surrounding wall or cabinet face — which scene-level scans provide
and object-level datasets do not. **USDNet** [8] is the closest comparison: a
Mask3D-based supervised model on Articulate3D that predicts motion type, axis and
origin, and releases both its checkpoint and its evaluator. §5 uses that
evaluator unmodified.

**Positioning.** The gap between our segmentation and Fun3DU's is *not* a
supervision gap — both are training-free. It is a difference in how language is
grounded to pixels, and it carries a compute cost we quantify in §8. Our
contribution is orthogonal: no prior training-free method estimates motion
parameters at all.

---

## 3. Problem formulation

Given a laser-scanned point cloud $P \in \mathbb{R}^{N\times3}$ and a natural
language instruction $q$, predict a **region** $R \subset \{1..N\}$, a **motion
type** $\tau \in \{\text{rot}, \text{trans}\}$, an **axis direction**
$\hat{d} \in S^2$, and an **origin** $o \in \mathbb{R}^3$ (undefined for
translation, which is origin-invariant).

**The motion gate.** Following the MultiScan protocol used by both SceneFun3D and
Articulate3D, a prediction passes when the orientation error OE < 15°, and
additionally, for revolute elements, when the minimum axis distance MD < 0.25 m.
Both benchmarks use these thresholds; they differ only in the base IoU required
for a detection (0.25 vs 0.50).

---

## 4. Method

The motion module is the contribution and is presented first. The segmentation
front end that feeds it (§4.3) is a means to an end, and §6 shows the module does
not depend on it.

### 4.1 Motion from the mounting surface

Given a region's points and a neighbourhood around them, the module returns type,
axis and origin. Every step below is a fixed geometric rule.

**Parent surface.** Fit a plane to a shell of neighbourhood points at 1.5–4.0×
the region's own extent, *excluding the region's own points*. The fit is
MAD-trimmed (k = 2.5, scale 1.4826) because the shell contains off-panel points
that least squares cannot down-weight, and the shell grows by ×1.5 (capped at
12×) when it holds fewer than 50 points, because a shell too thin to fit is
ill-posed. This yields the **parent normal** $n$: 3.21° median error on prismatic
elements against 11.50° for the region's own normal.

**Type.** Panel height or offset disambiguates the two affordance classes whose
motion type is mixed; the remaining seven classes are determined by class.

**Axis.** Levers and mixed-class elements predicted revolute take **gravity** —
SceneFun3D and Articulate3D hinge axes are canonically vertical, a convention of
how doors are built as much as a geometric fact. Everything else takes the parent
normal: a drawer slides out of the face it is set into.

**Origin** (revolute only). Knobs take the region centroid, which is essentially
solved at 4 mm median. Hinges take a **panel edge**. The panel is isolated by
coplanar region growing from the point nearest the element; the two candidate
edges lie along $\hat{a} = \hat{d} \times n$. Which edge is chosen follows an
**ergonomic prior**: a handle is mounted for leverage and is therefore opposite
the hinge, so when the element sits decisively toward one edge (asymmetry ≥ 0.6)
we take the *farther* one. Otherwise a corner score — counting non-coplanar
neighbours, since a hinge is structurally anchored — decides. A cascading
fallback (connected panel → raw coplanar extent → centroid) gives ~98% coverage.

**Figure 2** walks these steps through one real element.

![Figure 2](figures/fig3_insight.pdf)

### 4.2 Signing the direction

A plane fit determines an axis, not a vector, so the deployed prediction points
the wrong way roughly half the time. The composed metric is unsigned and this
costs nothing there, but ground-truth annotations *are* signed and meaningful —
the 57 `key_press` elements share $(0,0,-1)$, encoding "press downward" — so it
is real information discarded.

The fix uses no new quantity. An element protrudes from the surface it is mounted
on, so $(\bar{p} - \bar{s}) \cdot n > 0$ identifies the outward side, where
$\bar{p}$ and $\bar{s}$ are the element and shell centroids. Orient $n$ outward,
then let the affordance supply the sense: pull families go outward, press and
plug families go inward. Revolute elements are excluded — a hinge axis genuinely
has no preferred end.

| direction, n = 432 prismatic elements | accuracy |
|---|---|
| unsigned (axis within 15°) — the ceiling | 84.3% |
| signed, as deployed | 40.7% |
| **signed, outward-oriented** | **76.9%** |

The rule recovers 91% of the achievable signed accuracy. Per family: pull 83.3%,
plug 77.5%, press 64.0%.

### 4.3 Segmentation front end

Four stages, all frozen: CLIP ViT-L/14 retrieves visually similar exemplars for
the instruction; GroundingDINO-tiny proposes boxes under a class-conditioned
hint; DINOv3 ViT-L/16 dense correspondence against the retrieved exemplars picks
a point inside the best box; SAM ViT-B turns that point into a mask, which is
lifted to the raw scan by depth-checked projection and multi-view voting.

---

## 5. Cross-dataset transfer: Articulate3D

If the rules describe furniture rather than a dataset, they should work on a
dataset they have never seen. **Articulate3D** (280 ScanNet++ scans, splits
195/42/42) uses a different scanner, different object scale, and the same motion
thresholds as SceneFun3D. We change nothing.

One implementation detail did not transfer, and the diagnosis is a fact about the
scenes rather than about the method: **a door set flush in a wall is coplanar
with that wall.** No plane filter separates them, so recovering the panel by
region growing returned the wall — median span 1.44 m for a ~0.8 m door — and
capped even oracle edge choice at 46.3%. Articulate3D segments the *movable part*
directly, so the panel is given rather than recovered, and the same rules then
apply unchanged (`predict_motion_given_panel`).

| Articulate3D validation, USDNet's evaluator | n | axis ≤15° | origin <0.25 m | composed |
|---|---|---|---|---|
| **Ours** — all parts | 389 | **93.8%** [87.9, 97.7] | 77.4% [70.8, 84.0] | **83.0%** [75.8, 88.2] |
| USDNet, as published ‡ | — | 82.8% | 75.1% | 59.8% |
| Ours — panel recovered, not given | 183 | 88.9% | 35.2% | 45.9% |

**Scored with Articulate3D's criteria, not ours.** USDNet releases
`benchmark/evaluate_semantic_instance.py`. Both implementations test the axis
identically (unsigned, 15°), but their origin test is two-sided: the predicted
origin must lie within 0.25 m of the ground-truth axis *line* **and** the
ground-truth origin within 0.25 m of the predicted axis line. We had used the
minimum distance between the two lines, which is no greater than either.
Re-scoring under their rule moved origin from 82.6% to 77.4%. We report theirs.

‡ **Not yet like-for-like, and the intervals say why.** Our figures are on
ground-truth part masks; USDNet's are retention derived from its published AP50 of
41.8 on its *own* predicted masks. Under that asymmetry, axis and the composed
gate separate clearly — USDNet's 82.8% and 59.8% both fall outside our
intervals — but **origin does not: their 75.1% lies inside our [70.8, 84.0]**, so
the 2.3-point origin lead is not distinguishable from noise and we do not claim
it. §7.1 is the hybrid run that removes the asymmetry.

---

## 6. The motion module does not depend on its segmenter

Table 2 in the conventional design samples this claim at a handful of arbitrary
points — Fun3DU, OpenMask3D — each costing a GPU, a large download and somebody
else's dependency tree, and still cannot separate "our module is robust" from
"those segmenters happen to fail similarly."

The claim we actually make is mechanistic and therefore falsifiable: axis and
origin are read off the **parent surface**, not off the element. If that is true,
corrupting the element — which is exactly what a worse segmenter does — should
barely move the result, while the same corruption should wreck a method that reads
geometry off the element itself. Ablation A ("use the element's own normal") is
that control group.

![Figure 3](figures/fig7_robustness.pdf)

| corruption (n = 734, 3 reps) | ours | control |
|---|---|---|
| dilate region to 2× element extent | 60.9 (−1.6) | 36.5 (−0.5) |
| discard 50% of region points | 62.8 (**+0.3**) | 37.4 (+0.3) |
| 10% of region drawn from surroundings | 61.6 (−1.0) | 39.3 (+2.2) |
| centroid drift of 2 cm | 58.5 (−4.1) | 37.1 (+0.1) |

Two readings. First, within the regime a real segmenter occupies — an oversized
or partial mask — the module is flat; only gross drift, larger than the median
element itself, moves it appreciably. Second, and more diagnostic: **contaminating
the element with its surroundings improves the control by 12.9 points** (37.1 →
49.9 at 50%), because contamination accidentally hands it the very parent surface
it was denied. That is direct mechanistic support for the claim, from the control
group rather than from us.

**Discrete confirmation** ▢. Fun3DU is the strongest published SceneFun3D method
and the most different from ours, and is the natural second point. Our first run
is diagnosed but not yet usable: Fun3DU recalled the element in 40.8% of cases —
consistent with its published AP25 of 33.3 — but our readout of its score field
returned regions **14.2× the ground-truth element**, collapsing precision to
0.003. The fault is our threshold, not their model: `np_normalize(acc_f/n_views)
> 0.7` keeps the top 30% of the score *range*, not of the points.
`scripts/fun3du_threshold_sweep.py` re-derives the operating point in Fun3DU's
favour and is written and waiting on the stage-4 output.

---

## 7. SceneFun3D

### 7.1 Motion parameters, ground-truth regions

n = 734 elements, 48 scenes.

| | prismatic | hinge | knob | all revolute |
|---|---|---|---|---|
| direction error (median OE) | **2.58°** | **0.00°** | 17.18° | **8.45°** |
| origin error (median MD) | n/a | 0.133 m | **0.004 m** | — |
| motion gate | 80.9% [74.3, 86.5] | — | — | 36.0% [29.9, 41.7] |

Overall motion gate **63.6%** [57.3, 69.7]; type accuracy **89.9%** [85.9, 93.5].

Hinge direction is *exact*: the gravity prior is correct by construction for
vertically-hung elements and the annotations agree. Knob origin is effectively
solved at 4 mm. The remaining error concentrates in hinge **origin**, where the
correct panel edge must be identified.

![Figure 4](figures/fig5_qualitative.pdf)

### 7.2 The composed motion metric

SceneFun3D's own definition of the motion task, computed end-to-end on regions
our pipeline predicts. To our knowledge these are the first such numbers reported
by anyone other than the dataset authors.

| method | training | AP25 | +M | +MA | **+MAO** |
|---|---|---|---|---|---|
| Mask3D-FM (rgb) ‡ | supervised, hidden test | 26.6 | 23.8 | 9.8 | **7.9** |
| **Ours** | **training-free** | 9.85 | 9.20 | 6.60 | **6.07** |

**Retention** — the fraction of detections surviving each gate — is where the
method's character shows:

| | AP25→+M | →+MA | →+MAO |
|---|---|---|---|
| Mask3D-FM ‡ | 89.5% | 36.8% | **29.7%** |
| **Ours** | **93.4%** | **67.0%** | **61.6%** |

We begin with 2.7× worse detection and finish within 23% of a fully supervised
method on its own motion metric. Both classify type about equally well; the
separation appears at the axis gate (67.0% vs 36.8%) and holds through origin.

‡ **A cited reference, not a head-to-head.** Mask3D-FM's figures are its authors',
on the 85-scene hidden test set, with an unreleased evaluation implementation.
Two of the three differences favour the baseline: it trained on 545 scenes
against the 315 public, and our convention of counting a gate failure as a false
positive can only lower our numbers. The third (hidden test split vs ours) is of
unknown direction.

With ground-truth regions the chain reaches **+MAO = 49.5**, bounding what
perfect detection would deliver and locating the bottleneck in front-end
discovery rather than in motion.

### 7.3 Detection, for context

Segmentation is the front end that feeds motion, not our contribution. We report
it because it bounds the composed metric.

| method | training | VLM | AP25 | AP50 | mIoU |
|---|---|---|---|---|---|
| OpenMask3D | free | no | 0.4 | 0.2 | 0.2 |
| OpenIns3D | free | no | 0.0 | 0.0 | 0.1 |
| LERF | per-scene opt. | no | 0.0 | 0.0 | 0.0 |
| Fun3DU | free | 7B | **33.3** | **16.9** | **15.2** |
| Fun3DU, −VLM ablation | free | no | 13.9 | — | 6.3 |
| **Ours** † | **free** | **no** | ~12 | ~2.6 | ~9.5 |

† Our row is measured on our 48-scene set, not split0; the split0 run is
outstanding (§9). We exceed the three open-vocabulary methods on AP25 by 30–100×.
Against Fun3DU's own **−VLM ablation** — the row matching our compute class — we
are comparable on AP25 and ahead on mIoU, while additionally producing motion
parameters.

### 7.4 Where it fails

![Figure 5](figures/fig6_failures.pdf)

267 of 734 elements fail the gate, and the two halves have different causes.

| mode | share of failures | share of all |
|---|---|---|
| axis wrong (OE ≥ 15°) | 81.6% (n=218) | 29.7% |
| axis right, origin wrong | 18.4% (n=49) | 6.7% |

The origin half is the honest weak point of the ergonomic edge rule: it assumes
the handle is opposite the hinge, which fails on double doors and where region
growing crosses into an adjoining surface.

The axis half has a sharper cause than "geometry is hard". **33.5% of axis
failures also have the type wrong**, and 55 of those are revolute elements called
prismatic — which are then handed the parent *normal* as a direction, exactly
perpendicular to the hinge they turn about. Errors therefore cluster at 90°
rather than spreading. **Type classification, not axis geometry, is the lever on
the larger half of the failures.**

---

## 8. Compute

| | ours | Fun3DU [2] |
|---|---|---|
| largest model | DINOv3 ViT-L, **0.3B** | Molmo, **7B** |
| full stack | DINOv3 + GD-tiny + SAM-B + CLIP ≈ **0.9B** | OWLv2 + SAM + Molmo |
| GPU | **T4, 16 GB** | A100 |
| per element (full 3D lift, 6 frames) | **22.8 s** | ~140 s † |

† Measured on our own A100 run of the released code: stage 2 detection 61 s,
stage 3 Molmo 60 s, stage 4 lifting 18 s per task description. Fun3DU does not
report a runtime.

This is not efficiency for its own sake. **A mobile manipulator can carry a
T4-class device; it cannot carry an A100 running a 7B VLM** — an A100's 250–400 W
envelope alone exceeds the LoCoBot battery's 130 W maximum output. The compute
profile is what makes the method deployable on the robot this work targets, and
it is why we accept a weaker localiser rather than adopt a VLM front end.

---

## 9. Ablations

Each row removes exactly one design decision; nothing is re-tuned afterwards.
n = 734, 48 scenes.

| ablation | motion gate | Δ |
|---|---|---|
| **Full method** | **63.6%** | — |
| − parent surface normal (use element's own) | 37.1% | **−26.6** |
| − edge choice → region centroid | 56.3% | **−7.4** |
| − gravity prior for hinges | 58.4% | −5.2 |
| − robust fit **and** adaptive shell | 59.3% | −4.4 |
| − cascading origin fallback | 60.1% | −3.5 |
| − adaptive shell only | 60.6% | −3.0 |
| − corner score → always-farther edge | 61.0% | −2.6 |
| − robust fit only (plain SVD) | 61.2% | −2.5 |
| − type rule → majority baseline | 63.1% | −0.5 |

**The central insight is confirmed by its removal.** Substituting the element's
own normal for the parent-surface normal costs 26.6 gate points and drives
revolute direction error from 8.45° to **74.09°**. No other component is within a
factor of 3.5 of this.

---

## 10. Limitations and outstanding work

- **The Articulate3D comparison is not yet like-for-like.** Our numbers are on
  ground-truth part masks. The hybrid run — USDNet's predicted masks and classes,
  our axis and origin, all three rows through one metric implementation — is the
  experiment that settles it, and needs one GPU job behind a CUDA 12.1 Docker
  build. A dependency to resolve: the ergonomic rule needs the interactable
  element, and USDNet's *interactable* checkpoint is not released, so the hybrid
  must be reported with and without handles as separate rows.
- **Our detection row is not on split0.** §7.3's table is headed split0 and our
  row is a † estimate on the 48-scene set. One ~2 h GPU job fixes it.
- **Fun3DU as a second segmenter** is diagnosed, not delivered (§6).
- **Type classification is the dominant failure lever** (§7.4) and is currently a
  fixed rule over affordance class.
- **The composed metric is our reimplementation.** SceneFun3D's Task 3 evaluation
  is unreleased. Our ground-truth-region chain returns AP25 = 100 by
  construction, which we use as a wiring check.
- ▢ **Real-robot validation.** A LoCoBot test is the claim's true discharge. The
  gravity prior is *better* grounded on a robot, which gets it from the base
  frame; the risk is the ≥1.1 m neighbourhood radius the parent-surface fit
  requires, which a single RGB-D view may not cover.

---

## 11. Conclusion

Motion and detection do not fail together. A robot needs both, but they are not
equally hard and they are not bottlenecked by the same thing.

Motion parameters for functional elements can be estimated from a single static
scan using fixed geometry alone — no training, no per-dataset fitting. Moved
unchanged to a dataset with a different scanner and object scale and judged by
that dataset's own evaluator, the same rules reach a composed gate of 83.0%
against a supervised model's 59.8%. A parametric corruption sweep shows why the
module survives a bad segmenter: it never read the element in the first place.

The practical implication is a division of labour. The expensive, brittle part of
functional scene understanding is *finding* the element. The part that tells a
gripper what to do is cheap, transferable, and can be attached to whatever
segmenter is available.

---

## References

[1] A. Delitzas, A. Takmaz, F. Tombari, R. Sumner, M. Pollefeys, F. Engelmann.
*SceneFun3D: Fine-Grained Functionality and Affordance Understanding in 3D
Scenes.* CVPR 2024.

[2] J. Corsetti, F. Giuliari, A. Fasoli, D. Boscaini, F. Poiesi. *Functionality
Understanding and Segmentation in 3D Scenes.* CVPR 2025.

[3] A. Takmaz et al. *OpenMask3D: Open-Vocabulary 3D Instance Segmentation.*
NeurIPS 2023.

[4] Z. Huang et al. *OpenIns3D: Snap and Lookup for 3D Open-vocabulary Instance
Segmentation.* ECCV 2024.

[5] J. Kerr, C. M. Kim, K. Goldberg, A. Kanazawa, M. Tancik. *LERF: Language
Embedded Radiance Fields.* ICCV 2023.

[6] M. Oquab et al. *DINOv2 / DINOv3: Learning Robust Visual Features without
Supervision.*

[7] S. Liu et al. *Grounding DINO: Marrying DINO with Grounded Pre-Training for
Open-Set Object Detection.* ECCV 2024.

[8] *USDNet: Articulation understanding on Articulate3D.* (checkpoint and
evaluator released; used unmodified in §5.)
