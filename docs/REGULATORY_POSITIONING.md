# SkinAnalytica — regulatory positioning

Last updated: 2026-07-27.

**This is not legal or regulatory advice.** It's an internal positioning
document written by the engineering side of this project, grounded in the
evidence already recorded in [MODEL_CARD.md](MODEL_CARD.md) and
[KNOWN_GAPS.md](KNOWN_GAPS.md). Its job is to state plainly what
SkinAnalytica currently is (and isn't) allowed to claim, and what a real
path to a regulatory claim would require, so nobody downstream — a pitch
deck, a partner conversation, a future hire — accidentally overstates the
product's status. Before any actual regulatory filing, commercial launch,
or clinical claim, engage a qualified regulatory affairs / quality
assurance consultant. Nothing here substitutes for that.

## 1. What SkinAnalytica is today

A research prototype: a 7-class dermoscopy image classifier (ensemble of
EfficientNetV2-S, ViT-L/16, ConvNeXt-Large) that outputs a per-class
probability distribution and a rule-based verdict (`NORMAL` /
`REVIEW_REQUIRED` / `CANCER_FLAGGED`) derived from a melanoma-probability
threshold. It is not registered, cleared, approved, or CE/UKCA-marked as a
medical device in any jurisdiction. It holds no predicate-device
comparison, no clinical study, and no quality management system (QMS).

This status is already stated to end users — [index.html](../index.html)'s
disclaimer and [docs.html](../docs.html)'s "Known limitations" section both
say "research prototype — not a clinical diagnostic device — no regulatory
clearance," and that framing should not change until the gaps in section 3
are actually closed, not just documented.

### Current intended-use boundary (what can honestly be claimed now)

> Research and educational use only. Not intended to diagnose, treat,
> cure, or prevent any disease. Not a substitute for professional medical
> judgment. Any output must be confirmed by a qualified clinician before
> it informs a clinical decision.

This is deliberately narrow — narrower than the model's actual
discriminative performance on in-distribution data would suggest is
possible — because the gaps in section 3 mean the *conditions* under
which that stronger performance holds (data source, capture modality,
demographic makeup) aren't yet controlled for or disclosed to a user in a
way a regulator or clinician could rely on.

## 2. If this were ever submitted for clearance: likely framework and classification

This section is a directional read of publicly known frameworks, not a
determination — actual classification requires a qualified RA consultant
and, in the US, ideally an FDA pre-submission (Q-Sub) meeting.

**Why this is Software as a Medical Device (SaMD), not exempt wellness
software**: SkinAnalytica's melanoma-probability output and
`CANCER_FLAGGED` verdict are intended to inform a clinical decision about
a serious, life-threatening condition. Under the IMDRF SaMD risk framework
(the reference point both FDA and EU MDR software guidance use), that
combination — "informs clinical management" × "serious/critical
condition" — sits at Category II/III, not the lowest-risk category that
covers pure lifestyle/wellness apps.

| Jurisdiction | Likely framework | Likely classification | Likely pathway |
|---|---|---|---|
| **US** | FDA, Software as a Medical Device / CADe-CADx-style AI diagnostic aid | Class II (moderate risk) | 510(k) if a suitable predicate exists (several AI-based dermatology triage/CADx tools have reached market in recent years — verify current predicates before relying on this), otherwise De Novo |
| **EU** | MDR 2017/745, Rule 11 (software) | Class IIa or IIb, depending on whether the output is used to "take decisions with diagnosis or therapeutic purposes" and how directly | Conformity assessment via a Notified Body (self-certification is not available at this classification) |
| **UK** | UKCA, post-Brexit MHRA framework (currently aligned with EU MDR software rules) | Comparable to the EU MDR line above | MHRA-recognized approved body assessment |

Common cross-jurisdiction requirements regardless of which pathway: a
quality management system (ISO 13485), a documented risk management file
(ISO 14971), a software development lifecycle record (IEC 62304), and a
usability engineering file (IEC 62366-1). None of these exist for this
project today — see section 4.

**On predicate devices**: several AI-assisted dermatology tools (both
image-classification and non-imaging spectroscopy-based) have reached
regulatory clearance in different jurisdictions in recent years, which is
relevant context for pathway selection (510(k) vs. De Novo). This document
deliberately does not name or characterize specific competitor products or
their current clearance status — that landscape changes and any claim
about a specific device's status should be verified fresh, from the
regulator's own database, at the time it actually matters (e.g. during
predicate selection), not carried forward from this document.

## 3. Gap map — what's already documented in the model card, and what it blocks

Every row below cites a real, already-recorded finding — nothing here is a
new claim. The point of this section is to translate "here's an
engineering/data finding" into "here's the regulatory requirement it fails
to satisfy today."

| Known gap (see MODEL_CARD.md) | What it blocks |
|---|---|
| Metrics come from a **single train/val split, single seed** (Known limitations) | Clinical/analytical validation for a submission needs performance estimates that account for training-process variance, not just sampling noise from one split — typically k-fold or repeated-seed validation, or a prospective study |
| **External-population threshold transfer failure**: 42.7% sensitivity on the 143-case SLICE-3D external holdout vs. the 80% target, without specificity collapsing (findings #11, #12) | This is the single largest blocker to any performance claim. A submission's clinical validation must reflect real-world, out-of-distribution performance — the current deployed threshold does not meet its own stated sensitivity target on the best available external test set. This is Path B in [RETRAIN_PLAN.md](RETRAIN_PLAN.md), not yet started |
| **Subgroup sensitivity spreads (age, skin tone) not yet confirmed as real vs. small-sample noise** (findings #14 addendum, 13-34 melanoma cases per subgroup) | FDA's AI/ML good-machine-learning-practice guidance and EU MDR's General Safety and Performance Requirements (Annex I) both expect demographic-subgroup performance to be characterized with adequate sample sizes, not flagged as "too few cases to say." This needs materially more labeled data per subgroup before it could support a claim either way |
| **Zero validated performance on non-dermoscopic (phone-camera/clinical) images** (finding #9) — and the guardrail that exists is explicitly "narrow, low-confidence, advisory only, never a hard block" (finding #15) | A regulated device needs its intended-use input to be either reliably enforced or the device's labeling/claims scoped to only what's actually validated. An advisory-only heuristic doesn't satisfy either — it's a good-faith UX signal, not an input-validation control a regulator would accept |
| **No k-fold/repeated-seed retraining, no full retraining to fix the age-based sensitivity gap** (KNOWN_GAPS.md, blocked on sustained compute) | Same as the split-variance point above — also the root cause of the age gap is mitigated (threshold-level), not fixed (training-level), which matters for a submission that has to describe the actual mechanism, not just the current workaround |
| **Two training datasets (MILK10k, MRA-MIDAS) are non-commercially licensed** (KNOWN_GAPS.md, "Licensing decision") | This is a hard blocker independent of clinical evidence: a commercial product cannot ship a model trained on CC-BY-NC or non-commercial-DUA data. Any commercialization path requires either retraining without those sources (losing their skin-tone-fairness contribution — finding #14) or securing a commercial license, before any regulatory work is worth starting |
| **No quality management system, design history file, risk management file, or software lifecycle documentation exists anywhere in this repo** | This is the baseline infrastructure gap underneath everything else — ISO 13485 (QMS), ISO 14971 (risk management), IEC 62304 (software lifecycle), IEC 62366-1 (usability) are all prerequisites to *starting* a submission, separate from whether the model itself performs well enough |
| **No post-market surveillance or adverse-event-reporting mechanism** (the "no scheduler existed anywhere" finding, #8, is now fixed for internal drift monitoring — but that's not the same thing as a regulator-facing PMS/vigilance process) | Both FDA and EU MDR require an active post-market surveillance plan, and MDR specifically requires incident reporting (vigilance) infrastructure — the daily internal drift monitor added this session (`agents/scheduled_monitor.py`) is a good engineering foundation for this but isn't itself a compliant PMS system |
| **No predetermined change control plan (PCCP) or equivalent for model updates** | Every threshold/model change this session (age bands, TBP threshold, skin-tone thresholds) was made and deployed ad hoc. A regulated AI/ML device needs a pre-specified, reviewable process for what kinds of changes can ship without a new submission — FDA's PCCP framework is the current mechanism for this in the US |
| **No clinical validation study of any kind** — every number in the model card comes from retrospective scoring against existing datasets | All of the above assumes retrospective evidence can eventually close these gaps; at some point a prospective or at minimum a properly-designed retrospective *clinical* validation study (not just dataset scoring) becomes necessary, particularly for the pivotal evidence in a submission |

## 4. What's explicitly *not* a gap, for balance

Two things worth stating plainly, because an honest document should say
what's already solid, not just what's missing:

- **The engineering practice around this model is already close to what a
  QMS would formalize.** Version-pinned thresholds with regression tests
  (`tests/test_config_consistency.py`), a documented, chronologically
  numbered record of every finding and fix (this model card), and
  regression tests for three previously-manual-only deployment bugs
  (`tests/test_deployment_config.py`, `tests/test_training_utils.py`) are
  all in the spirit of design controls and traceability — they just
  aren't yet organized into the formal artifacts (design history file,
  risk file) a QMS requires.
- **The discrimination finding (AUC 0.6131, p=5.2×10⁻⁶ on n=143 external
  malignant cases, finding #12) is a real, statistically significant
  result**, not a marginal one. The blocker to a performance claim is the
  *threshold-transfer* problem on top of that real discrimination, not an
  absence of underlying signal. That distinction matters for prioritizing
  Path B over starting from scratch.

## 5. If commercialization is pursued: a directional, non-binding phase outline

This is a rough sequencing sketch, not a project plan with dates or
owners — those decisions are business calls this document doesn't make.

1. **Phase 0 — licensing cleanup.** Resolve the MILK10k/MRA-MIDAS
   non-commercial licensing conflict (retrain without them, or negotiate
   a commercial license) before spending further effort on anything else
   in this list — it's a binary blocker, not a matter of degree.
2. **Phase 1 — close Path B and expand external validation.** Bring the
   threshold-transfer problem down from "42.7% sensitivity on external
   data" toward the 80% target without the specificity collapse seen at
   the current fallback threshold (0.0199 → 15.6% specificity) — via more
   real TBP/external training data, metadata fusion, or
   segmentation-assisted classification (see RETRAIN_PLAN.md's Path B
   section). Grow subgroup (age, skin-tone) sample sizes enough to
   confirm or rule out the sensitivity spreads currently flagged as
   noise.
3. **Phase 2 — build the QMS.** ISO 13485 quality system, ISO 14971 risk
   management file, IEC 62304 software lifecycle documentation, IEC
   62366-1 usability engineering file. This can start in parallel with
   Phase 1 — it doesn't depend on the model improving further.
4. **Phase 3 — pathway determination.** Predicate search (US) or
   Notified Body pre-consultation (EU/UK); for the US, consider an FDA
   Q-Sub meeting before committing to 510(k) vs. De Novo.
5. **Phase 4 — pivotal clinical validation.** A properly designed
   (ideally prospective, multi-site, demographically stratified and
   powered for the subgroup questions Phase 1 couldn't fully answer
   retrospectively) clinical study — the evidence base a submission
   actually rests on, distinct from the retrospective dataset scoring
   this project has relied on so far.
6. **Phase 5 — submission and conformity assessment.**
7. **Phase 6 — post-market.** Build the vigilance/PMS process and a
   predetermined change control plan before the first post-clearance
   model update ships, not after.

## 6. Summary

SkinAnalytica today is honestly positioned as a research prototype with
one real, statistically significant strength (external discrimination,
finding #12) sitting on top of several concrete, already-documented gaps
(threshold transfer, subgroup sample size, non-dermoscopic input,
licensing, and a complete absence of QMS/regulatory infrastructure). None
of those gaps are secret or newly discovered by this document — they're
already in [MODEL_CARD.md](MODEL_CARD.md) and
[KNOWN_GAPS.md](KNOWN_GAPS.md). What this document adds is the mapping
from "engineering finding" to "regulatory requirement it fails to meet,"
so that mapping doesn't have to be reconstructed from scratch the next
time someone — inside or outside this project — asks what it would take
to make a real clinical claim.
