"""
AestheticAnalyzer — "chopped score" aggregator.

What it does
------------
Reads the merged result dict from every other analyzer and produces a
single numeric chopped_score (0-100) plus a per-factor breakdown.
Higher = more chopped = less conventionally attractive (by the
arbitrary rubric encoded here). The breakdown lets you tune weights
or flip polarity client-side without rerunning inference.

Score composition
-----------------
Final chopped_score is a weighted blend of two sources:

1. **Learned beauty regressor** (BeautyAnalyzer, SCUT-FBP5500):
   raw score in [1.0, 5.0] mapped to a "stretched" 0-100 axis. The
   raw model output is fairly concentrated around 2.5-3.5 (most
   faces), which would cluster scores near the middle. We stretch
   the [2.0, 4.0] sub-range to fill [0, 100] so the tier system
   gets meaningful spread.

2. **Rule-based factor sum**: scaled penalties + bonuses on top of
   a baseline of 50. Factors are documented in `_compute_rule_score`.

Blend math
----------
    learned_unattractive = stretched_unattractive(beauty_norm)
    chopped = 0.6 * learned_unattractive + 0.4 * rule_score
    chopped is clamped to [0, 100].

Tuning history
--------------
- Original rule factors landed almost everyone at 50 ± 10. Scaled up
  by ~1.6× to give attributes more bite. The learned signal got a
  matching stretch (75 → 100, 25 → 0) so it isn't drowned out.
- Freckles/moles penalty was removed — the underlying SegFormer-based
  detector was unreliable and the metric was effectively penalising
  shadows and pores.

Subjectivity disclaimer
-----------------------
Every weight in this file is a guess. "Beauty" is subjective,
culturally biased, and reductive. Treat the score as an in-joke
metric; never expose it as objective truth. The UI gates the row
behind a Settings toggle off-by-default for that reason.

Note: this analyzer takes no image input — it reads the merged result
dict produced by every other analyzer that ran ahead of it.
"""

from typing import Any


# How much weight the learned beauty regressor gets when both signals
# are available. 0.85 means the SCUT-FBP5500 ResNet-50 strongly
# dominates the chopped score — rule factors contribute 15% as a
# refinement layer rather than a primary driver. The trained model
# learned from 60-rater-averaged human ground truth, which is a much
# better signal than any hand-tuned heuristic.
LEARNED_WEIGHT = 0.85

# Baseline score. Penalties push up, bonuses pull down.
BASELINE = 50.0

# Stretch the learned-beauty 0-100 axis so it covers the chopped
# spectrum more dramatically. Beauty norms in (LEARNED_NORM_LO, _HI)
# map linearly to (0, 100). Below the lo bound is "fully chopped"
# territory (learned_unattractive = 100); above the hi bound is
# "fully gigachad" (learned_unattractive = 0).
LEARNED_NORM_LO = 25.0   # raw score ≈ 2.0
LEARNED_NORM_HI = 75.0   # raw score ≈ 4.0


def _stretch_unattractive(beauty_norm: float) -> float:
    """Map BeautyAnalyzer's 0-100 normalised score to a stretched
    unattractiveness 0-100. 75 → 0 (gigachad), 25 → 100 (megachopped).
    """
    if beauty_norm is None:
        return 50.0
    # Invert the axis then linearly stretch (LEARNED_NORM_LO, _HI).
    unattractive = 100.0 - float(beauty_norm)
    # unattractive: 25 (gigachad-ish) -> 100 (megachopped-ish)
    span = (100.0 - LEARNED_NORM_LO) - (100.0 - LEARNED_NORM_HI)  # = 50
    lo_after_invert = 100.0 - LEARNED_NORM_HI                     # = 25
    stretched = (unattractive - lo_after_invert) / span * 100.0
    return max(0.0, min(100.0, stretched))


class AestheticAnalyzer:
    def __init__(self):
        # No model to load.
        pass

    def analyze(self, merged: dict[str, Any]) -> dict[str, Any]:
        """Compute chopped score from the merged result dict.

        Unusual signature: not (img_rgb), since this analyzer aggregates
        prior results rather than running inference on the image.
        app.py special-cases the call to pass `merged` here.
        """
        rule_score, breakdown = self._compute_rule_score(merged)

        beauty_norm = merged.get("beauty_score_norm")
        if beauty_norm is not None:
            learned_unattractive = _stretch_unattractive(float(beauty_norm))
            chopped = (
                LEARNED_WEIGHT * learned_unattractive
                + (1.0 - LEARNED_WEIGHT) * rule_score
            )
            # Show the learned contribution as a signed offset from
            # baseline so the breakdown reads consistently with rule
            # factors.
            breakdown["learned_unattractive"] = round(
                LEARNED_WEIGHT * (learned_unattractive - BASELINE), 2
            )
            breakdown["_blend_weight_learned"] = LEARNED_WEIGHT
        else:
            chopped = rule_score
            breakdown["_blend_weight_learned"] = 0.0

        chopped = max(0.0, min(100.0, chopped))

        return {
            "chopped_score": round(chopped, 1),
            "chopped_breakdown": breakdown,
            "chopped_polarity_note": (
                "0 = least chopped, 100 = most chopped. "
                "Subtract from 100 for an 'attractiveness' read."
            ),
        }

    # ------------------------------------------------------------------
    # Rule-based scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rule_score(d: dict[str, Any]) -> tuple[float, dict[str, float]]:
        """Hand-tuned weighted sum over previously-extracted attributes.

        Returns (score, breakdown_dict). The breakdown gives each
        factor's signed contribution so a UI can show *why* a score
        landed where it did. Score starts at BASELINE (50) and moves
        up/down.

        All penalty / bonus magnitudes are scaled up vs. the original
        implementation so attributes contribute meaningfully to the
        spread instead of nudging everyone toward 50.
        """
        score = BASELINE
        breakdown: dict[str, float] = {}

        # ── Penalties (push score up = more chopped) ─────────────────

        # Facial asymmetry: 0 = perfectly symmetric, 1 = very asymmetric.
        # MediaPipe's score is noisier than we'd like — attractive faces
        # still come back with measurable asymmetry from natural
        # micro-expressions and camera angle. De-emphasised from ×30.
        asym = d.get("facial_asymmetry_score")
        if isinstance(asym, (int, float)):
            penalty = float(asym) * 18.0
            score += penalty
            breakdown["asymmetry_penalty"] = round(penalty, 2)

        # Wrinkle level from SegFormer + OpenCV Laplacian classification.
        wrinkle_penalty_map = {
            "smooth": 0.0, "slight": 6.0, "moderate": 14.0, "prominent": 20.0,
        }
        wrinkle = d.get("wrinkle_level")
        if wrinkle in wrinkle_penalty_map:
            penalty = wrinkle_penalty_map[wrinkle]
            score += penalty
            breakdown["wrinkle_penalty"] = penalty

        # Skin uniformity = LAB L* std-dev over the eroded interior face
        # mask. Higher std means uneven tone (shadows, blemishes).
        # De-emphasised: the metric over-penalises attractive faces in
        # warm/directional lighting, which is most photos.
        uniformity = d.get("skin_uniformity")
        if isinstance(uniformity, (int, float)) and uniformity > 0:
            # Empirically uniformity sits ~8-15 in clean skin and 20-30
            # in uneven skin. Cap reduced from 14 to 9.
            penalty = min(9.0, max(0.0, (float(uniformity) - 10.0) * 0.7))
            score += penalty
            breakdown["skin_unevenness_penalty"] = round(penalty, 2)

        # NOTE: freckles_or_moles penalty deliberately removed — the
        # detector was too noisy (shadows / pores counted as spots).

        # Smile asymmetry: 0 = perfectly symmetric smile, larger = lopsided.
        # De-emphasised — even attractive faces have natural smile
        # asymmetry, and the MediaPipe blendshape signal exaggerates it.
        smile_asym = d.get("smile_asymmetry")
        if isinstance(smile_asym, (int, float)):
            penalty = min(6.0, float(smile_asym) * 30.0)
            score += penalty
            breakdown["smile_asymmetry_penalty"] = round(penalty, 2)

        # Photo-quality penalty: sunglasses/mask hide features and the
        # model is guessing more. Mild penalty, not a personal trait.
        if d.get("wearing_sunglasses") or d.get("wearing_mask"):
            score += 8.0   # was 5
            breakdown["obstruction_penalty"] = 8.0

        # Hat coverage also obscures hairline / forehead — small fixed
        # penalty so a hat doesn't accidentally help the score by
        # blocking unflattering hair.
        if d.get("hat_detected"):
            score += 4.0
            breakdown["hat_obscuration_penalty"] = 4.0

        # ── Bonuses (pull score down = less chopped) ─────────────────

        # Defined jawline. EMPHASISED — strong jawline is one of the
        # most consistent visual cues for "conventionally attractive."
        # Two signals combine here:
        #
        #   (a) The MediaPipe `jawline_type` bucket gives a coarse
        #       qualitative read.
        #   (b) The numeric `jawline_angle` (degrees subtended at the
        #       chin by the two gonion landmarks) gives a continuous
        #       signal where lower = sharper. We map it linearly into
        #       a bonus that maxes out at very sharp angles and fades
        #       to zero by ~145°.
        #
        # We take whichever signal is more generous so the cue isn't
        # double-counted on a single face. Numeric bonus scales as:
        #
        #   angle ≤  95°  →  -22  (very sharp)
        #   angle  95-145 →  linearly -22 → 0
        #   angle ≥ 145°  →  0   (very soft)
        jaw_bucket_bonus = 0.0
        jaw_type = d.get("jawline_type")
        jaw_type_bonus_map = {"sharp": -16.0, "strong": -10.0, "soft": 0.0}
        if jaw_type in jaw_type_bonus_map:
            jaw_bucket_bonus = jaw_type_bonus_map[jaw_type]

        jaw_angle_bonus = 0.0
        jaw_angle = d.get("jawline_angle")
        if isinstance(jaw_angle, (int, float)):
            if jaw_angle <= 95:
                jaw_angle_bonus = -22.0
            elif jaw_angle < 145:
                # Linear ramp from -22 at 95° to 0 at 145°.
                jaw_angle_bonus = -22.0 * (145 - jaw_angle) / 50.0
            # else stays 0

        # Use whichever bonus is more pronounced (smaller / more
        # negative number = bigger bonus).
        jaw_bonus = min(jaw_bucket_bonus, jaw_angle_bonus)
        if jaw_bonus:
            score += jaw_bonus
            breakdown["jaw_definition_bonus"] = round(jaw_bonus, 2)

        # Cheekbone prominence.
        cheek_bonus_map = {"high": -11.0, "moderate": -5.0, "flat": 0.0}
        cheek = d.get("cheekbone_prominence")
        if cheek in cheek_bonus_map:
            bonus = cheek_bonus_map[cheek]
            score += bonus
            breakdown["cheekbone_bonus"] = bonus

        # Skin clarity bonus when the texture score is low (smooth).
        texture = d.get("skin_texture_score")
        if isinstance(texture, (int, float)) and 0 < texture <= 8:
            score -= 14.0   # was -9
            breakdown["skin_clarity_bonus"] = -14.0

        # Lip fullness — "average" and "full" both read as healthy.
        lip = d.get("lip_fullness")
        if lip == "full":
            score -= 8.0   # was -5
            breakdown["lip_fullness_bonus"] = -8.0
        elif lip == "average":
            score -= 4.0
            breakdown["lip_fullness_bonus"] = -4.0

        # Defined cupid's bow.
        if d.get("cupids_bow") == "defined":
            score -= 5.0   # was -3
            breakdown["cupids_bow_bonus"] = -5.0

        # Normal eye spacing.
        if d.get("eye_spacing") == "average":
            score -= 6.0   # was -4
            breakdown["eye_spacing_bonus"] = -6.0

        # Symmetric face (independent of asymmetry penalty above; we
        # explicitly reward very symmetric faces rather than just not
        # penalising them).
        if isinstance(asym, (int, float)) and asym < 0.15:
            score -= 6.0
            breakdown["symmetry_bonus"] = -6.0

        # Dimples — small bonus when the MediaPipe heuristic fires.
        if d.get("possible_dimples"):
            score -= 5.0   # was -3
            breakdown["dimples_bonus"] = -5.0

        # Eyes-open bonus (closed eyes makes a face look worse).
        if d.get("eyes_open") is True:
            score -= 3.0
            breakdown["eyes_open_bonus"] = -3.0

        return score, breakdown
