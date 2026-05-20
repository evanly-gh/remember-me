"""
AestheticAnalyzer — "chopped score" aggregator.

What it does
------------
Reads the merged result dict from every other analyzer and produces a
single numeric "chopped score" plus a per-factor breakdown. Higher
score = more chopped = less conventionally attractive (by the
arbitrary rubric encoded here). The breakdown lets you tune weights
or flip polarity client-side without rerunning inference.

Score composition
-----------------
The final chopped_score is a weighted blend of two sources:

1. **Learned beauty regressor** (from BeautyAnalyzer, trained on
   SCUT-FBP5500): a number in [1.0, 5.0] reflecting averaged human
   ratings. We rescale to a 0–100 unattractiveness axis. This is the
   dominant signal when available — heavy weight (default 0.7).

2. **Rule-based factor sum**: penalties for asymmetry, wrinkles,
   uneven skin, freckles, and asymmetric smile; bonuses for defined
   jawline, prominent cheekbones, clear skin, balanced lips, and
   dimples. Each factor is documented in `_compute_rule_score`.
   This is the only signal when the regressor isn't loaded
   (BeautyAnalyzer returns None).

Blend math
----------
    if beauty_score available:
        chopped = 0.7 * (100 - beauty_norm) + 0.3 * rule_score
    else:
        chopped = rule_score
    chopped is clamped to [0, 100].

Subjectivity disclaimer
-----------------------
Every weight in this file is a guess. "Beauty" is subjective, culturally
biased, and reductive. Treat the score as an in-joke metric; never
expose it as objective truth. The UI gates the row behind a
Settings toggle off-by-default for that reason.

Note: this analyzer takes no image input — it reads the merged result
dict produced by every other analyzer that ran ahead of it.
"""

from typing import Any


# How much weight the learned beauty regressor gets when both signals
# are available. The rule-based sum gets the rest (1 - this).
LEARNED_WEIGHT = 0.7

# Baseline score. Penalties push up, bonuses pull down.
BASELINE = 50.0


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
            # Beauty regressor: 0 = ugly, 100 = beautiful (per SCUT-FBP5500
            # scaling). Flip to unattractiveness axis: 100 - x.
            learned_unattractive = 100.0 - float(beauty_norm)
            chopped = (
                LEARNED_WEIGHT * learned_unattractive
                + (1.0 - LEARNED_WEIGHT) * rule_score
            )
            breakdown["learned_unattractive"] = round(
                LEARNED_WEIGHT * learned_unattractive - LEARNED_WEIGHT * BASELINE, 2
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

        Returns (score, breakdown_dict). The breakdown gives each factor's
        signed contribution so a UI can show *why* a score landed where
        it did. Score starts at BASELINE (50) and moves up/down.
        """
        score = BASELINE
        breakdown: dict[str, float] = {}

        # ── Penalties (push score up = more chopped) ─────────────────

        # Facial asymmetry: 0 = perfectly symmetric, 1 = very asymmetric.
        # MediaPipe `facial_asymmetry_score` is already in this range.
        asym = d.get("facial_asymmetry_score")
        if isinstance(asym, (int, float)):
            penalty = float(asym) * 18.0
            score += penalty
            breakdown["asymmetry_penalty"] = round(penalty, 2)

        # Wrinkle level from SegFormer + OpenCV Laplacian classification.
        wrinkle_penalty_map = {
            "smooth": 0.0, "slight": 4.0, "moderate": 8.0, "prominent": 12.0,
        }
        wrinkle = d.get("wrinkle_level")
        if wrinkle in wrinkle_penalty_map:
            penalty = wrinkle_penalty_map[wrinkle]
            score += penalty
            breakdown["wrinkle_penalty"] = penalty

        # Skin uniformity = LAB L* std-dev over the face mask. Higher
        # std means uneven tone (shadows, blemishes). Scale up to +8.
        uniformity = d.get("skin_uniformity")
        if isinstance(uniformity, (int, float)) and uniformity > 0:
            # Empirically, uniformity in clean skin is ~8-15; very uneven
            # skin pushes into the 20-30 range.
            penalty = min(8.0, max(0.0, (float(uniformity) - 10.0) * 0.5))
            score += penalty
            breakdown["skin_unevenness_penalty"] = round(penalty, 2)

        # Freckles/moles bucket.
        freckle_penalty_map = {"none": 0.0, "few": 1.0, "some": 3.0, "many": 5.0}
        freckles = d.get("freckles_or_moles")
        if freckles in freckle_penalty_map:
            penalty = freckle_penalty_map[freckles]
            score += penalty
            breakdown["freckles_penalty"] = penalty

        # Smile asymmetry: 0 = perfectly symmetric smile, larger = lopsided.
        smile_asym = d.get("smile_asymmetry")
        if isinstance(smile_asym, (int, float)):
            penalty = min(6.0, float(smile_asym) * 30.0)
            score += penalty
            breakdown["smile_asymmetry_penalty"] = round(penalty, 2)

        # Photo-quality penalty: sunglasses/mask hide features and the
        # model is guessing more. Mild penalty, not a personal trait.
        if d.get("wearing_sunglasses") or d.get("wearing_mask"):
            score += 5.0
            breakdown["obstruction_penalty"] = 5.0

        # ── Bonuses (pull score down = less chopped) ─────────────────

        # Defined jawline. Two signals (string bucket + numeric angle);
        # take the stronger of the two contributions.
        jaw_bonus = 0.0
        jaw_type = d.get("jawline_type")
        jaw_type_bonus_map = {"sharp": -10.0, "strong": -6.0, "soft": 0.0}
        if jaw_type in jaw_type_bonus_map:
            jaw_bonus = jaw_type_bonus_map[jaw_type]
        jaw_angle = d.get("jawline_angle")
        if isinstance(jaw_angle, (int, float)) and jaw_angle < 115:
            # Sharp angles add more on top of the categorical signal.
            jaw_bonus = min(jaw_bonus, -10.0)
        if jaw_bonus:
            score += jaw_bonus
            breakdown["jaw_definition_bonus"] = round(jaw_bonus, 2)

        # Cheekbone prominence.
        cheek_bonus_map = {"high": -7.0, "moderate": -3.0, "flat": 0.0}
        cheek = d.get("cheekbone_prominence")
        if cheek in cheek_bonus_map:
            bonus = cheek_bonus_map[cheek]
            score += bonus
            breakdown["cheekbone_bonus"] = bonus

        # Skin clarity bonus when the texture score is low (i.e. smooth skin).
        # skin_texture_score is the same Laplacian-density value used by
        # wrinkle_level; ≤4 is "smooth" territory.
        texture = d.get("skin_texture_score")
        if isinstance(texture, (int, float)) and 0 < texture <= 4:
            score -= 9.0
            breakdown["skin_clarity_bonus"] = -9.0

        # Lip fullness — "average" and "full" both read as healthy.
        lip = d.get("lip_fullness")
        if lip in {"average", "full"}:
            score -= 5.0
            breakdown["lip_fullness_bonus"] = -5.0

        # Defined cupid's bow.
        if d.get("cupids_bow") == "defined":
            score -= 3.0
            breakdown["cupids_bow_bonus"] = -3.0

        # Normal eye spacing.
        if d.get("eye_spacing") == "average":
            score -= 4.0
            breakdown["eye_spacing_bonus"] = -4.0

        # Dimples — small bonus when the MediaPipe heuristic fires.
        if d.get("possible_dimples"):
            score -= 3.0
            breakdown["dimples_bonus"] = -3.0

        return score, breakdown
