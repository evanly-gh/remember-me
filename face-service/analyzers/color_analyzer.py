"""
ColorAnalyzer — pixel-level color extraction.

Model
-----
None. All operations are deterministic OpenCV LAB/HSV statistics over
masks/landmarks supplied by upstream analyzers.

Inputs
------
img_rgb    : np.ndarray (H, W, 3) uint8
landmarks  : list[dict] of normalised MediaPipe landmarks (optional)
skin_mask  : bool ndarray (H, W) from SegFormer "face" class (optional)
hair_mask  : bool ndarray (H, W) from SegFormer "hair" class (optional)
lip_mask   : bool ndarray (H, W) — usually None; falls back to MediaPipe
             lip polygon when missing or too small

Outputs (dict)
--------------
skin_tone        — {fitzpatrick, lab_lightness, lab_a, lab_b, hex_color, rgb}
skin_undertone   — warm | cool | neutral
eye_color        — brown | hazel | amber | green | blue | gray | dark brown
hair_color       — {name, hex, rgb, hsv}
hair_texture     — straight | wavy | curly/coily   (coarse Laplacian signal,
                   the HairTypeViT analyzer is the authoritative source)
lip_color        — {shade, hex, rgb}

Notes
-----
LAB is preferred over RGB for skin tone classification because LAB's
L* channel is a perceptual lightness — Fitzpatrick bins line up with
fixed L* ranges regardless of camera white balance.
"""

from typing import Any

import cv2
import numpy as np

# Fitzpatrick scale boundaries on the LAB L* channel (true 0–100 range).
# OpenCV's uint8 LAB stores L scaled to 0–255, so we rescale before
# looking up bins.
FITZPATRICK_SCALE = [
    (85, 100, "Type I - Very Fair"),
    (70, 85, "Type II - Fair"),
    (55, 70, "Type III - Medium"),
    (40, 55, "Type IV - Olive"),
    (25, 40, "Type V - Brown"),
    (0, 25, "Type VI - Dark Brown/Black"),
]

# Hair-texture thresholds on std(Laplacian) computed over the *eroded*
# hair mask. Erosion prevents the mask boundary from contributing
# high-frequency step-edge energy.
HAIR_TEXTURE_CURLY_THRESHOLD = 25.0
HAIR_TEXTURE_WAVY_THRESHOLD = 15.0

# MediaPipe FaceMesh lip contours. The outer ring traces the lip
# border; the inner ring traces the mouth opening. Filling outer
# and then erasing inner gives only lip flesh, never teeth/tongue.
MEDIAPIPE_LIP_OUTER = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
MEDIAPIPE_LIP_INNER = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]


class ColorAnalyzer:
    def __init__(self):
        # No model to load — pure pixel arithmetic.
        pass

    def analyze(
        self,
        img_rgb: np.ndarray,
        landmarks: list[dict] | None = None,
        skin_mask: np.ndarray | None = None,
        hair_mask: np.ndarray | None = None,
        lip_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        h, w = img_rgb.shape[:2]
        result: dict[str, Any] = {}

        # Coerce masks to boolean so fancy-indexing selects pixels rather
        # than misinterpreting uint8 0/255 values as integer indices.
        if skin_mask is not None:
            skin_mask = skin_mask.astype(bool)
        if hair_mask is not None:
            hair_mask = hair_mask.astype(bool)
        if lip_mask is not None:
            lip_mask = lip_mask.astype(bool)

        # SegFormer human-parsing has no lip class, so callers usually
        # pass None for lip_mask. Build one from MediaPipe lip landmarks
        # whenever it's missing or too small to sample reliably.
        if (lip_mask is None or lip_mask.sum() < 50) and landmarks:
            derived = self._lip_mask_from_landmarks(landmarks, h, w)
            if derived is not None:
                lip_mask = derived

        # ── Skin Tone ────────────────────────────────────────────────
        # Need at least ~100 face pixels for stable statistics.
        if skin_mask is not None and skin_mask.sum() > 100:
            # Convert the whole image to LAB once and pull pixels under
            # the mask. cv2 returns uint8 LAB with L in 0–255 and a/b
            # offset by +128 (so neutral gray is L=128, a=128, b=128).
            skin_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            skin_pixels = skin_lab[skin_mask]

            # Rescale to standard LAB ranges before applying the
            # Fitzpatrick / undertone thresholds defined on those ranges.
            mean_l_raw = float(np.mean(skin_pixels[:, 0]))
            mean_l = mean_l_raw * 100.0 / 255.0
            mean_a = float(np.mean(skin_pixels[:, 1])) - 128.0
            mean_b = float(np.mean(skin_pixels[:, 2])) - 128.0

            # Bin into Fitzpatrick types — linear search over six bands.
            fitz = "Unknown"
            for low, high, label in FITZPATRICK_SCALE:
                if low <= mean_l < high:
                    fitz = label
                    break

            # Average RGB → hex for display.
            avg_rgb = np.mean(img_rgb[skin_mask], axis=0).astype(int)
            hex_color = "#{:02x}{:02x}{:02x}".format(*avg_rgb)

            result["skin_tone"] = {
                "fitzpatrick": fitz,
                "lab_lightness": round(mean_l, 1),
                "lab_a": round(mean_a, 1),
                "lab_b": round(mean_b, 1),
                "hex_color": hex_color,
                "rgb": avg_rgb.tolist(),
            }

            # Undertone from b* (yellow ↔ blue axis):
            # b* > +12  → yellow-leaning, warm
            # b* < -8   → blue-leaning,   cool
            # in between → neutral
            if mean_b > 12:
                result["skin_undertone"] = "warm"
            elif mean_b < -8:
                result["skin_undertone"] = "cool"
            else:
                result["skin_undertone"] = "neutral"
        else:
            result["skin_tone"] = {"fitzpatrick": "unknown"}
            result["skin_undertone"] = "unknown"

        # ── Eye Color ────────────────────────────────────────────────
        # Prefer the dedicated iris landmarks (468-477) when available.
        # Fall back to a rough eye-centre crop otherwise.
        if landmarks and len(landmarks) > 473:
            result["eye_color"] = self._detect_eye_color(img_rgb, landmarks, h, w)
        elif landmarks and len(landmarks) > 362:
            result["eye_color"] = self._detect_eye_color_fallback(img_rgb, landmarks, h, w)
        else:
            result["eye_color"] = "unknown"

        # ── Hair Color & Texture ────────────────────────────────────
        # Need at least 200 hair pixels for a stable median.
        if hair_mask is not None and hair_mask.sum() > 200:
            result["hair_color"] = self._estimate_hair_color(img_rgb, hair_mask)
            result["hair_texture"] = self._estimate_hair_texture(img_rgb, hair_mask)
        else:
            result["hair_color"] = {"name": "unknown"}
            result["hair_texture"] = "unknown"

        # ── Lip Color ────────────────────────────────────────────────
        # Average the masked lip pixels and bucket by HSV saturation/value.
        if lip_mask is not None and lip_mask.sum() > 50:
            lip_pixels = img_rgb[lip_mask]
            avg_lip = np.mean(lip_pixels, axis=0).astype(int)
            hex_lip = "#{:02x}{:02x}{:02x}".format(*avg_lip)

            # Convert the single average RGB triple to HSV for shade
            # classification. High saturation → rosy/red; high value but
            # low saturation → pink; low value → dark; otherwise natural.
            lip_hsv = cv2.cvtColor(
                avg_lip.reshape(1, 1, 3).astype(np.uint8),
                cv2.COLOR_RGB2HSV
            )[0, 0]
            lip_s = int(lip_hsv[1])
            lip_v = int(lip_hsv[2])

            if lip_s > 100:
                lip_shade = "rosy/red"
            elif lip_v > 160:
                lip_shade = "pink"
            elif lip_v < 80:
                lip_shade = "dark"
            else:
                lip_shade = "natural"

            result["lip_color"] = {
                "shade": lip_shade,
                "hex": hex_lip,
                "rgb": avg_lip.tolist(),
            }
        else:
            result["lip_color"] = {"shade": "unknown"}

        return result

    # ------------------------------------------------------------------
    # Hair color helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_hair_color(
        img_rgb: np.ndarray, hair_mask: np.ndarray
    ) -> dict[str, Any]:
        """Dominant hair color via LAB-lightness-trimmed median.

        Why median + L*-trim instead of k=2 k-means:
        - K-means with k=2 splits highlight vs shadow within a single
          hair color, so the "bigger cluster" can flip between photos
          of the same person depending on lighting. Median is robust
          and deterministic.
        - Trimming the top/bottom 10% of L* drops specular highlights
          and deep shadows, the main outlier sources.
        """
        hair_pixels = img_rgb[hair_mask]  # (N, 3) uint8 RGB

        # LAB conversion so we can trim by perceptual lightness.
        hair_lab = cv2.cvtColor(
            hair_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
        ).reshape(-1, 3)
        l_lo, l_hi = np.percentile(hair_lab[:, 0], [10, 90])
        keep = (hair_lab[:, 0] >= l_lo) & (hair_lab[:, 0] <= l_hi)
        # If trimming would leave us too few pixels, fall back to all.
        core_pixels = hair_pixels[keep] if keep.sum() > 50 else hair_pixels

        # Median is robust to mask leakage (a few stray non-hair pixels
        # don't shift the median).
        dominant_rgb = np.median(core_pixels, axis=0)
        dominant_rgb = np.clip(dominant_rgb, 0, 255).astype(np.uint8)

        hex_hair = "#{:02x}{:02x}{:02x}".format(*dominant_rgb)

        # Bucket the dominant color into a name via HSV thresholds.
        hair_hsv = cv2.cvtColor(
            dominant_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2HSV
        )[0, 0]
        h_val, s_val, v_val = int(hair_hsv[0]), int(hair_hsv[1]), int(hair_hsv[2])

        # Classification cascade — order matters. Falls through to
        # "unknown" instead of defaulting to a colour, so mask leakage
        # and unusual tints stay detectable downstream.
        if v_val < 45 and s_val < 60:
            hair_color_name = "black"
        elif s_val < 25:
            # Low saturation across the V range → gray family.
            hair_color_name = "gray/white" if v_val > 100 else "dark gray"
        elif (h_val < 12 or h_val > 168) and s_val > 60:
            hair_color_name = "red/auburn"
        elif 18 <= h_val <= 35 and v_val > 160 and s_val < 140:
            # Blond is desaturated yellow with high V — bright but not
            # too saturated (or it'd shade orange).
            hair_color_name = "blond"
        elif 5 <= h_val <= 30:
            hair_color_name = "brown" if v_val > 80 else "dark brown"
        else:
            hair_color_name = "unknown"

        return {
            "name": hair_color_name,
            "hex": hex_hair,
            "rgb": dominant_rgb.tolist(),
            "hsv": [h_val, s_val, v_val],
        }

    @staticmethod
    def _estimate_hair_texture(
        img_rgb: np.ndarray, hair_mask: np.ndarray
    ) -> str:
        """Coarse hair texture from local intensity variation.

        Computes std(Laplacian) over an *eroded* hair mask so the mask
        boundary itself doesn't contribute the high-frequency step
        edge that an un-eroded mask would.

        This is intentionally a fallback signal; the authoritative
        hair-texture output is HairTypeViT (curly/dreadlocks/kinky/
        straight/wavy), which is trained and ~93% accurate.
        """
        # Erode by ~10 px so we sample only deep-interior hair pixels.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        inner_mask = cv2.erode(
            hair_mask.astype(np.uint8), kernel, iterations=2
        ).astype(bool)

        # Not enough interior pixels to compute a reliable std.
        if inner_mask.sum() < 200:
            return "unknown"

        # Laplacian responds to local intensity curvature; its std over
        # the masked region is a proxy for "how much fine detail".
        hair_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(hair_gray, cv2.CV_64F, ksize=3)
        texture_score = float(np.std(lap[inner_mask]))

        if texture_score > HAIR_TEXTURE_CURLY_THRESHOLD:
            return "curly/coily"
        if texture_score > HAIR_TEXTURE_WAVY_THRESHOLD:
            return "wavy"
        return "straight"

    # ------------------------------------------------------------------
    # Lip mask helper
    # ------------------------------------------------------------------

    @staticmethod
    def _lip_mask_from_landmarks(
        landmarks: list[dict], h: int, w: int
    ) -> np.ndarray | None:
        """Build a lip-flesh mask by filling outer minus inner contour."""
        # Bail if the landmark list doesn't have indices the contours
        # reference (e.g. iris-less subset).
        max_idx = max(MEDIAPIPE_LIP_OUTER + MEDIAPIPE_LIP_INNER)
        if len(landmarks) <= max_idx:
            return None

        # Helper to convert a list of landmark indices into a pixel-
        # space polygon in (x, y) order.
        def _poly(indices: list[int]) -> np.ndarray:
            return np.array(
                [
                    [int(landmarks[i]["x"] * w), int(landmarks[i]["y"] * h)]
                    for i in indices
                ],
                dtype=np.int32,
            )

        # Fill the outer ring, then erase the inner ring → lip flesh
        # only, no teeth or tongue pixels.
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [_poly(MEDIAPIPE_LIP_OUTER)], 255)
        cv2.fillPoly(mask, [_poly(MEDIAPIPE_LIP_INNER)], 0)
        return mask.astype(bool)

    # ------------------------------------------------------------------
    # Eye color helpers
    # ------------------------------------------------------------------

    def _detect_eye_color(
        self, img_rgb: np.ndarray, lm: list[dict], h: int, w: int
    ) -> str:
        """Sample left-iris pixels using MediaPipe iris landmarks (468–477)."""
        # 468-473 cover the left iris ring; we average them to a centre
        # and pick a radius from the std-dev of the x-coordinates.
        iris_indices = list(range(468, 474))
        iris_points = [(int(lm[i]["x"] * w), int(lm[i]["y"] * h)) for i in iris_indices]

        cx = int(np.mean([p[0] for p in iris_points]))
        cy = int(np.mean([p[1] for p in iris_points]))
        radius = max(3, int(np.std([p[0] for p in iris_points]) * 1.5))

        # Filled disc mask centred on the iris → classify those pixels.
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

        iris_pixels = img_rgb[mask > 0]
        if len(iris_pixels) < 10:
            return "unknown"

        return self._classify_eye_color(iris_pixels)

    def _detect_eye_color_fallback(
        self, img_rgb: np.ndarray, lm: list[dict], h: int, w: int
    ) -> str:
        """Fallback when iris landmarks aren't available.

        Averages four points that bound the eye opening and treats the
        centre as a coarse "look here" target. Less accurate than the
        iris-landmark path because we sample some sclera too, but it's
        a graceful degradation.
        """
        eye_pts = [159, 145, 133, 33]
        cx = int(np.mean([lm[i]["x"] for i in eye_pts]) * w)
        cy = int(np.mean([lm[i]["y"] for i in eye_pts]) * h)
        # Radius scaled to ~30% of eye opening height.
        radius = max(3, int(abs(lm[159]["y"] - lm[145]["y"]) * h * 0.3))

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

        iris_pixels = img_rgb[mask > 0]
        if len(iris_pixels) < 5:
            return "unknown"

        return self._classify_eye_color(iris_pixels)

    @staticmethod
    def _classify_eye_color(pixels: np.ndarray) -> str:
        """Bucket sampled iris pixels by HSV mean.

        Hue ranges follow the standard OpenCV scale (H in 0–180, not
        0–360). The cascade order matters: gray is checked first because
        any sufficiently desaturated eye is gray regardless of its
        nominal hue.
        """
        hsv = cv2.cvtColor(
            pixels.reshape(-1, 1, 3).astype(np.uint8),
            cv2.COLOR_RGB2HSV
        ).reshape(-1, 3)

        mean_h = float(np.mean(hsv[:, 0]))
        mean_s = float(np.mean(hsv[:, 1]))
        mean_v = float(np.mean(hsv[:, 2]))

        # Gray eyes: any hue, but low saturation.
        if mean_s < 30:
            return "gray"

        # Hue-based buckets. Specific (amber) before general (brown).
        if 90 <= mean_h <= 130 and mean_s > 30:
            return "blue"
        if 35 <= mean_h <= 80 and mean_s > 30:
            return "green"
        if 20 <= mean_h <= 35 and mean_s > 40:
            return "hazel"
        if 15 <= mean_h <= 25 and mean_s > 80:
            return "amber"
        if 8 <= mean_h <= 28 and mean_s > 50:
            return "brown"
        # Anything left with low V is just dark brown.
        if mean_v < 60:
            return "dark brown"

        return "brown"
