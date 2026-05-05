"""
Color Analyzer — Pixel-level color extraction using masks from
BiSeNet and landmarks from MediaPipe.

Determines:
- Skin tone (Fitzpatrick scale, LAB lightness, hex color)
- Eye color (hue classification from iris region)
- Hair color (LAB-trimmed median over hair mask)
- Hair texture from local intensity variation (Laplacian std over eroded mask)
- Lip color
"""

from typing import Any

import cv2
import numpy as np

# Fitzpatrick scale boundaries based on LAB L* channel (true 0–100 range).
# OpenCV's uint8 LAB stores L scaled to 0–255, so we rescale before lookup.
FITZPATRICK_SCALE = [
    (85, 100, "Type I - Very Fair"),
    (70, 85, "Type II - Fair"),
    (55, 70, "Type III - Medium"),
    (40, 55, "Type IV - Olive"),
    (25, 40, "Type V - Brown"),
    (0, 25, "Type VI - Dark Brown/Black"),
]

EYE_COLOR_RANGES = {
    "brown": {"h_range": (8, 28), "s_min": 50},
    "hazel": {"h_range": (20, 35), "s_min": 40},
    "green": {"h_range": (35, 80), "s_min": 30},
    "blue": {"h_range": (90, 130), "s_min": 30},
    "gray": {"h_range": (0, 180), "s_max": 30},
    "amber": {"h_range": (15, 25), "s_min": 80},
}

# Hair-texture thresholds on std(Laplacian) computed over the *eroded* hair
# mask (so the mask boundary itself doesn't contribute high-frequency energy).
# These are reasonable starting points — tune on your own dataset.
HAIR_TEXTURE_CURLY_THRESHOLD = 25.0
HAIR_TEXTURE_WAVY_THRESHOLD = 15.0


class ColorAnalyzer:
    def __init__(self):
        pass  # No model to load — pure pixel analysis

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

        # ── Skin Tone ────────────────────────────────────────────────
        if skin_mask is not None and skin_mask.sum() > 100:
            skin_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            skin_pixels = skin_lab[skin_mask]

            # OpenCV uint8 LAB stores L in 0–255 and a/b offset by +128.
            # Rescale to the conventional ranges (L* in 0–100, a*/b* in
            # roughly -128..127) so the Fitzpatrick bins and undertone
            # thresholds operate in standard units.
            mean_l_raw = float(np.mean(skin_pixels[:, 0]))
            mean_l = mean_l_raw * 100.0 / 255.0
            mean_a = float(np.mean(skin_pixels[:, 1])) - 128.0
            mean_b = float(np.mean(skin_pixels[:, 2])) - 128.0

            # Fitzpatrick type
            fitz = "Unknown"
            for low, high, label in FITZPATRICK_SCALE:
                if low <= mean_l < high:
                    fitz = label
                    break

            # Get hex color of average skin tone
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

            # Undertone (warm/cool/neutral). Now that b* is centered on 0,
            # positive b* leans yellow (warm) and negative b* leans blue
            # (cool). Thresholds adjusted from the old 0–255 scale.
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
        if landmarks and len(landmarks) > 473:
            eye_color = self._detect_eye_color(img_rgb, landmarks, h, w)
            result["eye_color"] = eye_color
        elif landmarks and len(landmarks) > 362:
            # Fallback: sample from rough iris area
            eye_color = self._detect_eye_color_fallback(img_rgb, landmarks, h, w)
            result["eye_color"] = eye_color
        else:
            result["eye_color"] = "unknown"

        # ── Hair Color ───────────────────────────────────────────────
        if hair_mask is not None and hair_mask.sum() > 200:
            hair_color_info = self._estimate_hair_color(img_rgb, hair_mask)
            result["hair_color"] = hair_color_info

            result["hair_texture"] = self._estimate_hair_texture(img_rgb, hair_mask)
        else:
            result["hair_color"] = {"name": "unknown"}
            result["hair_texture"] = "unknown"

        # ── Lip Color ────────────────────────────────────────────────
        if lip_mask is not None and lip_mask.sum() > 50:
            lip_pixels = img_rgb[lip_mask]
            avg_lip = np.mean(lip_pixels, axis=0).astype(int)
            hex_lip = "#{:02x}{:02x}{:02x}".format(*avg_lip)

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
        """Estimate dominant hair color via LAB-lightness-trimmed median.

        Why median + L*-trim instead of k=2 k-means:
        - K-means with k=2 splits highlight vs shadow within a single hair
          color, so the "bigger cluster" can flip between photos of the same
          person depending on lighting. Median is robust and deterministic.
        - Trimming the top/bottom 10% of L* drops specular highlights and
          deep shadows, which are the main outlier sources.
        """
        hair_pixels = img_rgb[hair_mask]  # (N, 3) uint8 RGB

        # Trim by LAB L* to drop highlights and shadows.
        hair_lab = cv2.cvtColor(
            hair_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB
        ).reshape(-1, 3)
        l_lo, l_hi = np.percentile(hair_lab[:, 0], [10, 90])
        keep = (hair_lab[:, 0] >= l_lo) & (hair_lab[:, 0] <= l_hi)
        core_pixels = hair_pixels[keep] if keep.sum() > 50 else hair_pixels

        dominant_rgb = np.median(core_pixels, axis=0)
        dominant_rgb = np.clip(dominant_rgb, 0, 255).astype(np.uint8)

        hex_hair = "#{:02x}{:02x}{:02x}".format(*dominant_rgb)

        hair_hsv = cv2.cvtColor(
            dominant_rgb.reshape(1, 1, 3), cv2.COLOR_RGB2HSV
        )[0, 0]
        h_val, s_val, v_val = int(hair_hsv[0]), int(hair_hsv[1]), int(hair_hsv[2])

        # Classification cascade — order matters. Falls through to "unknown"
        # rather than a default of "brown" so mask leakage / unusual tints
        # are detectable downstream.
        if v_val < 45 and s_val < 60:
            hair_color_name = "black"
        elif s_val < 25:
            # Low saturation across the V range → gray family.
            hair_color_name = "gray/white" if v_val > 100 else "dark gray"
        elif (h_val < 12 or h_val > 168) and s_val > 60:
            hair_color_name = "red/auburn"
        elif 18 <= h_val <= 35 and v_val > 160 and s_val < 140:
            # Blond: yellow hue, high V, and not too saturated (real blond
            # is desaturated yellow, not orange).
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
        """Estimate hair texture from local intensity variation.

        Computes std(Laplacian) over an *eroded* hair mask. Erosion stays
        strictly inside the hair region so the mask boundary itself doesn't
        contribute the high-frequency step edge that the previous FFT-on-
        zeroed-region implementation was inadvertently measuring.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        inner_mask = cv2.erode(
            hair_mask.astype(np.uint8), kernel, iterations=2
        ).astype(bool)

        if inner_mask.sum() < 200:
            return "unknown"

        hair_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(hair_gray, cv2.CV_64F, ksize=3)
        texture_score = float(np.std(lap[inner_mask]))

        if texture_score > HAIR_TEXTURE_CURLY_THRESHOLD:
            return "curly/coily"
        if texture_score > HAIR_TEXTURE_WAVY_THRESHOLD:
            return "wavy"
        return "straight"

    # ------------------------------------------------------------------
    # Eye color helpers
    # ------------------------------------------------------------------

    def _detect_eye_color(
        self, img_rgb: np.ndarray, lm: list[dict], h: int, w: int
    ) -> str:
        """Use iris landmarks (468-477) to sample eye color."""
        iris_indices = list(range(468, 474))  # Left iris
        iris_points = [(int(lm[i]["x"] * w), int(lm[i]["y"] * h)) for i in iris_indices]

        # Create a small mask around iris center
        cx = int(np.mean([p[0] for p in iris_points]))
        cy = int(np.mean([p[1] for p in iris_points]))
        radius = max(3, int(np.std([p[0] for p in iris_points]) * 1.5))

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

        iris_pixels = img_rgb[mask > 0]
        if len(iris_pixels) < 10:
            return "unknown"

        return self._classify_eye_color(iris_pixels)

    def _detect_eye_color_fallback(
        self, img_rgb: np.ndarray, lm: list[dict], h: int, w: int
    ) -> str:
        """Fallback: sample from center of eye region."""
        # Center of left eye
        eye_pts = [159, 145, 133, 33]
        cx = int(np.mean([lm[i]["x"] for i in eye_pts]) * w)
        cy = int(np.mean([lm[i]["y"] for i in eye_pts]) * h)
        radius = max(3, int(abs(lm[159]["y"] - lm[145]["y"]) * h * 0.3))

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

        iris_pixels = img_rgb[mask > 0]
        if len(iris_pixels) < 5:
            return "unknown"

        return self._classify_eye_color(iris_pixels)

    @staticmethod
    def _classify_eye_color(pixels: np.ndarray) -> str:
        """Classify eye color from pixel samples using HSV."""
        hsv = cv2.cvtColor(
            pixels.reshape(-1, 1, 3).astype(np.uint8),
            cv2.COLOR_RGB2HSV
        ).reshape(-1, 3)

        mean_h = float(np.mean(hsv[:, 0]))
        mean_s = float(np.mean(hsv[:, 1]))
        mean_v = float(np.mean(hsv[:, 2]))

        # Gray eyes: low saturation
        if mean_s < 30:
            return "gray"

        # Classify by hue
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
        if mean_v < 60:
            return "dark brown"

        return "brown"