"""
Color Analyzer — Pixel-level color extraction using masks from
BiSeNet and landmarks from MediaPipe.

Determines:
- Skin tone (Fitzpatrick scale, LAB lightness, hex color)
- Eye color (hue classification from iris region)
- Hair color (K-means dominant color from hair mask)
- Hair texture hint from FFT frequency analysis
- Lip color
"""

from typing import Any

import cv2
import numpy as np

# Fitzpatrick scale boundaries based on LAB L* channel
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

        # ── Skin Tone ────────────────────────────────────────────────
        if skin_mask is not None and skin_mask.sum() > 100:
            skin_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            skin_pixels = skin_lab[skin_mask]

            mean_l = float(np.mean(skin_pixels[:, 0]))
            mean_a = float(np.mean(skin_pixels[:, 1]))
            mean_b = float(np.mean(skin_pixels[:, 2]))

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

            # Undertone (warm/cool/neutral)
            if mean_b > 140:
                result["skin_undertone"] = "warm"
            elif mean_b < 120:
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
            hair_pixels = img_rgb[hair_mask]

            # K-means for dominant color
            pixels_float = hair_pixels.astype(np.float32)
            # Sample up to 5000 pixels for speed
            if len(pixels_float) > 5000:
                idx = np.random.choice(len(pixels_float), 5000, replace=False)
                pixels_float = pixels_float[idx]

            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(
                pixels_float, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
            # Pick the cluster with more pixels
            counts = np.bincount(labels.flatten())
            dominant_center = centers[np.argmax(counts)].astype(int)
            hex_hair = "#{:02x}{:02x}{:02x}".format(*dominant_center)

            # Classify by luminance
            hair_hsv = cv2.cvtColor(
                dominant_center.reshape(1, 1, 3).astype(np.uint8),
                cv2.COLOR_RGB2HSV
            )[0, 0]
            h_val, s_val, v_val = int(hair_hsv[0]), int(hair_hsv[1]), int(hair_hsv[2])

            if v_val < 50:
                hair_color_name = "black"
            elif v_val > 180 and s_val < 40:
                hair_color_name = "gray/white"
            elif h_val < 15 or h_val > 165:
                hair_color_name = "red/auburn"
            elif 15 <= h_val < 25 and v_val > 150:
                hair_color_name = "blond"
            elif 10 <= h_val < 25:
                hair_color_name = "brown"
            else:
                hair_color_name = "brown"

            result["hair_color"] = {
                "name": hair_color_name,
                "hex": hex_hair,
                "rgb": dominant_center.tolist(),
            }

            # Hair texture from FFT (frequency analysis)
            hair_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            hair_region = hair_gray.copy()
            hair_region[~hair_mask] = 0

            # Crop to hair bounding box for FFT
            rows = np.any(hair_mask, axis=1)
            cols = np.any(hair_mask, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                crop = hair_region[rmin:rmax + 1, cmin:cmax + 1].astype(np.float32)

                if crop.shape[0] > 10 and crop.shape[1] > 10:
                    f_transform = np.fft.fft2(crop)
                    f_shift = np.fft.fftshift(f_transform)
                    magnitude = np.log1p(np.abs(f_shift))
                    high_freq_ratio = np.sum(magnitude > np.mean(magnitude) + np.std(magnitude))
                    total_freq = magnitude.size
                    hf_ratio = high_freq_ratio / total_freq if total_freq else 0

                    if hf_ratio > 0.25:
                        result["hair_texture"] = "curly/coily"
                    elif hf_ratio > 0.15:
                        result["hair_texture"] = "wavy"
                    else:
                        result["hair_texture"] = "straight"
                else:
                    result["hair_texture"] = "unknown"
            else:
                result["hair_texture"] = "unknown"
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
