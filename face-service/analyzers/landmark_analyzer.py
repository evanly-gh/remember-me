"""
MediaPipe Face Landmarker — 478 3D landmarks + 52 blendshapes.
Derives geometric facial features from landmark positions using pure math.

This is the backbone of the system. From 478 3D points placed on the face,
we calculate distances, ratios, and angles to determine:
- Face shape (oval, round, square, heart, diamond, oblong, triangle)
- Jawline type (sharp, soft, strong)
- Chin type (pointed, wide, normal)
- Cheekbone prominence
- Forehead width
- Eye shape, spacing, size, depth
- Eyebrow shape, arch, thickness
- Nose shape, bridge height, nostril width, tip shape
- Lip fullness, mouth width, cupid's bow
- Smile detection, asymmetry, dimples
- Overall facial asymmetry score
"""

import math
import os
import urllib.request
from typing import Any

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = "models/face_landmarker.task"


class LandmarkAnalyzer:
    def __init__(self):
        base_options = mp_python.BaseOptions(
            model_asset_path=self._ensure_model()
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _ensure_model() -> str:
        """Download the MediaPipe model if not already cached."""
        if not os.path.exists(MODEL_PATH):
            os.makedirs("models", exist_ok=True)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return MODEL_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.detector.detect(mp_image)

        if not result.face_landmarks:
            return {"error": "No face detected by MediaPipe"}

        landmarks = result.face_landmarks[0]
        lm = [{"x": l.x, "y": l.y, "z": l.z} for l in landmarks]

        blendshapes: dict[str, float] = {}
        if result.face_blendshapes:
            for bs in result.face_blendshapes[0]:
                blendshapes[bs.category_name] = round(bs.score, 4)

        attrs: dict[str, Any] = {"_raw_landmarks": lm}

        # ── Face Shape ────────────────────────────────────────────────
        face_height = self._dist(lm[10], lm[152])
        face_width = self._dist(lm[234], lm[454])
        jaw_width = self._dist(lm[172], lm[397])
        cheekbone_width = self._dist(lm[93], lm[323])
        forehead_width = self._dist(lm[54], lm[284])

        wh_ratio = face_width / face_height if face_height else 1
        jaw_to_face = jaw_width / face_width if face_width else 1
        forehead_to_jaw = forehead_width / jaw_width if jaw_width else 1
        cheek_to_jaw = cheekbone_width / jaw_width if jaw_width else 1

        if wh_ratio > 0.85 and jaw_to_face > 0.75:
            attrs["face_shape"] = "round"
        elif wh_ratio > 0.8 and jaw_to_face > 0.8 and forehead_to_jaw < 1.1:
            attrs["face_shape"] = "square"
        elif wh_ratio < 0.75:
            attrs["face_shape"] = "oblong"
        elif forehead_to_jaw > 1.3:
            attrs["face_shape"] = "heart"
        elif cheek_to_jaw > 1.25 and forehead_to_jaw < 1.15:
            attrs["face_shape"] = "diamond"
        elif forehead_to_jaw < 0.85:
            attrs["face_shape"] = "triangle"
        else:
            attrs["face_shape"] = "oval"

        attrs["face_shape_metrics"] = {
            "width_height_ratio": round(wh_ratio, 3),
            "jaw_to_face_ratio": round(jaw_to_face, 3),
            "forehead_to_jaw_ratio": round(forehead_to_jaw, 3),
            "cheekbone_to_jaw_ratio": round(cheek_to_jaw, 3),
        }

        # ── Forehead ─────────────────────────────────────────────────
        fh_ratio = forehead_width / face_width if face_width else 0.6
        attrs["forehead_width"] = (
            "broad" if fh_ratio > 0.7 else "narrow" if fh_ratio < 0.55 else "average"
        )

        # ── Jawline ──────────────────────────────────────────────────
        jaw_angle = self._jaw_angle(lm)
        attrs["jawline_angle"] = round(jaw_angle, 1)
        if jaw_angle < 110:
            attrs["jawline_type"] = "sharp"
        elif jaw_angle > 140:
            attrs["jawline_type"] = "soft"
        elif jaw_to_face > 0.75:
            attrs["jawline_type"] = "strong"
        else:
            attrs["jawline_type"] = "soft"

        # ── Chin ─────────────────────────────────────────────────────
        chin_width = self._dist(lm[175], lm[396])
        chin_ratio = chin_width / jaw_width if jaw_width else 0.4
        attrs["chin_type"] = (
            "pointed" if chin_ratio < 0.3
            else "wide" if chin_ratio > 0.5
            else "normal"
        )

        # ── Cheekbones ───────────────────────────────────────────────
        cheek_z = (lm[93]["z"] + lm[323]["z"]) / 2
        attrs["cheekbone_prominence"] = (
            "high" if cheek_z < -0.04
            else "flat" if cheek_z > 0.0
            else "moderate"
        )
        cheek_puff = blendshapes.get("cheekPuff", 0)
        if cheek_puff > 0.3:
            attrs["cheek_fullness"] = "full"
        elif cheek_z > -0.01:
            attrs["cheek_fullness"] = "hollow"
        else:
            attrs["cheek_fullness"] = "normal"

        # ── Eyes ─────────────────────────────────────────────────────
        l_top, l_bot = lm[159], lm[145]
        l_inner, l_outer = lm[133], lm[33]
        eye_open = self._dist(l_top, l_bot)
        eye_w = self._dist(l_inner, l_outer)
        eye_ratio = eye_open / eye_w if eye_w else 0.3

        outer_angle = l_outer["y"] - l_inner["y"]
        if outer_angle < -0.012:
            attrs["eye_shape"] = "upturned"
        elif outer_angle > 0.012:
            attrs["eye_shape"] = "downturned"
        elif eye_ratio > 0.38:
            attrs["eye_shape"] = "round"
        elif eye_ratio < 0.2:
            attrs["eye_shape"] = "hooded"
        else:
            attrs["eye_shape"] = "almond"

        # Deep-set vs protruding
        eye_z = (lm[159]["z"] + lm[145]["z"]) / 2
        nose_bridge_z = lm[6]["z"]
        if eye_z > nose_bridge_z + 0.02:
            attrs["eye_depth"] = "deep-set"
        elif eye_z < nose_bridge_z - 0.01:
            attrs["eye_depth"] = "protruding"
        else:
            attrs["eye_depth"] = "normal"

        # Eye spacing
        if len(lm) > 473:
            inter_pupillary = self._dist(lm[468], lm[473])
        else:
            inter_pupillary = self._dist(lm[133], lm[362])
        ip_ratio = inter_pupillary / face_width if face_width else 0.35
        attrs["eye_spacing"] = (
            "wide-set" if ip_ratio > 0.38
            else "close-set" if ip_ratio < 0.28
            else "average"
        )

        # Eye size
        r_top, r_bot = lm[386], lm[374]
        r_inner, r_outer = lm[362], lm[263]
        r_area = self._dist(r_top, r_bot) * self._dist(r_inner, r_outer)
        l_area = eye_open * eye_w
        avg_eye_area = (l_area + r_area) / 2
        face_area = face_width * face_height
        es_ratio = avg_eye_area / face_area if face_area else 0.015
        attrs["eye_size"] = (
            "large" if es_ratio > 0.02
            else "small" if es_ratio < 0.012
            else "average"
        )

        blink_l = blendshapes.get("eyeBlinkLeft", 0)
        blink_r = blendshapes.get("eyeBlinkRight", 0)
        attrs["eyes_open"] = (blink_l + blink_r) / 2 < 0.5

        # ── Eyebrows ─────────────────────────────────────────────────
        brow_mid = lm[105]
        brow_outer = lm[46]
        brow_inner = lm[70]
        brow_to_eye = self._dist(brow_mid, lm[159])
        brow_arch_ratio = brow_to_eye / eye_open if eye_open else 1.5

        attrs["eyebrow_arch_height"] = (
            "high" if brow_arch_ratio > 2.2
            else "low" if brow_arch_ratio < 1.3
            else "average"
        )

        mid_y = brow_mid["y"]
        avg_end_y = (brow_inner["y"] + brow_outer["y"]) / 2
        curvature = mid_y - avg_end_y
        if abs(curvature) < 0.003:
            attrs["eyebrow_shape"] = "straight"
        elif curvature < -0.008:
            attrs["eyebrow_shape"] = "arched"
        else:
            attrs["eyebrow_shape"] = "flat"

        brow_top = lm[66]
        brow_bottom = lm[105]
        brow_thickness = self._dist(brow_top, brow_bottom)
        attrs["eyebrow_thickness"] = (
            "thick" if brow_thickness > 0.015
            else "thin" if brow_thickness < 0.008
            else "medium"
        )

        inner_brow_dist = self._dist(lm[70], lm[300])
        attrs["possible_unibrow"] = inner_brow_dist < 0.04

        # ── Nose ─────────────────────────────────────────────────────
        nose_bridge_top = lm[6]
        nose_tip = lm[1]
        nose_bottom = lm[2]
        left_nostril = lm[129]
        right_nostril = lm[358]
        nostril_w = self._dist(left_nostril, right_nostril)

        nw_ratio = nostril_w / face_width if face_width else 0.24
        attrs["nostril_width"] = (
            "wide" if nw_ratio > 0.28
            else "narrow" if nw_ratio < 0.2
            else "average"
        )

        tip_angle = nose_tip["y"] - nose_bottom["y"]
        if tip_angle < -0.005:
            attrs["nose_shape"] = "upturned"
        elif tip_angle > 0.01:
            attrs["nose_shape"] = "aquiline"
        elif nw_ratio > 0.28:
            attrs["nose_shape"] = "wide"
        elif nw_ratio < 0.2:
            attrs["nose_shape"] = "narrow"
        else:
            attrs["nose_shape"] = "straight"

        attrs["nose_bridge"] = (
            "high" if nose_bridge_top["z"] < -0.05
            else "flat" if nose_bridge_top["z"] > 0.0
            else "average"
        )
        attrs["nose_tip_shape"] = (
            "pointed" if nose_tip["z"] < nose_bottom["z"] - 0.01 else "rounded"
        )

        # ── Lips & Mouth ─────────────────────────────────────────────
        ul_top, ul_bot = lm[0], lm[13]
        ll_top, ll_bot = lm[14], lm[17]
        m_left, m_right = lm[61], lm[291]

        ul_h = self._dist(ul_top, ul_bot)
        ll_h = self._dist(ll_top, ll_bot)
        total_lip = ul_h + ll_h
        mouth_w = self._dist(m_left, m_right)

        lip_ratio = total_lip / mouth_w if mouth_w else 0.3
        attrs["lip_fullness"] = (
            "full" if lip_ratio > 0.38
            else "thin" if lip_ratio < 0.22
            else "average"
        )
        attrs["lip_balance"] = (
            "top-heavy" if ul_h > ll_h * 1.2
            else "bottom-heavy" if ll_h > ul_h * 1.2
            else "balanced"
        )

        mw_ratio = mouth_w / face_width if face_width else 0.37
        attrs["mouth_width"] = (
            "wide" if mw_ratio > 0.42
            else "small" if mw_ratio < 0.32
            else "average"
        )

        # Cupid's bow
        c_left, c_center, c_right = lm[37], lm[0], lm[267]
        bow = c_center["y"] - (c_left["y"] + c_right["y"]) / 2
        attrs["cupids_bow"] = (
            "defined" if bow > 0.005
            else "subtle" if bow > 0.002
            else "flat"
        )

        # Smile
        smile_l = blendshapes.get("mouthSmileLeft", 0)
        smile_r = blendshapes.get("mouthSmileRight", 0)
        attrs["smiling"] = (smile_l + smile_r) / 2 > 0.4
        attrs["smile_asymmetry"] = round(abs(smile_l - smile_r), 3)
        attrs["possible_dimples"] = (
            (smile_l > 0.5 or smile_r > 0.5) and cheek_puff < 0.2
        )

        # ── Facial Asymmetry ─────────────────────────────────────────
        pairs = [
            (33, 263), (133, 362), (70, 300), (93, 323), (172, 397),
            (61, 291), (159, 386), (145, 374), (46, 276),
        ]
        asym = 0.0
        for li, ri in pairs:
            asym += abs(abs(lm[li]["x"] - 0.5) - abs(lm[ri]["x"] - 0.5))
        attrs["facial_asymmetry_score"] = round(
            min(asym / len(pairs) / 0.05, 1.0), 3
        )

        attrs["blendshapes"] = blendshapes
        return attrs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(a: dict, b: dict) -> float:
        return math.sqrt(
            (a["x"] - b["x"]) ** 2
            + (a["y"] - b["y"]) ** 2
            + (a.get("z", 0) - b.get("z", 0)) ** 2
        )

    @staticmethod
    def _jaw_angle(lm: list[dict]) -> float:
        chin = lm[152]
        left_jaw, right_jaw = lm[172], lm[397]
        v1 = (left_jaw["x"] - chin["x"], left_jaw["y"] - chin["y"])
        v2 = (right_jaw["x"] - chin["x"], right_jaw["y"] - chin["y"])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        if mag1 * mag2 == 0:
            return 120.0
        cos_a = max(-1, min(1, dot / (mag1 * mag2)))
        return math.acos(cos_a) * (180 / math.pi)
