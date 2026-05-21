"""
InsightFaceAnalyzer — detection + age + gender + recognition embedding.

Model
-----
- Package    : `insightface` (https://github.com/deepinsight/insightface)
- Bundle     : buffalo_l (ResNet50@WebFace600K backbone, ONNX)
- Components : SCRFD-10GF detector, ArcFace 512-d recognition,
               2d106 + 3d68 landmark regressors, age + gender heads
- Size       : ~280 MB (ONNX, mixed FP16/FP32)
- License    : weights research-only; code Apache 2.0
- Source     : https://github.com/deepinsight/insightface/tree/master/python-package

Inputs
------
img_rgb : np.ndarray (H, W, 3) uint8

Outputs (dict)
--------------
face_bbox            : [x1, y1, x2, y2] in pixel coordinates
face_confidence      : SCRFD detection score
face_embedding       : list[float] of length 512 (ArcFace, L2-normalised)
age_estimate         : float years (regression head, not bucketed)
age_range            : string bucket derived from age_estimate for
                       backwards compatibility with the legacy UI
gender               : "male" | "female"
gender_confidence    : 1.0 by default (InsightFace doesn't expose a
                       gender softmax score; the head is argmax-only)
_insight_landmarks_2d : list of (x, y) tuples — 106 points (internal)

Accuracy
--------
- Recognition (ArcFace via buffalo_l): 99.83% LFW, 96.21% IJB-B FAR=1e-4.
- Age / gender heads are widely used but lack a clean published metric.
  In practice age MAE is ~5 years and gender ~94-96%.

Notes
-----
We run the bundle once per image and expose only the highest-confidence
face when multiple are detected — the rest of the pipeline assumes a
single subject.
"""

import os
from typing import Any

import numpy as np

# insightface is a relatively heavy import; deferred so the module can
# at least load when the package isn't installed.
try:
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    HAS_INSIGHTFACE = False


MODEL_NAME = "buffalo_l"

# InsightFace's genderage head is known to overshoot adult ages by
# roughly 5 years in informal testing (no published calibration). We
# subtract a fixed offset to undo this bias; clamp to ≥1 so we never
# emit negative ages for kids. Override at runtime via the
# AGE_OFFSET_YEARS env var if you want to tune for your dataset.
AGE_OFFSET_YEARS = float(os.environ.get("AGE_OFFSET_YEARS", "5"))

# Age buckets used by the legacy UI. We derive these from the regression
# output so existing screens keep working.
AGE_BUCKETS = [
    (0, 3, "0-2"), (3, 10, "3-9"), (10, 20, "10-19"),
    (20, 30, "20-29"), (30, 40, "30-39"), (40, 50, "40-49"),
    (50, 60, "50-59"), (60, 70, "60-69"), (70, 200, "70+"),
]


class InsightFaceAnalyzer:
    def __init__(self):
        self.app = None
        if not HAS_INSIGHTFACE:
            print(
                "[InsightFaceAnalyzer] insightface package not installed; "
                "detection, age, gender, and recognition will degrade to 'unknown'."
            )
            return

        try:
            # Buffalo_L bundle auto-resolves under ~/.insightface/models/.
            # CPUExecutionProvider is the right default for HF Spaces;
            # ctx_id=0 + 'CUDAExecutionProvider' would be the GPU path.
            self.app = FaceAnalysis(
                name=MODEL_NAME,
                providers=["CPUExecutionProvider"],
            )
            # det_size=(640, 640) is the canonical SCRFD input. Smaller
            # speeds inference but loses small faces.
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception as exc:
            print(f"[InsightFaceAnalyzer] Failed to load {MODEL_NAME}: {exc}")
            self.app = None

    def analyze(self, img_rgb: np.ndarray) -> dict[str, Any]:
        if self.app is None:
            return self._empty_result()

        try:
            # InsightFace expects BGR, OpenCV-native order.
            img_bgr = img_rgb[..., ::-1]
            faces = self.app.get(img_bgr)
        except Exception as exc:
            print(f"[InsightFaceAnalyzer] Inference failed: {exc}")
            return self._empty_result()

        if not faces:
            return self._empty_result()

        # If multiple faces, take the highest-confidence one. The rest of
        # the pipeline assumes a single subject.
        face = max(faces, key=lambda f: float(f.det_score))

        # Bounding box is float32 in [x1, y1, x2, y2] image-pixel space.
        bbox = [float(v) for v in face.bbox.tolist()]

        # Recognition embedding: 512-d, L2-normalised by InsightFace.
        # Cast to plain list[float] for clean JSON.
        embedding = (
            [float(v) for v in face.normed_embedding.tolist()]
            if getattr(face, "normed_embedding", None) is not None
            else None
        )

        # Age head is a single float (years). Buffalo_L systematically
        # over-predicts adults by ~5 years; subtract AGE_OFFSET_YEARS
        # to recalibrate. Don't drop below 1 (negative ages would be
        # absurd, and very young children are already on the noisy
        # end of the model's training distribution).
        raw_age = float(getattr(face, "age", 0.0))
        age = max(1.0, raw_age - AGE_OFFSET_YEARS)

        # Gender is exposed as 0 (female) / 1 (male) on Face objects.
        # InsightFace doesn't surface a softmax probability — we report
        # confidence 1.0 to indicate "argmax, no soft signal".
        gender_idx = int(getattr(face, "gender", -1))
        gender = "male" if gender_idx == 1 else "female" if gender_idx == 0 else "unknown"

        return {
            "face_bbox": bbox,
            "face_confidence": round(float(face.det_score), 3),
            "face_embedding": embedding,
            "age_estimate": round(age, 1),
            "age_range": self._bucket_age(age),
            "age_confidence": 1.0,
            "gender": gender,
            "gender_confidence": 1.0,
            # 106 2D landmarks (forehead, jaw, brows, eyes, nose, lips).
            # Underscore-prefixed → stripped from JSON, available to
            # downstream analyzers that want a tighter face crop.
            "_insight_landmarks_2d": (
                [(float(p[0]), float(p[1])) for p in face.landmark_2d_106.tolist()]
                if getattr(face, "landmark_2d_106", None) is not None
                else None
            ),
        }

    @staticmethod
    def _bucket_age(age: float) -> str:
        for lo, hi, label in AGE_BUCKETS:
            if lo <= age < hi:
                return label
        return "unknown"

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "face_bbox": None,
            "face_confidence": 0.0,
            "face_embedding": None,
            "age_estimate": 0.0,
            "age_range": "unknown",
            "age_confidence": 0.0,
            "gender": "unknown",
            "gender_confidence": 0.0,
            "_insight_landmarks_2d": None,
        }
