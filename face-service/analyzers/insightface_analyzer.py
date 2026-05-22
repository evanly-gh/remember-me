"""
InsightFaceAnalyzer — face detection + ArcFace recognition embedding.

Model
-----
- Package    : `insightface` (https://github.com/deepinsight/insightface)
- Bundle     : buffalo_l (ResNet50@WebFace600K backbone, ONNX)
- Used here  : SCRFD-10GF detector + ArcFace 512-d recognition + 106
               2D landmarks. The bundle ALSO ships a genderage head,
               but we ignore it: it routinely calls 20-year-olds "52"
               and no calibration trick reliably undoes that drift.
               Age comes from FairFace ViT (AgeAnalyzer), gender from
               FairFace ViT (GenderAnalyzer).
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
_insight_landmarks_2d : 106 2D points (internal, stripped from JSON)

Accuracy
--------
- Recognition (ArcFace via buffalo_l): 99.83% LFW, 96.21% IJB-B FAR=1e-4.
- Detection (SCRFD-10GF): >99% recall on WIDER FACE easy / medium.
"""

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


class InsightFaceAnalyzer:
    def __init__(self):
        self.app = None
        if not HAS_INSIGHTFACE:
            print(
                "[InsightFaceAnalyzer] insightface package not installed; "
                "face detection and recognition will be unavailable."
            )
            return

        try:
            # CPUExecutionProvider is the right default for HF Spaces;
            # add 'CUDAExecutionProvider' first for GPU.
            self.app = FaceAnalysis(
                name=MODEL_NAME,
                providers=["CPUExecutionProvider"],
            )
            # det_size=(640, 640) is the canonical SCRFD input.
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
        embedding = (
            [float(v) for v in face.normed_embedding.tolist()]
            if getattr(face, "normed_embedding", None) is not None
            else None
        )

        return {
            "face_bbox": bbox,
            "face_confidence": round(float(face.det_score), 3),
            "face_embedding": embedding,
            # 106 2D landmarks (forehead, jaw, brows, eyes, nose, lips).
            # Underscore-prefixed → stripped from JSON, available to
            # downstream analyzers that want tighter face geometry.
            "_insight_landmarks_2d": (
                [(float(p[0]), float(p[1])) for p in face.landmark_2d_106.tolist()]
                if getattr(face, "landmark_2d_106", None) is not None
                else None
            ),
        }

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "face_bbox": None,
            "face_confidence": 0.0,
            "face_embedding": None,
            "_insight_landmarks_2d": None,
        }
