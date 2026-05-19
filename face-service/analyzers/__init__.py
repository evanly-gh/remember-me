# face-service analyzers package
#
# Each analyzer in this package exposes a class with:
#   __init__(self)                    — load model, register device
#   analyze(self, img_rgb) -> dict    — run inference, return attribute dict
#
# Analyzers are independent: they don't import from each other. Cross-
# analyzer plumbing (passing SegFormer masks into ColorAnalyzer, etc.)
# is orchestrated entirely in app.py.
