"""Pluggable OCR backends for A/B testing axis-label recognition.

Every backend has the same signature as calibrate._ocr_tokens:

    backend(crop_bgr, scale) -> list[(text, cx, cy)]

where (cx, cy) is the token centre in the ORIGINAL crop's pixel coordinates, so
calibrate._collect can offset and fit it identically regardless of engine. This
lets us swap recognisers and measure them with the same calibration math and the
same accuracy probe (tests/accuracy_probe.py).

Backends here:
  * "tesseract"        -> the production path (imported from calibrate)
  * "rapidocr"         -> RapidOCR / PP-OCRv5 mobile, ONNXRuntime CPU
  * "rapidocr-coreml"  -> same, but forces the CoreML EP (M4 GPU / Neural Engine)

VLM backends (Surya-OCR-2, LightOnOCR-2) are full-page document models and are
~100-1000x heavier per call than this task needs; they belong in a *fallback
cascade* for the few hard crops, not the hot path. Add one here behind the same
signature when you want to measure it.
"""
import numpy as np


def get_backend(name):
    if name == "tesseract":
        from calibrate import _ocr_tokens
        return _ocr_tokens
    if name == "rapidocr":
        return _make_rapidocr(coreml=False)
    if name == "rapidocr-coreml":
        return _make_rapidocr(coreml=True)
    if name == "pipeline":
        return _make_pipeline()
    if name == "strip-pipeline":
        return _make_strip_pipeline()
    raise ValueError(f"unknown backend: {name}")


def _make_strip_pipeline(cfg_path=None, psm=11):
    """Alternative deployment: apply the winning preprocessing transforms to the
    WHOLE strip and let tesseract do its own layout (psm 11), instead of relying
    on segment_strip. Isolates 'do the tuned transforms help in the production
    strip context?' from segmentation quality. Returns tokens with positions
    mapped back through the pad+upscale geometry."""
    import json, os
    import cv2
    import pytesseract
    from ocr_pipeline import preprocess, _WHITELIST
    here = os.path.dirname(__file__)
    cfg = json.load(open(cfg_path or os.path.join(here, "..", "..", "data", "best_ocr_cfg.json")))
    s, pad = cfg["upscale"], cfg["pad"]

    def tokens(crop, scale=None, axis=None):
        g = preprocess(crop, cfg)              # transforms (pad then upscale then ...)
        d = pytesseract.image_to_data(
            g, config=f"--psm {psm} --oem {cfg['oem']} -c tessedit_char_whitelist={_WHITELIST}",
            output_type=pytesseract.Output.DICT)
        out = []
        for i in range(len(d["text"])):
            if d["text"][i].strip() and int(d["conf"][i]) > 20:
                cx = (d["left"][i] + d["width"][i] / 2) / s - pad   # undo upscale+pad
                cy = (d["top"][i] + d["height"][i] / 2) / s - pad
                out.append((d["text"][i], cx, cy))
        return out

    return tokens


def _make_pipeline(cfg_path=None):
    """Deploy the optimizer's winning config (data/best_ocr_cfg.json) as a
    segment-then-recognize backend: split the axis strip into individual label
    crops (labelkit.segment_strip) and OCR each with the tuned single-label
    pipeline. This is how the per-crop accuracy win is realised end-to-end."""
    import json, os
    here = os.path.dirname(__file__)
    cfg_path = cfg_path or os.path.join(here, "..", "..", "data", "best_ocr_cfg.json")
    cfg = json.load(open(cfg_path))
    import sys
    sys.path.insert(0, here)   # labelkit is a sibling module in this folder
    import labelkit as lk
    from ocr_pipeline import predict_value

    def tokens(crop, scale=None, axis=None):
        if axis is None:                       # infer from strip aspect ratio
            axis = "y" if crop.shape[0] >= crop.shape[1] else "x"
        out = []
        for lbl, cx, cy in lk.segment_strip(crop, axis):
            v = predict_value(lbl, cfg)
            if v is not None:
                out.append((repr(v), cx, cy))
        return out

    return tokens


def _make_rapidocr(coreml=False):
    """Build a RapidOCR token extractor. RapidOCR runs its own detection on the
    strip, so it returns one token per number it finds, with box centres already
    in crop pixels -- no upscaling needed (the rec model normalises height)."""
    if coreml:
        _force_coreml_ep()
    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()

    def tokens(crop, scale=None, axis=None):
        # RapidOCR wants RGB; our crops are BGR. It needs a little resolution to
        # detect tiny labels, so upscale modestly (its rec normalises anyway).
        import cv2
        up = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
        res, _ = engine(rgb)
        out = []
        if not res:
            return out
        for box, text, conf in res:
            if not text.strip() or float(conf) < 0.3:
                continue
            pts = np.asarray(box, float)
            cx, cy = pts[:, 0].mean() / 2.0, pts[:, 1].mean() / 2.0  # undo 2x
            out.append((text, cx, cy))
        return out

    return tokens


def _force_coreml_ep():
    """Monkeypatch RapidOCR's onnxruntime session to prepend the CoreML EP, so
    inference runs on the M4 GPU/Neural Engine instead of CPU. RapidOCR's own
    config only exposes CUDA/DirectML, so we inject CoreML at the EP-list level.
    Falls back silently to CPU per-op for anything CoreML can't run."""
    from rapidocr_onnxruntime.utils import infer_engine as ie

    if getattr(ie.OrtInferSession, "_coreml_patched", False):
        return
    orig = ie.OrtInferSession._get_ep_list

    def patched(self):
        eps = orig(self)
        if "CoreMLExecutionProvider" in self.had_providers:
            eps.insert(0, ("CoreMLExecutionProvider", {}))
        return eps

    ie.OrtInferSession._get_ep_list = patched
    ie.OrtInferSession._coreml_patched = True
