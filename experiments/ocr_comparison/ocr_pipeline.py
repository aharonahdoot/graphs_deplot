"""Configurable single-label OCR pipeline: preprocess transforms + tesseract.

A config (cfg dict) fully specifies how one tick-LABEL crop is turned into a
numeric value. tools/optimize_ocr.py searches this config space against
ground-truth labels; the winning cfg is saved to data/best_ocr_cfg.json and can
be deployed (e.g. as a calibrate OCR backend) without code changes.

The transforms are the ones the misread analysis implicated for the real failure
modes (digit-insertion + decimal/comma confusion at certain resolutions):
upscale/interp, local thresholding, stroke thicken/thin, CLAHE, unsharp, gamma,
white padding -- plus the tesseract page-segmentation mode and OCR engine mode.
"""
import re
import cv2
import numpy as np
import pytesseract
from skimage.filters import threshold_sauvola

_WHITELIST = "0123456789,.-"
_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

# Production baseline: grayscale, 4x cubic, psm 11 (what calibrate._ocr_tokens does).
BASELINE = {"pad": 0, "upscale": 4.0, "interp": "cubic", "gamma": 1.0,
            "clahe": False, "unsharp": 0.0, "thresh": "none", "block": 31,
            "sauvola_k": 0.2, "morph": "none", "morph_k": 2, "psm": 11, "oem": 3}

_INTERP = {"cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4,
           "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}


def preprocess(crop_bgr, cfg):
    """Apply the cfg's transform chain; return a grayscale uint8 (dark text)."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if cfg["pad"]:
        g = cv2.copyMakeBorder(g, *([cfg["pad"]] * 4), cv2.BORDER_CONSTANT, value=255)
    if cfg["gamma"] != 1.0:
        g = np.clip(((g / 255.0) ** cfg["gamma"]) * 255, 0, 255).astype(np.uint8)
    if cfg["clahe"]:
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    s = cfg["upscale"]
    if s and s != 1.0:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=_INTERP[cfg["interp"]])
    if cfg["unsharp"] > 0:
        blur = cv2.GaussianBlur(g, (0, 0), 1.0)
        g = cv2.addWeighted(g, 1 + cfg["unsharp"], blur, -cfg["unsharp"], 0)
    th = cfg["thresh"]
    if th == "otsu":
        _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif th in ("adaptive_mean", "adaptive_gauss"):
        m = cv2.ADAPTIVE_THRESH_MEAN_C if th == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        b = cfg["block"] | 1
        g = cv2.adaptiveThreshold(g, 255, m, cv2.THRESH_BINARY, max(3, b), 10)
    elif th == "sauvola":
        b = cfg["block"] | 1
        g = (g > threshold_sauvola(g, window_size=max(3, b), k=cfg["sauvola_k"])).astype(np.uint8) * 255
    # stroke morphology in INK space (text is dark): thicken = grow dark = erode
    if cfg["morph"] != "none":
        k = np.ones((cfg["morph_k"], cfg["morph_k"]), np.uint8)
        if cfg["morph"] == "thicken":
            g = cv2.erode(g, k)
        elif cfg["morph"] == "thin":
            g = cv2.dilate(g, k)
        elif cfg["morph"] == "open":
            g = cv2.morphologyEx(g, cv2.MORPH_OPEN, k)
        elif cfg["morph"] == "close":
            g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k)
    return g


def parse_value(text):
    m = _NUM.findall(text.replace(" ", ""))
    if not m:
        return None
    try:
        return float(max(m, key=len).replace(",", ""))
    except ValueError:
        return None


def predict_value(crop_bgr, cfg):
    """Full pipeline: preprocess -> tesseract -> parsed numeric value (or None)."""
    g = preprocess(crop_bgr, cfg)
    txt = pytesseract.image_to_string(
        g, config=f"--psm {cfg['psm']} --oem {cfg['oem']} "
                  f"-c tessedit_char_whitelist={_WHITELIST}")
    return parse_value(txt)
