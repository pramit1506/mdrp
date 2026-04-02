"""
api.py
======
Flask REST API backend — v3

Endpoints
---------
GET  /           → serves index.html
POST /predict    → accepts basic info + lab values → risk JSON
POST /parse-pdf  → accepts PDF upload → extracted lab values JSON

Fixes vs v2
-----------
1. ALL_LAB_FIELDS and FIELD_BOUNDS now include hba1c, ldl, hdl,
   triglycerides, egfr, mcv — these feed clinical_risk.py and the
   health-markers model. Without them the 30% clinical and 20% HM
   blend components degraded to defaults on every prediction.

2. _extract_with_regex rebuilt to use FEATURE_MAP synonym patterns
   from input_mapper.py (the same source-of-truth used everywhere else).
   Old approach matched internal names like "glucose", "hemo" which
   never appear in real lab reports.

3. Tracebacks no longer leaked in JSON responses; logged server-side
   only. Set env MDRP_DEBUG=1 to restore for local development.

4. /predict endpoint passes all new fields through to predict_all().
"""

import logging
import os
import re
import json
import traceback as _tb
from collections import defaultdict

from flask import Flask, request, jsonify, send_from_directory

from predict import predict_all

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

# Set MDRP_DEBUG=1 to include tracebacks in API error responses (dev only)
_DEBUG = os.environ.get("MDRP_DEBUG", "0") == "1"


# ──────────────────────────────────────────────────────────────────────────────
# ALL BLOOD TEST / LAB FIELDS ACCEPTED BY /predict AND /parse-pdf
#
# Order matters for nothing here — it is just the acceptance list.
# The exact feature order used by each model is defined in predict.py.
# ──────────────────────────────────────────────────────────────────────────────
ALL_LAB_FIELDS = [
    # Glucose & metabolic
    "glucose",          # fasting blood glucose (mg/dL)       — PIMA, HM
    "bgr",              # random / post-prandial glucose       — CKD
    "hba1c",            # glycated haemoglobin (%)             — clinical diabetes score, HM
    "insulin",          # fasting insulin (µU/mL)              — PIMA

    # Lipid panel
    "chol",             # total cholesterol (mg/dL)            — heart UCI
    "ldl",              # LDL cholesterol (mg/dL)              — clinical heart score, HM
    "hdl",              # HDL cholesterol (mg/dL)              — clinical heart score, HM
    "triglycerides",    # triglycerides (mg/dL)                — clinical heart score, HM

    # Kidney numeric (UCI CKD)
    "sc",               # serum creatinine (mg/dL)
    "bu",               # blood urea / BUN (mg/dL)
    "sod",              # sodium (mEq/L)
    "pot",              # potassium (mEq/L)
    "sg",               # urine specific gravity
    "al",               # urine albumin (0–5 scale)
    "su",               # urine sugar (0–5 scale)
    "egfr",             # eGFR mL/min/1.73m²                  — clinical kidney score

    # CBC
    "hemo",             # hemoglobin (g/dL)                    — CKD, HM
    "pcv",              # packed cell volume / haematocrit (%) — CKD
    "wbcc",             # WBC count (/cumm)                    — CKD
    "rbcc",             # RBC count (millions/cumm)            — CKD
    "mcv",              # mean corpuscular volume (fL)         — HM anemia signal

    # Kidney categorical (0 / 1)
    "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "appet", "pe", "ane",
]

# ──────────────────────────────────────────────────────────────────────────────
# PHYSIOLOGICAL SANITY BOUNDS
#
# Values outside these ranges are almost certainly mis-extractions.
# Previously missing: hba1c, ldl, hdl, triglycerides, egfr, mcv
# Their absence caused the sanity-check to silently drop them, meaning
# every prediction ran with defaults instead of the patient's real values.
# ──────────────────────────────────────────────────────────────────────────────
FIELD_BOUNDS = {
    # Demographics
    "age":            (1,      120),

    # Glucose & metabolic
    "glucose":        (40,     600),
    "bgr":            (40,     600),
    "hba1c":          (3.0,    15.0),   # non-diabetic ~4–6%, diabetic up to ~14%
    "insulin":        (1,      800),

    # Lipid panel
    "chol":           (80,     500),
    "ldl":            (20,     400),
    "hdl":            (10,     150),
    "triglycerides":  (20,    1500),

    # Kidney numeric
    "sc":             (0.2,    20.0),
    "bu":             (5,      400),
    "sod":            (100,    175),
    "pot":            (1.5,    9.0),
    "sg":             (1.001,  1.040),
    "al":             (0,      5),
    "su":             (0,      5),
    "egfr":           (0,      200),

    # CBC
    "hemo":           (3.0,    22.0),
    "pcv":            (10,     65),
    "wbcc":           (1500,   60000),
    "rbcc":           (1.5,    9.0),
    "mcv":            (50,     130),

    # Categorical 0 / 1
    "rbc":   (0, 1), "pc":    (0, 1), "pcc":   (0, 1), "ba":    (0, 1),
    "htn":   (0, 1), "dm":    (0, 1), "cad":   (0, 1), "appet": (0, 1),
    "pe":    (0, 1), "ane":   (0, 1),

    # BP (extraction only; not forwarded as model features by these names)
    "systolic_bp":    (60,  260),
    "diastolic_bp":   (30,  160),
}


# ──────────────────────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"success": False, "error": "No JSON body received"}), 400

        # ── Compute BMI from height / weight ─────────────────────────────────
        try:
            height_cm = float(data.get("height_cm") or 0)
            weight_kg = float(data.get("weight_kg") or 0)
        except (TypeError, ValueError):
            height_cm = weight_kg = 0.0

        if height_cm > 0 and weight_kg > 0:
            bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
        else:
            bmi = float(data.get("bmi") or 24.5)

        # ── Core demographics ─────────────────────────────────────────────────
        patient_data = {
            "age":           _safe_float(data.get("age"),            45.0),
            "sex":           int(float(data.get("sex",               1))),
            "trestbps":      _safe_float(data.get("systolic_bp"),    120.0),
            "bloodpressure": _safe_float(data.get("diastolic_bp"),    80.0),
            "bmi":           bmi,
            "preg":          int(float(data.get("preg") or 0)),
        }

        # Kidney model alias: bp = diastolic
        patient_data["bp"] = patient_data["bloodpressure"]

        # ── Lab / blood test values — include only if explicitly provided ─────
        for field in ALL_LAB_FIELDS:
            raw = data.get(field)
            if raw is None or raw == "":
                continue
            try:
                patient_data[field] = float(raw)
            except (ValueError, TypeError):
                pass   # skip malformed

        # ── Run blended prediction ────────────────────────────────────────────
        results = predict_all(patient_data)

        return jsonify({
            "success":       True,
            "heart":         results["heart"],
            "diabetes":      results["diabetes"],
            "kidney":        results["kidney"],
            "bmi_used":      bmi,
            "scores_detail": results.get("scores_detail", {}),
            "used_defaults": results.get("used_defaults", []),
        })

    except Exception as exc:
        logger.exception("Error in /predict")
        resp = {"success": False, "error": str(exc)}
        if _DEBUG:
            resp["trace"] = _tb.format_exc()
        return jsonify(resp), 500


@app.route("/parse-pdf", methods=["POST"])
def parse_pdf():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files["file"]
    fname = file.filename or ""
    if not fname.lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "Only PDF files are accepted"}), 400

    try:
        import pdfplumber
    except ImportError:
        return jsonify({"success": False,
                        "error": "pdfplumber is not installed. Run: pip install pdfplumber"}), 500

    try:
        pdf_bytes = file.read()
        extracted, method = None, "regex"

        # ── Tier 1: Claude API ────────────────────────────────────────────────
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            extracted = _extract_with_claude(pdf_bytes, api_key)
            if extracted is not None:
                method = "claude_api"

        # ── Tier 2: Regex fallback ────────────────────────────────────────────
        if extracted is None:
            import io
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            extracted = _extract_with_regex(text)
            method = "regex"

        # ── Sanity-check all extracted values ─────────────────────────────────
        cleaned = _sanity_check(extracted)

        return jsonify({
            "success":    True,
            "extracted":  cleaned,
            "count":      len(cleaned),
            "all_fields": list(cleaned.keys()),
            "method":     method,
        })

    except Exception as exc:
        logger.exception("Error in /parse-pdf")
        resp = {"success": False, "error": f"PDF parse failed: {exc}"}
        if _DEBUG:
            resp["trace"] = _tb.format_exc()
        return jsonify(resp), 500


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ──────────────────────────────────────────────────────────────────────────────
# TIER 1 — CLAUDE API EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────────

_CLAUDE_PROMPT = """You are a specialist medical lab report parser.
Carefully read every line of this report and extract ONLY the requested numeric
and categorical values into a JSON object.

CRITICAL RULES:
1. Do NOT extract Hemoglobin ('hemo') from "HbA1c" or "Glycated Haemoglobin".
   They are completely different tests.
2. Do NOT extract Fasting Glucose from "Estimated Average Glucose (eAG)".
3. For blood urea ('bu'), prefer "Urea" over "BUN" if both are present.
4. Do NOT extract Urine Sugar ('su') from Blood Sugar or LDL/HDL ratio.
5. Do NOT extract Urine Albumin ('al') from Serum Albumin.
6. For eGFR: extract only if explicitly reported; do not calculate from creatinine.

Return ONLY a valid JSON object with these exact keys where values are present:

{
  "age":          <patient age in years, integer>,
  "systolic_bp":  <systolic blood pressure mmHg, number>,
  "diastolic_bp": <diastolic blood pressure mmHg, number>,

  "glucose":      <fasting blood glucose / FBS in mg/dL, number>,
  "bgr":          <random / post-prandial blood sugar / PPBS in mg/dL, number>,
  "hba1c":        <HbA1c / Glycated Haemoglobin in %, number>,
  "insulin":      <fasting insulin in µU/mL, number>,

  "chol":         <total cholesterol in mg/dL, number>,
  "ldl":          <LDL cholesterol in mg/dL, number>,
  "hdl":          <HDL cholesterol in mg/dL, number>,
  "triglycerides":<triglycerides in mg/dL, number>,

  "hemo":         <hemoglobin / Hb / HGB in g/dL — NOT HbA1c, number>,
  "pcv":          <packed cell volume / haematocrit / HCT in %, number>,
  "wbcc":         <WBC / TLC count in cells/cumm (full count, e.g. 8500), number>,
  "rbcc":         <RBC count in millions/cumm, number>,
  "mcv":          <mean corpuscular volume in fL, number>,

  "sc":           <serum creatinine in mg/dL, number>,
  "bu":           <blood urea / urea in mg/dL, number>,
  "sod":          <serum sodium in mEq/L, number>,
  "pot":          <serum potassium in mEq/L, number>,
  "egfr":         <eGFR in mL/min/1.73m² if reported, number>,

  "sg":           <urine specific gravity e.g. 1.020, number>,
  "al":           <urine albumin on 0–5 scale: nil=0,trace=1,+1=2,+2=3,+3=4,+4=5, integer>,
  "su":           <urine sugar on 0–5 scale: nil=0,trace=1,+1=2,+2=3,+3=4,+4=5, integer>,

  "rbc":          <urine RBC: normal=1, abnormal=0, integer>,
  "pc":           <urine pus cells: normal=1, abnormal=0, integer>,
  "pcc":          <pus cell clumps: present=1, not present=0, integer>,
  "ba":           <bacteria: present=1, not present=0, integer>,

  "htn":          <hypertension in history: yes=1, no=0, integer>,
  "dm":           <diabetes mellitus in history: yes=1, no=0, integer>,
  "cad":          <coronary artery disease / IHD: yes=1, no=0, integer>,
  "appet":        <appetite: good=1, poor=0, integer>,
  "pe":           <pedal edema: yes=1, no=0, integer>,
  "ane":          <anemia explicitly mentioned: yes=1, no=0, integer>
}

Return only the JSON object. No markdown. No text before or after.
"""


def _extract_with_claude(pdf_bytes: bytes, api_key: str) -> dict | None:
    """Send the raw PDF to Claude and parse the returned JSON."""
    try:
        import anthropic
        import base64

        client = anthropic.Anthropic(api_key=api_key)
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {"type": "text", "text": _CLAUDE_PROMPT},
                ],
            }],
        )

        raw = response.content[0].text.strip()
        # Strip any stray markdown fences Claude might add
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        # Remove any Google iframe chips that can appear in some environments
        raw = re.sub(r"https?://googleusercontent\.com/\S+", "", raw).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Second attempt with more aggressive cleaning
            clean = re.sub(r"[^\x20-\x7E\n]", "", raw).strip()
            try:
                parsed = json.loads(clean)
            except json.JSONDecodeError:
                logger.warning("Claude returned non-JSON: %s", raw[:200])
                return None

        return parsed if isinstance(parsed, dict) else None

    except Exception:
        logger.exception("Claude extraction failed")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# TIER 2 — REGEX FALLBACK EXTRACTOR
#
# Uses FEATURE_MAP from input_mapper (the single source-of-truth for all
# clinical synonym ↔ canonical mappings) to build search patterns.
# Longest synonyms are tried first to avoid "Na" matching before "Sodium".
# ──────────────────────────────────────────────────────────────────────────────

def _build_regex_patterns() -> dict:
    """
    Build a dict mapping canonical field names → compiled regex patterns.
    Pulls synonym lists from input_mapper.FEATURE_MAP so there is one
    source of truth for all clinical name mappings.

    Returns
    -------
    dict: canonical_field → compiled re.Pattern
    """
    try:
        from input_mapper import FEATURE_MAP
    except ImportError:
        logger.warning("input_mapper not found; regex extractor will use basic patterns")
        FEATURE_MAP = {}

    # Invert the map: canonical → [all synonyms]
    reverse: dict[str, list[str]] = defaultdict(list)
    for synonym, canonical in FEATURE_MAP.items():
        reverse[canonical].append(synonym)

    # Build one alternation pattern per canonical field
    patterns: dict[str, re.Pattern] = {}
    target_fields = set(FIELD_BOUNDS.keys()) - {"systolic_bp", "diastolic_bp"}

    for canonical in target_fields:
        synonyms = reverse.get(canonical, [])
        # Include the canonical name itself as a fallback
        if canonical not in synonyms:
            synonyms.append(canonical)
        # Longest synonym first → avoids prefix false-matches (e.g. "Na" vs "Sodium")
        synonyms = sorted(synonyms, key=len, reverse=True)

        # Build alternation; escape each part for regex safety
        alts = "|".join(re.escape(s) for s in synonyms)
        # Pattern: synonym → optional non-digit chars → capture number
        pattern = re.compile(
            r"(?i)\b(?:" + alts + r")\b"
            r"[^\n\d]{0,40}?"               # separator (colon, space, unit text)
            r"((?:\d{1,6})(?:\.\d{1,4})?)"  # numeric value
        )
        patterns[canonical] = pattern

    return patterns


# Build once at import time (warm path)
_REGEX_PATTERNS: dict | None = None


def _get_regex_patterns() -> dict:
    global _REGEX_PATTERNS
    if _REGEX_PATTERNS is None:
        _REGEX_PATTERNS = _build_regex_patterns()
    return _REGEX_PATTERNS


def _extract_with_regex(text: str) -> dict:
    """
    Fallback regex extraction from raw PDF text.
    Uses clinical synonym patterns — not internal field names.
    """
    extracted: dict = {}
    if not text:
        return extracted

    patterns = _get_regex_patterns()

    for canonical, pattern in patterns.items():
        if canonical not in FIELD_BOUNDS:
            continue
        match = pattern.search(text)
        if match:
            try:
                val = float(match.group(1))
                low, high = FIELD_BOUNDS[canonical]
                if low <= val <= high:
                    extracted[canonical] = val
            except (ValueError, KeyError):
                pass

    # BP — also try "120/80" format
    bp_match = re.search(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", text)
    if bp_match:
        sys_v = float(bp_match.group(1))
        dia_v = float(bp_match.group(2))
        if 60 <= sys_v <= 260 and 30 <= dia_v <= 160:
            extracted["systolic_bp"]  = sys_v
            extracted["diastolic_bp"] = dia_v

    return extracted


# ──────────────────────────────────────────────────────────────────────────────
# SANITY CHECK
# ──────────────────────────────────────────────────────────────────────────────

def _sanity_check(extracted: dict) -> dict:
    """
    Return only fields whose values fall within physiological bounds.
    Fields not present in FIELD_BOUNDS are passed through unchanged
    (e.g. if Claude adds extra context keys we don't object to).
    """
    if not isinstance(extracted, dict):
        return {}

    cleaned: dict = {}
    for field, value in extracted.items():
        if field not in FIELD_BOUNDS:
            # Pass through unknown fields from Claude (they'll be ignored by predict)
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        low, high = FIELD_BOUNDS[field]
        if low <= val <= high:
            cleaned[field] = val

    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)