"""
api.py  —  v3
=============
Flask REST API backend.

Endpoints
---------
GET  /           → serves index.html
POST /predict    → accepts basic info + lab values → risk JSON
POST /parse-pdf  → accepts PDF upload → extracted lab values JSON

Changes vs v2
-------------
- ALL_LAB_FIELDS and FIELD_BOUNDS: CBC fields removed (hemo, pcv, wbcc, rbcc,
  mcv) and urine fields removed (sg, al, su, rbc, pc, pcc, ba).
- Claude prompt updated to omit CBC and urine extraction instructions.
- scores_detail now reflects 2-component breakdown (ML + Clinical) only.
- health_condition (multi-class HM output) forwarded to frontend.
"""

import logging
import os
import re
import json
import traceback as _tb
from collections import defaultdict

from flask import Flask, request, jsonify, send_from_directory
from predict import predict_all

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

_DEBUG = os.environ.get("MDRP_DEBUG", "0") == "1"


# ──────────────────────────────────────────────────────────────────────────────
# ACCEPTED LAB FIELDS  (CBC and urine removed in v3)
# ──────────────────────────────────────────────────────────────────────────────
ALL_LAB_FIELDS = [
    # Glucose & metabolic
    "glucose",          # fasting blood glucose (mg/dL)
    "bgr",              # post-prandial / random glucose (mg/dL)
    "hba1c",            # glycated haemoglobin (%)
    "insulin",          # fasting insulin (µU/mL)

    # Lipid panel
    "chol",             # total cholesterol (mg/dL)
    "ldl",              # LDL cholesterol (mg/dL)
    "hdl",              # HDL cholesterol (mg/dL)
    "triglycerides",    # triglycerides (mg/dL)

    # Kidney function
    "sc",               # serum creatinine (mg/dL)
    "bu",               # blood urea / BUN (mg/dL)
    "sod",              # sodium (mEq/L)
    "pot",              # potassium (mEq/L)
    "egfr",             # eGFR mL/min/1.73m²

    # Clinical history flags (0 / 1)
    "htn", "dm", "cad", "appet", "pe", "ane",
]

# ──────────────────────────────────────────────────────────────────────────────
# PHYSIOLOGICAL SANITY BOUNDS  (CBC and urine entries removed)
# ──────────────────────────────────────────────────────────────────────────────
FIELD_BOUNDS = {
    "age":            (1,     120),

    # Glucose & metabolic
    "glucose":        (40,    600),
    "bgr":            (40,    600),
    "hba1c":          (3.0,   15.0),
    "insulin":        (1,     800),

    # Lipid panel
    "chol":           (80,    500),
    "ldl":            (20,    400),
    "hdl":            (10,    150),
    "triglycerides":  (20,   1500),

    # Kidney function
    "sc":             (0.2,   20.0),
    "bu":             (5,     400),
    "sod":            (100,   175),
    "pot":            (1.5,   9.0),
    "egfr":           (0,     200),

    # Clinical history flags
    "htn":   (0, 1), "dm":    (0, 1), "cad":   (0, 1),
    "appet": (0, 1), "pe":    (0, 1), "ane":   (0, 1),

    # BP (extraction only; forwarded as trestbps / bloodpressure)
    "systolic_bp":  (60,  260),
    "diastolic_bp": (30,  160),
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

        try:
            height_cm = float(data.get("height_cm") or 0)
            weight_kg = float(data.get("weight_kg") or 0)
        except (TypeError, ValueError):
            height_cm = weight_kg = 0.0

        bmi = (round(weight_kg / ((height_cm / 100) ** 2), 1)
               if height_cm > 0 and weight_kg > 0
               else float(data.get("bmi") or 22.5))

        patient_data = {
            "age":           _safe_float(data.get("age"),           45.0),
            "sex":           int(float(data.get("sex",              1))),
            "trestbps":      _safe_float(data.get("systolic_bp"),  118.0),
            "bloodpressure": _safe_float(data.get("diastolic_bp"),  78.0),
            "bmi":           bmi,
            "preg":          int(float(data.get("preg") or 0)),
        }
        patient_data["bp"] = patient_data["bloodpressure"]

        for field in ALL_LAB_FIELDS:
            raw = data.get(field)
            if raw is None or raw == "":
                continue
            try:
                patient_data[field] = float(raw)
            except (ValueError, TypeError):
                pass

        results = predict_all(patient_data)

        return jsonify({
            "success":          True,
            "heart":            results["heart"],
            "diabetes":         results["diabetes"],
            "kidney":           results["kidney"],
            "bmi_used":         bmi,
            "scores_detail":    results.get("scores_detail", {}),
            "health_condition": results.get("health_condition", {}),
            "used_defaults":    results.get("used_defaults", []),
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
    if not (file.filename or "").lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "Only PDF files are accepted"}), 400

    try:
        import pdfplumber
    except ImportError:
        return jsonify({"success": False,
                        "error": "pdfplumber not installed. Run: pip install pdfplumber"}), 500

    try:
        pdf_bytes = file.read()
        extracted, method = None, "regex"

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            extracted = _extract_with_claude(pdf_bytes, api_key)
            if extracted is not None:
                method = "claude_api"

        if extracted is None:
            import io
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            extracted = _extract_with_regex(text)
            method = "regex"

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
# TIER 1 — CLAUDE API EXTRACTOR  (CBC and urine fields removed from prompt)
# ──────────────────────────────────────────────────────────────────────────────

_CLAUDE_PROMPT = """You are a specialist medical lab report parser.
Carefully read every line of this report and extract ONLY the requested values.

CRITICAL RULES:
1. Do NOT extract Fasting Glucose from "Estimated Average Glucose (eAG)".
2. For blood urea ('bu'), prefer "Urea" over "BUN" if both are present.
3. For eGFR: extract only if explicitly reported; do not calculate from creatinine.
4. Do NOT extract LDL/HDL/cholesterol from urine reports.
5. HbA1c and Hemoglobin are different tests — DO NOT confuse them.

Return ONLY a valid JSON object with these exact keys where values are present:

{
  "age":           <patient age in years, integer>,
  "systolic_bp":   <systolic blood pressure mmHg, number>,
  "diastolic_bp":  <diastolic blood pressure mmHg, number>,

  "glucose":       <fasting blood glucose / FBS in mg/dL, number>,
  "bgr":           <random / post-prandial blood sugar / PPBS in mg/dL, number>,
  "hba1c":         <HbA1c / Glycated Haemoglobin in %, number>,
  "insulin":       <fasting insulin in µU/mL, number>,

  "chol":          <total cholesterol in mg/dL, number>,
  "ldl":           <LDL cholesterol in mg/dL, number>,
  "hdl":           <HDL cholesterol in mg/dL, number>,
  "triglycerides": <triglycerides in mg/dL, number>,

  "sc":            <serum creatinine in mg/dL, number>,
  "bu":            <blood urea / urea in mg/dL, number>,
  "sod":           <serum sodium in mEq/L, number>,
  "pot":           <serum potassium in mEq/L, number>,
  "egfr":          <eGFR in mL/min/1.73m² if reported, number>,

  "htn":   <hypertension in history: yes=1, no=0, integer>,
  "dm":    <diabetes mellitus in history: yes=1, no=0, integer>,
  "cad":   <coronary artery disease / IHD: yes=1, no=0, integer>,
  "appet": <appetite: good=1, poor=0, integer>,
  "pe":    <pedal edema: yes=1, no=0, integer>,
  "ane":   <anemia explicitly mentioned: yes=1, no=0, integer>
}

Return only the JSON object. No markdown. No text before or after.
"""


def _extract_with_claude(pdf_bytes: bytes, api_key: str) -> dict | None:
    try:
        import anthropic, base64
        client  = anthropic.Anthropic(api_key=api_key)
        pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "document",
                     "source": {"type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64}},
                    {"type": "text", "text": _CLAUDE_PROMPT},
                ],
            }],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        raw = re.sub(r"https?://googleusercontent\.com/\S+", "", raw).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
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
# ──────────────────────────────────────────────────────────────────────────────

def _build_regex_patterns() -> dict:
    try:
        from input_mapper import FEATURE_MAP
    except ImportError:
        logger.warning("input_mapper not found; regex extractor will use basic patterns")
        FEATURE_MAP = {}

    reverse: dict[str, list[str]] = defaultdict(list)
    for synonym, canonical in FEATURE_MAP.items():
        reverse[canonical].append(synonym)

    patterns: dict[str, re.Pattern] = {}
    target_fields = set(FIELD_BOUNDS.keys()) - {"systolic_bp", "diastolic_bp"}

    for canonical in target_fields:
        synonyms = reverse.get(canonical, [])
        if canonical not in synonyms:
            synonyms.append(canonical)
        synonyms = sorted(synonyms, key=len, reverse=True)
        alts     = "|".join(re.escape(s) for s in synonyms)
        pattern  = re.compile(
            r"(?i)\b(?:" + alts + r")\b"
            r"[^\n\d]{0,40}?"
            r"((?:\d{1,6})(?:\.\d{1,4})?)"
        )
        patterns[canonical] = pattern

    return patterns


_REGEX_PATTERNS: dict | None = None


def _get_regex_patterns() -> dict:
    global _REGEX_PATTERNS
    if _REGEX_PATTERNS is None:
        _REGEX_PATTERNS = _build_regex_patterns()
    return _REGEX_PATTERNS


def _extract_with_regex(text: str) -> dict:
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
    if not isinstance(extracted, dict):
        return {}
    cleaned: dict = {}
    for field, value in extracted.items():
        if field not in FIELD_BOUNDS:
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
