"""
api.py  —  v4.0
===============
Flask REST API backend.

Endpoints
---------
GET  /           → serves index.html
POST /predict    → accepts basic info + lab values → risk JSON
POST /parse-pdf  → accepts PDF upload → extracted lab values JSON

Changes in v4
-------------
- Replaced deprecated Claude-based PDF extractor with Gemini (google-genai SDK).
- Extended Gemini prompt to explicitly capture Post-Prandial / Random Blood Sugar
  (mapped to internal key 'bgr') — previously missed in extraction.
- Regex fallback patterns also extended with explicit 'ppbs' / 'rbs' terms.
- Environment variable for AI key: GOOGLE_API_KEY (set MDRP_DEBUG=1 for tracebacks).
"""

import logging, os, re, json, traceback as _tb
from collections import defaultdict
from flask import Flask, request, jsonify, send_from_directory
from predict import predict_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")
_DEBUG = os.environ.get("MDRP_DEBUG", "0") == "1"

# ── Accepted lab fields ────────────────────────────────────────────────────────
ALL_LAB_FIELDS = [
    "glucose", "bgr", "hba1c", "insulin",
    "chol", "ldl", "hdl", "triglycerides",
    "sc", "bu", "sod", "pot", "egfr",
    "htn", "dm", "cad", "appet", "pe", "ane",
]

FIELD_BOUNDS = {
    "age":          (1,   120),
    "glucose":      (40,  600),
    "bgr":          (40,  600),   # post-prandial / random blood glucose
    "hba1c":        (3.0, 15.0),
    "insulin":      (1,   800),
    "chol":         (80,  500),
    "ldl":          (20,  400),
    "hdl":          (10,  150),
    "triglycerides":(20,  1500),
    "sc":           (0.2, 20.0),
    "bu":           (5,   400),
    "sod":          (100, 175),
    "pot":          (1.5, 9.0),
    "egfr":         (0,   200),
    "htn":          (0,   1),
    "dm":           (0,   1),
    "cad":          (0,   1),
    "appet":        (0,   1),
    "pe":           (0,   1),
    "ane":          (0,   1),
    "systolic_bp":  (60,  260),
    "diastolic_bp": (30,  160),
}


# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


# ─────────────────────────────────────────────────────────────────────────────
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
            "age":           _safe_float(data.get("age"),          45.0),
            "sex":           int(float(data.get("sex",             1))),
            "trestbps":      _safe_float(data.get("systolic_bp"), 118.0),
            "bloodpressure": _safe_float(data.get("diastolic_bp"), 78.0),
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


# ─────────────────────────────────────────────────────────────────────────────
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

        # ── Try Gemini first if GOOGLE_API_KEY is set ──────────────────────
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if api_key:
            import io
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                raw_text = "\n".join(p.extract_text() or "" for p in pdf.pages)

            extracted = _extract_with_gemini(raw_text, api_key)
            if extracted is not None:
                method = "gemini_ai"

        # ── Regex fallback ─────────────────────────────────────────────────
        if extracted is None:
            import io
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                raw_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            extracted = _extract_with_regex(raw_text)
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


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── Gemini extraction ──────────────────────────────────────────────────────────
_GEMINI_PROMPT = """You are a specialist medical lab report parser.
Read the complete report text below and extract ONLY the listed values.

CRITICAL EXTRACTION RULES:
1. "glucose" = FASTING glucose only (FBS / FPG / Fasting Blood Sugar / Fasting Plasma Glucose).
   Do NOT use Estimated Average Glucose (eAG) for this field.
2. "bgr" = POST-PRANDIAL or RANDOM blood glucose (PPBS / RBS / Post Prandial Blood Sugar /
   Post Meal Glucose / Random Blood Sugar / Random Blood Glucose / 2hr Post Glucose).
   This is DIFFERENT from fasting glucose. If found, always populate this key.
3. "bu" = Blood Urea (prefer "Urea" over "BUN" if both appear).
4. "egfr" = extract ONLY if explicitly stated; never calculate from creatinine yourself.
5. "ldl", "hdl", "chol" must come from the BLOOD/SERUM lipid panel — NOT from urine reports.
6. "hba1c" = HbA1c / Glycated Haemoglobin / A1c — NOT the same as haemoglobin level.
7. For binary clinical flags (htn, dm, cad, pe, ane): use 1 if condition is mentioned as
   present/yes/positive, 0 if absent/no/normal. Use null if not mentioned.
8. "appet" = 1 for good/normal appetite, 0 for poor/reduced. null if not mentioned.
9. All numeric values must be plain numbers — NO units, NO ranges, NO comparison symbols.
10. Return null for any key not found or not clearly readable in the report.

Return ONLY a valid JSON object with exactly these keys:
{
  "age":          <integer or null>,
  "systolic_bp":  <number or null>,
  "diastolic_bp": <number or null>,
  "glucose":      <number or null>,
  "bgr":          <number or null>,
  "hba1c":        <number or null>,
  "insulin":      <number or null>,
  "chol":         <number or null>,
  "ldl":          <number or null>,
  "hdl":          <number or null>,
  "triglycerides":<number or null>,
  "sc":           <number or null>,
  "bu":           <number or null>,
  "sod":          <number or null>,
  "pot":          <number or null>,
  "egfr":         <number or null>,
  "htn":          <0 or 1 or null>,
  "dm":           <0 or 1 or null>,
  "cad":          <0 or 1 or null>,
  "appet":        <0 or 1 or null>,
  "pe":           <0 or 1 or null>,
  "ane":          <0 or 1 or null>
}

No markdown. No explanation. No text before or after the JSON object.

Report text:
"""


def _extract_with_gemini(raw_text: str, api_key: str) -> dict | None:
    """Use Gemini (google-genai SDK) to extract lab values from raw PDF text."""
    try:
        from google import genai as google_genai

        client  = google_genai.Client(api_key=api_key)
        prompt  = _GEMINI_PROMPT + raw_text

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )

        raw = response.text.strip()
        # Strip markdown code fences if the model adds them
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Remove non-printable characters and retry
            clean = re.sub(r"[^\x20-\x7E\n]", "", raw).strip()
            parsed = json.loads(clean)

        if not isinstance(parsed, dict):
            logger.warning("Gemini returned non-dict JSON; falling back to regex.")
            return None

        # Drop null values — let sanity_check handle what's left
        return {k: v for k, v in parsed.items() if v is not None}

    except Exception:
        logger.exception("Gemini extraction failed; falling back to regex.")
        return None


# ── Regex fallback ─────────────────────────────────────────────────────────────
#
# We build patterns from the FEATURE_MAP in input_mapper, but 'bgr' needs
# special attention because its synonyms contain spaces that trip up simple
# word-boundary patterns.  We keep a hand-crafted bgr pattern as a supplement.
#
_BGR_PATTERN = re.compile(
    r"(?i)"
    r"(?:post[\s\-]?prandial[\s\-]?(?:blood[\s\-]?)?(?:sugar|glucose)"
    r"|ppbs|rbs|random[\s\-]?blood[\s\-]?(?:sugar|glucose)"
    r"|post[\s\-]?meal[\s\-]?glucose"
    r"|2[\s\-]?hr[\s\-]?post"
    r"|pp[\s\-]?glucose"
    r")"
    r"[^\n\d]{0,50}?"
    r"((?:\d{1,4})(?:\.\d{1,2})?)"
)

_FBS_PATTERN = re.compile(
    r"(?i)"
    r"(?:fasting[\s\-]?(?:blood[\s\-]?)?(?:sugar|glucose|plasma[\s\-]?glucose)"
    r"|fbs|fpg|fbg|fasting[\s\-]?sugar"
    r")"
    r"[^\n\d]{0,50}?"
    r"((?:\d{1,4})(?:\.\d{1,2})?)"
)


def _build_regex_patterns():
    try:
        from input_mapper import FEATURE_MAP
    except ImportError:
        FEATURE_MAP = {}

    reverse = defaultdict(list)
    for synonym, canonical in FEATURE_MAP.items():
        reverse[canonical].append(synonym)

    patterns = {}
    # Fields where we use the hand-crafted patterns instead (avoid false matches)
    skip_auto = {"bgr", "glucose"}

    target_fields = set(FIELD_BOUNDS.keys()) - {"systolic_bp", "diastolic_bp"} - skip_auto
    for canonical in target_fields:
        synonyms = reverse.get(canonical, [])
        if canonical not in synonyms:
            synonyms.append(canonical)
        synonyms = sorted(synonyms, key=len, reverse=True)
        alts    = "|".join(re.escape(s) for s in synonyms)
        pattern = re.compile(
            r"(?i)\b(?:" + alts + r")\b"
            r"[^\n\d]{0,40}?"
            r"((?:\d{1,6})(?:\.\d{1,4})?)"
        )
        patterns[canonical] = pattern

    return patterns


_REGEX_PATTERNS = None

def _get_regex_patterns():
    global _REGEX_PATTERNS
    if _REGEX_PATTERNS is None:
        _REGEX_PATTERNS = _build_regex_patterns()
    return _REGEX_PATTERNS


def _extract_with_regex(text: str) -> dict:
    extracted = {}
    if not text:
        return extracted

    patterns = _get_regex_patterns()

    # Standard field patterns
    for canonical, pattern in patterns.items():
        if canonical not in FIELD_BOUNDS:
            continue
        match = pattern.search(text)
        if match:
            try:
                val = float(match.group(1))
                lo, hi = FIELD_BOUNDS[canonical]
                if lo <= val <= hi:
                    extracted[canonical] = val
            except (ValueError, KeyError):
                pass

    # Hand-crafted: fasting glucose (must not accidentally match PPBS lines)
    fbs_match = _FBS_PATTERN.search(text)
    if fbs_match:
        try:
            val = float(fbs_match.group(1))
            if FIELD_BOUNDS["glucose"][0] <= val <= FIELD_BOUNDS["glucose"][1]:
                extracted["glucose"] = val
        except ValueError:
            pass

    # Hand-crafted: post-prandial / random glucose → bgr
    bgr_match = _BGR_PATTERN.search(text)
    if bgr_match:
        try:
            val = float(bgr_match.group(1))
            if FIELD_BOUNDS["bgr"][0] <= val <= FIELD_BOUNDS["bgr"][1]:
                extracted["bgr"] = val
        except ValueError:
            pass

    # Blood pressure: "120/80" format
    bp_match = re.search(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", text)
    if bp_match:
        sys_v, dia_v = float(bp_match.group(1)), float(bp_match.group(2))
        if 60 <= sys_v <= 260 and 30 <= dia_v <= 160:
            extracted["systolic_bp"]  = sys_v
            extracted["diastolic_bp"] = dia_v

    return extracted


# ── Sanity check ───────────────────────────────────────────────────────────────
def _sanity_check(extracted: dict) -> dict:
    if not isinstance(extracted, dict):
        return {}
    cleaned = {}
    for field, value in extracted.items():
        if field not in FIELD_BOUNDS:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        lo, hi = FIELD_BOUNDS[field]
        if lo <= val <= hi:
            cleaned[field] = val
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
