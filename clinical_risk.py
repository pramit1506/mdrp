"""
clinical_risk.py  —  Unified Weighted Clinical Risk Scoring (v3)
=================================================================
Computes evidence-based, tiered clinical risk scores for 3 diseases.

Architecture change (v3)
------------------------
Previous system: three separate components (ML 50 % + clinical 30 % + HM 20 %).
New system      : two components — ML ensemble (40 %) + this unified weighted
                  clinical score (60 %).  CBC and urine features removed throughout.

Each disease score is 0–100 and built from features weighted by their
published clinical significance.  Missing values receive medically normal
defaults (not worst-case assumptions).

Scoring references
------------------
Heart   : ACC/AHA 2019 Cardiovascular Risk Guideline · Framingham Risk Score ·
          NCEP ATP III Lipid Guidelines · JNC 8 BP Classification
Diabetes: ADA Standards of Medical Care 2024 · IDF Diabetes Atlas
Kidney  : KDIGO 2022 CKD Clinical Practice Guideline · NKF KDOQI

Feature weights (points) per disease  — sums to 100
-----------------------------------------------------
Heart (100 pts):
  LDL/HDL ratio         25   primary atherogenic lipid marker (ACC/AHA)
  Systolic BP           20   major modifiable CV risk factor
  Age                   15   10-yr risk doubles each decade ≥45 (male) / 55 (female)
  HbA1c                 10   diabetes doubles CV risk (ADA/AHA joint statement)
  Total cholesterol     10   NCEP ATP III independent risk factor
  Triglycerides         10   independent CV risk per ESC 2016
  BMI                   10   obesity × CV risk (WHO / AHA)

Diabetes (100 pts):
  HbA1c                 40   gold-standard glycaemic marker (ADA diagnostic ≥6.5 %)
  Fasting glucose       25   IFG → diabetes progression (ADA: ≥126 mg/dL = DM)
  Post-prandial glucose 15   IGT → diabetes (ADA 2-hr OGTT ≥200 mg/dL = DM)
  BMI                   12   obesity = primary modifiable T2DM risk factor
  Age                    5   T2DM incidence rises with age, especially >45
  Triglycerides          3   proxy for insulin resistance / metabolic syndrome

Kidney (100 pts):
  eGFR                  50   KDIGO primary CKD staging criterion (G1–G5)
  Serum creatinine      25   filtration surrogate; rises as GFR falls
  Blood urea / BUN      10   uremia marker; elevated in CKD stages 3–5
  Systolic BP            8   sustained HTN accelerates CKD progression
  Fasting glucose        7   diabetic nephropathy — leading cause of CKD worldwide
"""

# ---------------------------------------------------------------------------
# Medically-normal reference values (used when a field is absent)
# ---------------------------------------------------------------------------
_NORMAL = {
    "ldl":          90.0,   # mg/dL — optimal <100 (ACC/AHA)
    "hdl":          55.0,   # mg/dL — healthy: >40 men, >50 women (AHA)
    "trestbps":    118.0,   # mmHg  — optimal systolic (<120)
    "hba1c":         5.3,   # %     — normal <5.7 (ADA)
    "chol":        175.0,   # mg/dL — desirable <200 (NCEP)
    "triglycerides":115.0,  # mg/dL — normal <150 (AHA)
    "bmi":          22.5,   # kg/m² — healthy 18.5–24.9 (WHO)
    "age":          45.0,   # years
    "glucose":      95.0,   # mg/dL — normal fasting <100 (ADA)
    "bgr":         115.0,   # mg/dL — normal post-prandial <140 (ADA)
    "egfr":         95.0,   # mL/min — G1 normal (KDIGO ≥90)
    "sc":            0.9,   # mg/dL — normal 0.6–1.2 (male)
    "bu":           14.0,   # mg/dL — normal 7–20
}


def _get(row: dict, key: str) -> float:
    """Retrieve a value from the patient row; fall back to medically normal default."""
    v = row.get(key)
    if v is None or (isinstance(v, float) and v != v):   # None or NaN
        return _NORMAL.get(key, 0.0)
    try:
        return float(v)
    except (TypeError, ValueError):
        return _NORMAL.get(key, 0.0)


# ---------------------------------------------------------------------------
# Heart Disease — 100 pts
# ---------------------------------------------------------------------------
def calculate_heart_risk(row: dict) -> float:
    """
    Heart Disease Unified Weighted Clinical Score (0–100).

    Weights derived from ACC/AHA 2019 risk guidelines and Framingham Risk Score.
    """
    score = 0.0

    # ── 1. LDL/HDL Ratio (25 pts) — primary atherogenic marker ──────────────
    ldl = _get(row, "ldl")
    hdl = max(_get(row, "hdl"), 1.0)     # guard divide-by-zero
    ratio = ldl / hdl
    # Thresholds: <2.0 optimal · 2.0–2.5 near-optimal · 2.5–3.0 borderline ·
    #             3.0–3.5 high · >3.5 very high  (AHA Lipid Guideline)
    if ratio < 2.0:
        score += 0
    elif ratio < 2.5:
        score += 8
    elif ratio < 3.0:
        score += 15
    elif ratio < 3.5:
        score += 20
    else:
        score += 25

    # ── 2. Systolic BP (20 pts) — ACC/AHA 2017 BP classification ─────────────
    sbp = _get(row, "trestbps")
    if sbp < 120:
        score += 0     # Normal
    elif sbp < 130:
        score += 5     # Elevated
    elif sbp < 140:
        score += 10    # Stage 1 HTN
    elif sbp < 160:
        score += 16    # Stage 2 HTN
    else:
        score += 20    # Severe HTN (>160 mmHg)

    # ── 3. Age (15 pts) ───────────────────────────────────────────────────────
    age = _get(row, "age")
    if age < 40:
        score += 0
    elif age < 50:
        score += 5
    elif age < 55:
        score += 8
    elif age < 65:
        score += 12
    else:
        score += 15

    # ── 4. HbA1c (10 pts) — diabetes doubles CV risk (ADA/AHA 2021) ─────────
    hba1c = _get(row, "hba1c")
    if hba1c < 5.7:
        score += 0     # Normal
    elif hba1c < 6.5:
        score += 5     # Prediabetic
    else:
        score += 10    # Diabetic

    # ── 5. Total Cholesterol (10 pts) — NCEP ATP III ──────────────────────────
    chol = _get(row, "chol")
    if chol < 200:
        score += 0     # Desirable
    elif chol < 240:
        score += 5     # Borderline high
    else:
        score += 10    # High

    # ── 6. Triglycerides (10 pts) — ESC 2016 CVD guideline ───────────────────
    trig = _get(row, "triglycerides")
    if trig < 150:
        score += 0     # Normal
    elif trig < 200:
        score += 4     # Borderline high
    elif trig < 500:
        score += 7     # High
    else:
        score += 10    # Very high (pancreatitis risk, extreme CV risk)

    # ── 7. BMI (10 pts) — WHO obesity classification ──────────────────────────
    bmi = _get(row, "bmi")
    if bmi < 25.0:
        score += 0     # Normal weight
    elif bmi < 30.0:
        score += 4     # Overweight
    elif bmi < 35.0:
        score += 7     # Obese Class I
    else:
        score += 10    # Obese Class II+

    return round(min(score, 100.0), 2)


# ---------------------------------------------------------------------------
# Diabetes — 100 pts
# ---------------------------------------------------------------------------
def calculate_diabetes_risk(row: dict) -> float:
    """
    Diabetes Unified Weighted Clinical Score (0–100).

    Weights and thresholds from ADA Standards of Medical Care 2024.
    """
    score = 0.0

    # ── 1. HbA1c (40 pts) — gold-standard glycaemic diagnostic ──────────────
    hba1c = _get(row, "hba1c")
    if hba1c < 5.7:
        score += 0      # Normal (ADA)
    elif hba1c < 6.0:
        score += 12     # Early prediabetes
    elif hba1c < 6.5:
        score += 24     # High-risk prediabetes
    elif hba1c < 7.5:
        score += 34     # Diabetic — well controlled (HbA1c target <7 %)
    elif hba1c < 9.0:
        score += 38     # Diabetic — suboptimal control
    else:
        score += 40     # Diabetic — uncontrolled (>9 %)

    # ── 2. Fasting Glucose (25 pts) — ADA 2024 diagnostic criteria ───────────
    glucose = _get(row, "glucose")
    if glucose < 100:
        score += 0      # Normal
    elif glucose < 110:
        score += 8      # IFG mild (100–109 mg/dL)
    elif glucose < 126:
        score += 16     # IFG high-risk prediabetes (110–125 mg/dL)
    else:
        score += 25     # Diabetic range (≥126 mg/dL, ADA diagnostic)

    # ── 3. Post-prandial / Random Glucose (15 pts) ────────────────────────────
    bgr = _get(row, "bgr")
    if bgr < 140:
        score += 0      # Normal 2-hr PG (<140 mg/dL)
    elif bgr < 200:
        score += 8      # Impaired glucose tolerance (140–199)
    else:
        score += 15     # Diabetic range (≥200 mg/dL, ADA diagnostic)

    # ── 4. BMI (12 pts) — obesity: primary modifiable T2DM risk factor ───────
    bmi = _get(row, "bmi")
    if bmi < 23.0:
        score += 0      # Normal (Asian cut-off per WHO)
    elif bmi < 25.0:
        score += 3      # Near overweight
    elif bmi < 30.0:
        score += 6      # Overweight (~2× T2DM risk)
    elif bmi < 35.0:
        score += 9      # Obese Class I (~7× T2DM risk)
    else:
        score += 12     # Obese Class II+ (>10× T2DM risk)

    # ── 5. Age (5 pts) — T2DM incidence rises sharply after 45 ──────────────
    age = _get(row, "age")
    if age < 35:
        score += 0
    elif age < 45:
        score += 2
    elif age < 60:
        score += 4
    else:
        score += 5

    # ── 6. Triglycerides (3 pts) — insulin resistance / metabolic syndrome ───
    trig = _get(row, "triglycerides")
    if trig < 150:
        score += 0
    elif trig < 250:
        score += 2
    else:
        score += 3

    return round(min(score, 100.0), 2)


# ---------------------------------------------------------------------------
# Kidney Disease — 100 pts
# ---------------------------------------------------------------------------
def calculate_kidney_risk(row: dict) -> float:
    """
    Chronic Kidney Disease Unified Weighted Clinical Score (0–100).

    Weights and thresholds from KDIGO 2022 CKD Clinical Practice Guideline.
    """
    score = 0.0

    # ── 1. eGFR (50 pts) — KDIGO G1–G5 staging (primary CKD criterion) ──────
    egfr = _get(row, "egfr")
    if egfr >= 90:
        score += 0      # G1: Normal or high
    elif egfr >= 60:
        score += 12     # G2: Mildly decreased (60–89)
    elif egfr >= 45:
        score += 28     # G3a: Mildly to moderately decreased (45–59)
    elif egfr >= 30:
        score += 38     # G3b: Moderately to severely decreased (30–44)
    elif egfr >= 15:
        score += 46     # G4: Severely decreased (15–29)
    else:
        score += 50     # G5: Kidney failure (<15)

    # ── 2. Serum Creatinine (25 pts) — primary filtration surrogate ──────────
    sc = _get(row, "sc")
    if sc < 1.0:
        score += 0      # Normal
    elif sc < 1.3:
        score += 8      # Upper normal / mild elevation
    elif sc < 2.0:
        score += 16     # Moderate (CKD Stage 2–3 range)
    elif sc < 4.0:
        score += 22     # Severe (CKD Stage 3–4)
    else:
        score += 25     # Very severe (CKD Stage 4–5, ≥4 mg/dL)

    # ── 3. Blood Urea / BUN (10 pts) — uremia marker ─────────────────────────
    bu = _get(row, "bu")
    if bu < 20:
        score += 0      # Normal (7–20 mg/dL)
    elif bu < 40:
        score += 3      # Mild elevation
    elif bu < 80:
        score += 6      # Moderate uremia
    else:
        score += 10     # Severe uremia (≥80 mg/dL)

    # ── 4. Systolic BP (8 pts) — HTN accelerates CKD progression ─────────────
    sbp = _get(row, "trestbps")
    if sbp < 130:
        score += 0      # Normal / Elevated (KDIGO BP target <120 in CKD)
    elif sbp < 140:
        score += 3      # Stage 1 HTN
    elif sbp < 160:
        score += 6      # Stage 2 HTN
    else:
        score += 8      # Severe HTN

    # ── 5. Fasting Glucose (7 pts) — diabetic nephropathy risk ───────────────
    glucose = _get(row, "glucose")
    if glucose < 100:
        score += 0      # Normal
    elif glucose < 126:
        score += 3      # Prediabetes (elevated CKD risk)
    else:
        score += 7      # Diabetes (leading cause of CKD worldwide)

    return round(min(score, 100.0), 2)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def calculate_all_risks(patient_data: dict) -> dict:
    """
    Compute all three unified weighted clinical risk scores.

    Parameters
    ----------
    patient_data : dict — flat dict of patient values.
                   Missing keys receive medically normal defaults.

    Returns
    -------
    dict with keys:
        diabetes_clinical : float (0–100)
        heart_clinical    : float (0–100)
        kidney_clinical   : float (0–100)
    """
    return {
        "diabetes_clinical": calculate_diabetes_risk(patient_data),
        "heart_clinical":    calculate_heart_risk(patient_data),
        "kidney_clinical":   calculate_kidney_risk(patient_data),
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "age":          35,
        "glucose":      102.9,
        "hba1c":        5.6,
        "bmi":          24.5,
        "trestbps":     120,
        "chol":         139.3,
        "ldl":          75.97,
        "hdl":          37.2,
        "triglycerides": 130.65,
        "sc":           0.41,
        "bu":           14.32,
        "egfr":         159.0,
    }
    risks = calculate_all_risks(sample)
    print("── Unified Weighted Clinical Risk Scores ──")
    for name, score in risks.items():
        print(f"  {name:<25}: {score:6.2f} / 100")
