"""
clinical_risk.py
================
Computes domain-knowledge-based clinical risk scores for 3 diseases.
These scores are derived from published clinical guidelines and serve as
a strong interpretable prior that complements ML model predictions.

Scoring Breakdown (max 100 points each)
----------------------------------------
Diabetes  : HbA1C (45 pts) + Fasting_Glucose (30 pts) + BMI (15 pts) + Age (10 pts)
Heart     : LDL/HDL Ratio (35 pts) + Systolic_BP (25 pts) + Total_Chol (20 pts) + Age (15 pts) + Triglycerides (5 pts)
Kidney    : eGFR (60 pts) + Serum Creatinine (25 pts) + BUN (15 pts)

References
----------
- ADA Clinical Practice Guidelines 2024 (HbA1C, glucose thresholds)
- ACC/AHA Cardiovascular Risk Guidelines (lipid panel scoring)
- KDIGO CKD Classification (eGFR staging)
- Tanaka et al. 2001 (max heart rate formula, used in heart context)
"""


# ─────────────────────────────────────────────────────────────────────────────
def _cap(value: float, max_val: float) -> float:
    """Clamp a score contribution to range [0, max_val]."""
    return max(0.0, min(float(value), float(max_val)))


# ─────────────────────────────────────────────────────────────────────────────
def calculate_diabetes_risk(row: dict) -> float:
    """
    Diabetes Clinical Risk Score (0–100).

    Inputs (keys from patient_data dict):
        hba1c   : HbA1c % (Glycated Haemoglobin)
        glucose : Fasting Blood Glucose mg/dL
        bmi     : Body Mass Index kg/m²
        age     : Patient age in years

    Scoring logic:
        HbA1C  ≥5.0 → each +0.05% adds 1 pt, capped at 45
        FBS    ≥90  → each +1 mg/dL above 90 adds 0.6 pts, capped at 30
        BMI    ≥23  → each +1 unit above 23 adds 1.5 pts, capped at 15
        Age    ≥30  → each +1 yr above 30 adds 0.3 pts, capped at 10
    """
    hba1c_risk   = _cap((row.get("hba1c",   5.0) - 5.0) * 20.0, 45.0)
    glucose_risk = _cap((row.get("glucose", 90.0) - 90.0) * 0.6,  30.0)
    bmi_risk     = _cap((row.get("bmi",     22.0) - 23.0) * 1.5,  15.0)
    age_risk     = _cap((row.get("age",     30.0) - 30.0) * 0.3,  10.0)
    return round(hba1c_risk + glucose_risk + bmi_risk + age_risk, 2)


# ─────────────────────────────────────────────────────────────────────────────
def calculate_heart_risk(row: dict) -> float:
    """
    Heart Disease Clinical Risk Score (0–100).

    Inputs (keys from patient_data dict):
        ldl          : LDL Cholesterol mg/dL
        hdl          : HDL Cholesterol mg/dL
        trestbps     : Systolic Blood Pressure mmHg
        chol         : Total Cholesterol mg/dL
        age          : Patient age in years
        triglycerides: Triglycerides mg/dL

    Scoring logic:
        LDL/HDL ratio ≥1.5  → each +0.067 adds 1 pt, capped at 35
        SBP       ≥120     → each +2 mmHg adds 1 pt, capped at 25
        Total_Chol≥150     → each +5 mg/dL adds 1 pt, capped at 20
        Age       ≥35      → each +2.5 yr adds 1 pt, capped at 15
        Trig      ≥100     → each +20 mg/dL adds 1 pt, capped at 5
    """
    ldl = float(row.get("ldl", 100.0))
    hdl = float(row.get("hdl", 50.0))
    ratio = ldl / hdl if hdl > 0 else 2.0

    lipid_risk = _cap((ratio - 1.5) * 15.0, 35.0)
    bp_risk    = _cap((float(row.get("trestbps",      120.0)) - 120.0) * 0.5, 25.0)
    chol_risk  = _cap((float(row.get("chol",          150.0)) - 150.0) * 0.2, 20.0)
    age_risk   = _cap((float(row.get("age",            35.0)) -  35.0) * 0.4, 15.0)
    trig_risk  = _cap((float(row.get("triglycerides", 100.0)) - 100.0) * 0.05, 5.0)
    return round(lipid_risk + bp_risk + chol_risk + age_risk + trig_risk, 2)


# ─────────────────────────────────────────────────────────────────────────────
def calculate_kidney_risk(row: dict) -> float:
    """
    Chronic Kidney Disease Clinical Risk Score (0–100).

    Inputs (keys from patient_data dict):
        egfr : estimated Glomerular Filtration Rate mL/min/1.73m²
                (Calculate via CKD-EPI formula before calling this if not provided)
        sc   : Serum Creatinine mg/dL
        bu   : Blood Urea Nitrogen mg/dL

    Scoring logic (aligned with KDIGO CKD staging):
        eGFR: score = (100 - min(eGFR,100)) × 0.8, capped at 60
              → eGFR 90 → score 8 (normal)
              → eGFR 60 → score 32 (mild↓)
              → eGFR 15 → score 68 (capped at 60, severe)
        S.Cr ≥1.0  → each +0.067 mg/dL adds 1 pt, capped at 25
        BUN  ≥15   → each +1.33 mg/dL above 15 adds 1 pt, capped at 15
    """
    egfr = float(row.get("egfr", 100.0))
    egfr_risk  = _cap((100.0 - min(egfr, 100.0)) * 0.8, 60.0)
    creat_risk = _cap((float(row.get("sc", 0.8)) - 1.0) * 15.0, 25.0)
    bun_risk   = _cap((float(row.get("bu", 15.0)) - 15.0) * 0.75, 15.0)
    return round(egfr_risk + creat_risk + bun_risk, 2)


# ─────────────────────────────────────────────────────────────────────────────
def calculate_all_risks(patient_data: dict) -> dict:
    """
    Compute all three clinical risk scores from a patient data dictionary.

    Parameters
    ----------
    patient_data : dict
        Flat dict of patient values (see individual function docstrings).
        Missing keys receive safe physiological defaults.

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


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Values from the provided BLOOD_TEST.pdf
    sample = {
        "age":          45,
        "glucose":      102.9,   # FBS = Impaired Fasting (101–125)
        "hba1c":        5.6,     # Non-diabetic range (4–6 %)
        "bmi":          26.0,
        "trestbps":     120,
        "chol":         139.3,   # Desirable
        "ldl":          75.97,   # Optimal (<100)
        "hdl":          37.2,    # Low (risk factor)
        "triglycerides": 130.65, # Borderline high (150–199)
        "sc":           0.41,    # Normal (0.5–1.3)
        "bu":           14.32,   # Normal (3.3–18.7)
        "egfr":         159.0,   # Normal (>90)
    }

    risks = calculate_all_risks(sample)
    print("── Clinical Risk Scores (from BLOOD_TEST.pdf values) ──")
    for name, score in risks.items():
        print(f"  {name:<25}: {score:6.2f} / 100")
