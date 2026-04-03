_NORMAL = {
    "ldl": 90.0, "hdl": 55.0, "trestbps": 118.0, "hba1c": 5.3,
    "chol": 175.0, "triglycerides": 115.0, "bmi": 22.5, "age": 45.0,
    "glucose": 95.0, "bgr": 115.0, "egfr": 95.0, "sc": 0.9, "bu": 14.0,
}

def _get(row, key):
    v = row.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return _NORMAL.get(key, 0.0)
    try:
        return float(v)
    except (TypeError, ValueError):
        return _NORMAL.get(key, 0.0)

def calculate_heart_risk(row):
    score = 0.0
    ldl = _get(row, "ldl"); hdl = max(_get(row, "hdl"), 1.0)
    ratio = ldl / hdl
    score += 0 if ratio < 2.0 else 8 if ratio < 2.5 else 15 if ratio < 3.0 else 20 if ratio < 3.5 else 25
    sbp = _get(row, "trestbps")
    score += 0 if sbp < 120 else 5 if sbp < 130 else 10 if sbp < 140 else 16 if sbp < 160 else 20
    age = _get(row, "age")
    score += 0 if age < 40 else 5 if age < 50 else 8 if age < 55 else 12 if age < 65 else 15
    hba1c = _get(row, "hba1c")
    score += 0 if hba1c < 5.7 else 5 if hba1c < 6.5 else 10
    chol = _get(row, "chol")
    score += 0 if chol < 200 else 5 if chol < 240 else 10
    trig = _get(row, "triglycerides")
    score += 0 if trig < 150 else 4 if trig < 200 else 7 if trig < 500 else 10
    bmi = _get(row, "bmi")
    score += 0 if bmi < 25 else 4 if bmi < 30 else 7 if bmi < 35 else 10
    return round(min(score, 100.0), 2)

def calculate_diabetes_risk(row):
    score = 0.0
    hba1c = _get(row, "hba1c")
    score += 0 if hba1c < 5.7 else 12 if hba1c < 6.0 else 24 if hba1c < 6.5 else 34 if hba1c < 7.5 else 38 if hba1c < 9.0 else 40
    glucose = _get(row, "glucose")
    score += 0 if glucose < 100 else 8 if glucose < 110 else 16 if glucose < 126 else 25
    bgr = _get(row, "bgr")
    score += 0 if bgr < 140 else 8 if bgr < 200 else 15
    bmi = _get(row, "bmi")
    score += 0 if bmi < 23 else 3 if bmi < 25 else 6 if bmi < 30 else 9 if bmi < 35 else 12
    age = _get(row, "age")
    score += 0 if age < 35 else 2 if age < 45 else 4 if age < 60 else 5
    trig = _get(row, "triglycerides")
    score += 0 if trig < 150 else 2 if trig < 250 else 3
    return round(min(score, 100.0), 2)

def calculate_kidney_risk(row):
    score = 0.0
    egfr = _get(row, "egfr")
    score += 0 if egfr >= 90 else 12 if egfr >= 60 else 28 if egfr >= 45 else 38 if egfr >= 30 else 46 if egfr >= 15 else 50
    sc = _get(row, "sc")
    score += 0 if sc < 1.0 else 8 if sc < 1.3 else 16 if sc < 2.0 else 22 if sc < 4.0 else 25
    bu = _get(row, "bu")
    score += 0 if bu < 20 else 3 if bu < 40 else 6 if bu < 80 else 10
    sbp = _get(row, "trestbps")
    score += 0 if sbp < 130 else 3 if sbp < 140 else 6 if sbp < 160 else 8
    glucose = _get(row, "glucose")
    score += 0 if glucose < 100 else 3 if glucose < 126 else 7
    return round(min(score, 100.0), 2)

def calculate_all_risks(patient_data):
    return {
        "diabetes_clinical": calculate_diabetes_risk(patient_data),
        "heart_clinical":    calculate_heart_risk(patient_data),
        "kidney_clinical":   calculate_kidney_risk(patient_data),
    }
