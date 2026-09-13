"""
router.py — Rule-Based Multi-Disease Triage
=============================================
No ML. No training data. Pure logic.

Checks what clinical features the user provided and determines
which disease modules have enough data to fire.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# DISEASE MODULE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

DISEASE_MODULES = {
    "heart": {
        "display_name": "❤️ Heart Disease",
        "required": ["age", "sex", "chest_pain_type", "resting_bp"],
        "optional": [
            "cholesterol", "fasting_bs", "rest_ecg",
            "max_hr", "exercise_angina", "oldpeak", "st_slope",
        ],
        "min_total": 6,
        "pipeline_type": "hybrid",
    },
    "diabetes": {
        "display_name": "🩸 Diabetes",
        "required": ["glucose", "bmi", "age"],
        "optional": [
            "blood_pressure", "insulin", "skin_thickness",
            "pregnancies", "diabetes_pedigree", "HbA1c_level",
            "hypertension", "heart_disease", "smoking_history",
        ],
        "min_total": 5,
        "pipeline_type": "hybrid",
    },
    "breast_cancer": {
        "display_name": "🎗️ Breast Cancer",
        "required": ["radius_mean", "texture_mean", "perimeter_mean"],
        "optional": [
            "area_mean", "smoothness_mean", "compactness_mean",
            "concavity_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se",
            "area_se", "smoothness_se", "compactness_se",
            "concavity_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst",
            "area_worst", "smoothness_worst", "compactness_worst",
            "concavity_worst", "symmetry_worst", "fractal_dimension_worst",
        ],
        "min_total": 5,
        "pipeline_type": "hybrid",
    },
    "parkinsons": {
        "display_name": "🧠 Parkinson's Disease",
        "required": ["mdvp_fo", "mdvp_jitter", "mdvp_shimmer"],
        "optional": [
            "nhr", "hnr", "rpde", "dfa",
            "spread1", "spread2", "d2", "ppe",
        ],
        "min_total": 6,
        "pipeline_type": "hybrid",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAGE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def triage(patient_inputs: dict) -> dict:
    """
    Determine which disease modules can fire given available patient data.

    Parameters
    ----------
    patient_inputs : dict
        Keys are feature names, values are the patient's values.
        Only keys that are present and non-empty are considered.

    Returns
    -------
    dict with keys:
        "eligible" : list of disease names that CAN run
        "ineligible" : dict mapping disease name → reason it can't run
        "details" : dict mapping disease name → {available, missing_required, feature_count}
    """
    # Filter out empty/None values
    available_features = {
        k for k, v in patient_inputs.items()
        if v is not None and str(v).strip() != ""
    }

    eligible = []
    ineligible = {}
    details = {}

    for disease, spec in DISEASE_MODULES.items():
        all_features = spec["required"] + spec["optional"]

        # Check required features
        missing_required = [f for f in spec["required"] if f not in available_features]
        available_count = sum(1 for f in all_features if f in available_features)

        detail = {
            "display_name": spec["display_name"],
            "pipeline_type": spec["pipeline_type"],
            "available_features": available_count,
            "total_possible": len(all_features),
            "missing_required": missing_required,
            "meets_minimum": available_count >= spec["min_total"],
        }
        details[disease] = detail

        if missing_required:
            ineligible[disease] = (
                f"Missing required features: {', '.join(missing_required)}"
            )
        elif available_count < spec["min_total"]:
            ineligible[disease] = (
                f"Only {available_count}/{spec['min_total']} features available"
            )
        else:
            eligible.append(disease)

    return {
        "eligible": eligible,
        "ineligible": ineligible,
        "details": details,
    }


def get_feature_list(disease: str) -> dict:
    """
    Get the full feature list for a disease module.
    Useful for building the Streamlit input form.
    """
    spec = DISEASE_MODULES[disease]
    return {
        "required": spec["required"],
        "optional": spec["optional"],
        "all": spec["required"] + spec["optional"],
    }


def get_all_features() -> list[str]:
    """
    Get a deduplicated list of ALL features across all diseases.
    Used to build the unified patient input form.
    """
    all_features = set()
    for spec in DISEASE_MODULES.values():
        all_features.update(spec["required"])
        all_features.update(spec["optional"])
    return sorted(all_features)
