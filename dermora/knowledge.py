"""
Dermora AI - Clinical knowledge base for the ISIC 2018 Task 3 (HAM10000) taxonomy.

Everything in this module is static reference data: lesion definitions, risk
stratification, differential-diagnosis neighbours and management language.
It contains no model logic, so it can be reviewed by a clinician independently
of the network.
"""

from __future__ import annotations

# Canonical class order. This MUST match the order the detector was trained
# with (dataset.yaml -> names: [MEL, NV, BCC, AKIEC, BKL, DF, VASC]).
CLASS_ORDER = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

# Risk tiers drive the UI colour system and the triage sort order.
RISK_TIERS = {
    "CRITICAL": {"rank": 4, "label": "Critical", "color": "#ff3b5c"},
    "HIGH": {"rank": 3, "label": "High", "color": "#ff7a45"},
    "MODERATE": {"rank": 2, "label": "Moderate", "color": "#ffc53d"},
    "LOW": {"rank": 1, "label": "Low", "color": "#3ddc97"},
}

LESIONS = {
    "MEL": {
        "name": "Melanoma",
        "full_name": "Malignant Melanoma",
        "category": "Malignant",
        "risk": "CRITICAL",
        "prevalence": "11.1% of the HAM10000 corpus (1,113 images)",
        "summary": (
            "A malignant tumour arising from melanocytes. It is responsible for the large "
            "majority of skin-cancer deaths despite accounting for a small fraction of cases, "
            "because of its capacity for early metastatic spread."
        ),
        "dermoscopy": [
            "Atypical pigment network with irregular thickened lines",
            "Irregular streaks, pseudopods or radial streaming at the periphery",
            "Blue-white veil overlying raised areas",
            "Multiple colours (>3) and irregular dots or globules",
            "Regression structures / peppering",
        ],
        "differentials": ["NV", "BKL", "BCC"],
        "management": (
            "Urgent dermatology referral for dermoscopic review and excisional biopsy with "
            "narrow margins. Do not shave-biopsy a suspected melanoma: full-thickness sampling "
            "is required for accurate Breslow depth measurement, which drives staging."
        ),
        "urgency": "Refer within 2 weeks (or per local urgent suspected-cancer pathway).",
    },
    "NV": {
        "name": "Melanocytic Nevus",
        "full_name": "Benign Melanocytic Nevus",
        "category": "Benign",
        "risk": "LOW",
        "prevalence": "66.9% of the HAM10000 corpus (6,705 images)",
        "summary": (
            "A benign proliferation of melanocytes - the ordinary mole. Junctional, compound "
            "and intradermal subtypes all fall in this class. It is by far the most common "
            "class in the training data, which is important context for the model's priors."
        ),
        "dermoscopy": [
            "Symmetric, regular pigment network fading gently at the periphery",
            "Uniform globules or homogeneous brown pigmentation",
            "One or two colours only",
            "Stable appearance across serial imaging",
        ],
        "differentials": ["MEL", "BKL", "DF"],
        "management": (
            "No intervention required for a typical nevus. Advise the patient on self-examination "
            "and on the ugly-duckling sign. Photograph and review in 3-6 months if any single "
            "feature is atypical."
        ),
        "urgency": "Routine. Re-image if the lesion changes.",
    },
    "BCC": {
        "name": "Basal Cell Carcinoma",
        "full_name": "Basal Cell Carcinoma",
        "category": "Malignant",
        "risk": "HIGH",
        "prevalence": "5.1% of the HAM10000 corpus (514 images)",
        "summary": (
            "The most common human malignancy. Locally invasive and capable of significant local "
            "destruction if neglected, but metastasis is rare. Prognosis is excellent with timely "
            "treatment, which makes early detection genuinely valuable."
        ),
        "dermoscopy": [
            "Arborising (branched) telangiectatic vessels",
            "Blue-grey ovoid nests and leaf-like areas",
            "Spoke-wheel structures",
            "Shiny white-red structureless zones, frequent ulceration",
            "Absence of a pigment network",
        ],
        "differentials": ["AKIEC", "MEL", "BKL"],
        "management": (
            "Dermatology referral for biopsy confirmation. Treatment depends on subtype and site: "
            "surgical excision, Mohs micrographic surgery for high-risk facial sites, or topical "
            "therapy for superficial variants."
        ),
        "urgency": "Refer within 4-6 weeks; sooner if ulcerated or periocular.",
    },
    "AKIEC": {
        "name": "Actinic Keratosis",
        "full_name": "Actinic Keratosis / Intraepithelial Carcinoma (Bowen disease)",
        "category": "Pre-malignant",
        "risk": "MODERATE",
        "prevalence": "3.3% of the HAM10000 corpus (327 images)",
        "summary": (
            "A spectrum from sun-induced dysplasia to full-thickness intraepidermal carcinoma. "
            "Individual lesions carry a low annual risk of progressing to invasive squamous cell "
            "carcinoma, but that risk is cumulative across a field of many lesions."
        ),
        "dermoscopy": [
            "Erythematous pseudo-network with white-yellow surface scale",
            "Strawberry pattern on the face",
            "Rosettes under polarised light",
            "Dotted or glomerular vessels in Bowen disease",
        ],
        "differentials": ["BCC", "BKL", "MEL"],
        "management": (
            "Dermatology review. Options include cryotherapy for isolated lesions, or field "
            "treatment with topical 5-fluorouracil, imiquimod or photodynamic therapy where there "
            "is widespread actinic damage. Biopsy anything indurated, tender or rapidly growing."
        ),
        "urgency": "Routine referral; expedite if thickened or symptomatic.",
    },
    "BKL": {
        "name": "Benign Keratosis",
        "full_name": "Benign Keratosis-like Lesion (seborrheic keratosis, solar lentigo, lichen planus-like keratosis)",
        "category": "Benign",
        "risk": "LOW",
        "prevalence": "11.0% of the HAM10000 corpus (1,099 images)",
        "summary": (
            "A grouped class of benign keratinocytic lesions. Clinically important mainly as the "
            "great mimic: irritated or regressing seborrheic keratoses are the single most common "
            "cause of a false melanoma alarm, and vice versa."
        ),
        "dermoscopy": [
            "Milia-like cysts and comedo-like openings",
            "Cerebriform 'brain-like' surface with fissures and ridges",
            "Sharply demarcated, stuck-on border",
            "Fingerprint-like structures in solar lentigo",
        ],
        "differentials": ["MEL", "NV", "AKIEC"],
        "management": (
            "Reassurance is usually sufficient. Cryotherapy or curettage if the lesion is "
            "symptomatic or cosmetically troublesome. Biopsy where melanoma cannot be excluded "
            "on dermoscopy - the cost of a biopsy is far lower than the cost of a missed melanoma."
        ),
        "urgency": "Routine.",
    },
    "DF": {
        "name": "Dermatofibroma",
        "full_name": "Dermatofibroma",
        "category": "Benign",
        "risk": "LOW",
        "prevalence": "1.1% of the HAM10000 corpus (115 images)",
        "summary": (
            "A benign fibrous nodule, often following minor trauma or an insect bite, most commonly "
            "on the lower legs of adult women. Only 115 training images, so treat any DF prediction "
            "with corresponding caution."
        ),
        "dermoscopy": [
            "Central white scar-like patch",
            "Peripheral delicate pigment network",
            "Firm to palpation with a positive dimple sign",
        ],
        "differentials": ["NV", "BKL", "MEL"],
        "management": (
            "No treatment required. Excision only if the diagnosis is uncertain, or if the lesion "
            "is painful or catches on clothing."
        ),
        "urgency": "Routine.",
    },
    "VASC": {
        "name": "Vascular Lesion",
        "full_name": "Vascular Lesion (angioma, angiokeratoma, pyogenic granuloma, haemorrhage)",
        "category": "Benign",
        "risk": "LOW",
        "prevalence": "1.4% of the HAM10000 corpus (142 images)",
        "summary": (
            "Benign lesions of vascular origin. The clinically important caveat is that amelanotic "
            "or nodular melanoma can present as a red nodule and mimic a pyogenic granuloma, so a "
            "rapidly growing, bleeding red nodule still warrants histology."
        ),
        "dermoscopy": [
            "Well-demarcated red, purple or blue-black lacunae",
            "Whitish septa between lacunae",
            "No pigment network",
        ],
        "differentials": ["MEL", "BCC", "DF"],
        "management": (
            "Reassurance. Laser or excision for cosmetic reasons. Biopsy a rapidly growing, "
            "friable red nodule to exclude amelanotic melanoma."
        ),
        "urgency": "Routine; expedite a rapidly growing bleeding nodule.",
    },
}

# Classes that should trigger the malignant-pathway banner in the UI.
MALIGNANT = {"MEL", "BCC"}
PREMALIGNANT = {"AKIEC"}


def lesion(code: str) -> dict:
    """Return the knowledge-base record for a class code."""
    return LESIONS[code]


def risk_color(risk: str) -> str:
    return RISK_TIERS.get(risk, RISK_TIERS["LOW"])["color"]


def malignant_probability(probs: dict[str, float]) -> float:
    """Aggregate probability mass sitting on malignant classes."""
    return float(sum(probs.get(c, 0.0) for c in MALIGNANT))


def concerning_probability(probs: dict[str, float]) -> float:
    """Malignant plus pre-malignant mass - the number that should drive triage."""
    return float(sum(probs.get(c, 0.0) for c in (MALIGNANT | PREMALIGNANT)))


def triage_band(concern: float, top_class: str, confidence: float) -> dict:
    """
    Map the aggregate concerning-probability onto a four-band triage decision.

    Deliberately conservative: the cost matrix in skin oncology is asymmetric, so
    the thresholds favour recall of malignant disease over precision.
    """
    if concern >= 0.60 or (top_class in MALIGNANT and confidence >= 0.50):
        return {
            "band": "URGENT",
            "risk": "CRITICAL" if top_class == "MEL" else "HIGH",
            "headline": "Urgent specialist review indicated",
            "detail": (
                "The model places substantial probability mass on a malignant class. "
                "Escalate on the urgent suspected-skin-cancer pathway."
            ),
        }
    if concern >= 0.30:
        return {
            "band": "PRIORITY",
            "risk": "HIGH",
            "headline": "Priority dermatology assessment advised",
            "detail": (
                "Malignant and pre-malignant classes together hold a meaningful share of the "
                "prediction. In-person dermoscopy is warranted before discharge."
            ),
        }
    if concern >= 0.12:
        return {
            "band": "REVIEW",
            "risk": "MODERATE",
            "headline": "Clinician review recommended",
            "detail": (
                "The prediction favours a benign diagnosis but residual concerning probability "
                "remains. Correlate with history, and re-image in 3 months if the lesion is new "
                "or changing."
            ),
        }
    return {
        "band": "ROUTINE",
        "risk": "LOW",
        "headline": "Benign appearance - routine surveillance",
        "detail": (
            "The prediction is dominated by benign classes. Advise self-monitoring and return "
            "if the lesion changes in size, shape, colour or symptom profile."
        ),
    }


def narrative(top_class: str, confidence: float, margin: float, probs: dict[str, float]) -> str:
    """
    Build a clinician-readable paragraph. Templated rather than generated, so the
    wording is auditable and cannot hallucinate.
    """
    rec = LESIONS[top_class]
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    runner_up = ranked[1][0] if len(ranked) > 1 else None

    if confidence >= 0.75 and margin >= 0.35:
        strength = "The classifier is confident and well separated from the alternatives."
    elif confidence >= 0.55:
        strength = "The classifier favours this class, though the margin over the alternatives is moderate."
    elif margin < 0.10:
        strength = (
            "The classifier is effectively undecided: the top two classes are separated by less "
            "than ten percentage points. Treat this as a flag for human review, not an answer."
        )
    else:
        strength = "Confidence is low. This output should be read as a prompt for review rather than a diagnosis."

    parts = [
        f"The leading class is {rec['full_name']} ({top_class}) at {confidence * 100:.1f}% "
        f"posterior probability. {strength}"
    ]
    if runner_up:
        ru = LESIONS[runner_up]
        parts.append(
            f"The principal differential is {ru['name']} ({runner_up}) at "
            f"{ranked[1][1] * 100:.1f}%."
        )
    parts.append(rec["management"])
    return " ".join(parts)
