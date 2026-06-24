"""
SkinAnalytica — question_library.py
20 pre-built researcher questions with routing metadata.
"""

QUESTION_LIBRARY = [
    # ── Data-grounded (queries actual scan/batch results) ──────────
    {
        "id"      : "Q01",
        "category": "data",
        "question": "How many melanoma cases were flagged this week?",
        "description": "Count of CANCER_FLAGGED verdicts with mel prediction",
    },
    {
        "id"      : "Q02",
        "category": "data",
        "question": "What was the false negative rate on the last batch?",
        "description": "FNR from most recent verification report",
    },
    {
        "id"      : "Q03",
        "category": "data",
        "question": "Which anatomical sites have the lowest model confidence?",
        "description": "Confidence breakdown by site from bias reports",
    },
    {
        "id"      : "Q04",
        "category": "data",
        "question": "Show me all annotation conflicts from recent batches",
        "description": "ANNOTATION_CONFLICT cases from verification reports",
    },
    {
        "id"      : "Q05",
        "category": "data",
        "question": "What percentage of images are below confidence threshold?",
        "description": "LOW_CONFIDENCE flag rate from verification reports",
    },
    {
        "id"      : "Q06",
        "category": "data",
        "question": "What is the current ensemble MelAUC?",
        "description": "From ensemble_metrics.json",
    },
    {
        "id"      : "Q07",
        "category": "data",
        "question": "Are there any drift warnings from the last batch?",
        "description": "DriftMonitorAgent last result",
    },
    {
        "id"      : "Q08",
        "category": "data",
        "question": "How many images are in the active learning queue?",
        "description": "Top-N uncertain cases from ActiveLearningAgent",
    },
    {
        "id"      : "Q09",
        "category": "data",
        "question": "Is the model overconfident or underconfident?",
        "description": "CalibrationAgent ECE and temperature recommendation",
    },
    {
        "id"      : "Q10",
        "category": "data",
        "question": "What is the sex disparity in model performance?",
        "description": "From fairness reports — male vs female MelAUC gap",
    },

    # ── Knowledge-grounded (dermatology + ISIC domain) ─────────────
    {
        "id"      : "Q11",
        "category": "knowledge",
        "question": "What is the clinical significance of a Breslow depth > 1mm?",
        "description": "Dermatology knowledge — melanoma staging",
    },
    {
        "id"      : "Q12",
        "category": "knowledge",
        "question": "How does our MelAUC compare to the ISIC 2020 challenge leaderboard?",
        "description": "Benchmarking against published ISIC results",
    },
    {
        "id"      : "Q13",
        "category": "knowledge",
        "question": "What causes high false negatives in acral lentiginous melanoma?",
        "description": "Dermatology knowledge — acral melanoma detection",
    },
    {
        "id"      : "Q14",
        "category": "knowledge",
        "question": "What does TIxAI score mean for clinical trustworthiness?",
        "description": "Explainability metric interpretation",
    },
    {
        "id"      : "Q15",
        "category": "knowledge",
        "question": "Why is temperature scaling used for probability calibration?",
        "description": "ML knowledge — temperature scaling for overconfidence",
    },
    {
        "id"      : "Q16",
        "category": "knowledge",
        "question": "What are the ISIC annotation quality standards?",
        "description": "ISIC protocol knowledge",
    },
    {
        "id"      : "Q17",
        "category": "knowledge",
        "question": "How does Grad-CAM++ differ from standard Grad-CAM?",
        "description": "Explainability method comparison",
    },
    {
        "id"      : "Q18",
        "category": "knowledge",
        "question": "What is the dermoscopic ABCDE rule for melanoma?",
        "description": "Clinical dermatology — ABCDE criteria",
    },
    {
        "id"      : "Q19",
        "category": "knowledge",
        "question": "What are the NCCN guidelines for melanoma treatment?",
        "description": "Clinical guidelines — treatment pathways",
    },
    {
        "id"      : "Q20",
        "category": "knowledge",
        "question": "How should I interpret a pAUC score for melanoma detection?",
        "description": "ISIC 2024 metric interpretation",
    },
]

def get_question_by_id(qid: str) -> dict:
    return next((q for q in QUESTION_LIBRARY if q["id"] == qid), None)

def get_questions_by_category(category: str) -> list:
    return [q for q in QUESTION_LIBRARY if q["category"] == category]

def list_questions() -> str:
    lines = []
    for cat in ["data", "knowledge"]:
        label = "Data-grounded questions" if cat == "data" else "Knowledge-grounded questions"
        lines.append(f"\n{label}:")
        for q in get_questions_by_category(cat):
            lines.append(f"  [{q['id']}] {q['question']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(list_questions())
