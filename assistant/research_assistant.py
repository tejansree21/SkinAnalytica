"""
SkinAnalytica — research_assistant.py
Core conversational AI for ISIC researchers.
Routes questions → data layer OR Claude API knowledge layer.
Maintains conversation history per session.
"""

import os, json, re
from datetime import datetime
from typing import Optional
from question_library import QUESTION_LIBRARY, get_question_by_id
from data_layer import query_data, DATA_ROUTES

BASE    = os.environ.get("SKINANALYTICA_BASE",
          r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica")
OUT_DIR = os.path.join(BASE, "outputs", "assistant_sessions")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Dermatology system prompt ─────────────────────────────────────
SYSTEM_PROMPT = """You are SkinAnalytica Research Assistant, an expert AI assistant 
for ISIC dermoscopy researchers and clinical dermatologists.

You have deep expertise in:
- Dermoscopy image analysis and ISIC dataset standards
- Melanoma detection, ABCDE criteria, Breslow depth, Clark levels
- AI/ML for skin lesion classification (ViT, EfficientNet, ConvNeXt)
- Model evaluation metrics: AUC, pAUC, sensitivity, specificity, ECE
- Explainability: Grad-CAM++, TIxAI, SHAP
- ISIC challenge history and benchmarks (2018, 2019, 2020, 2024)
- NCCN clinical guidelines for melanoma, BCC, AK treatment
- Fairness in medical AI: Fitzpatrick scale, demographic bias
- Temperature scaling, ensemble calibration, active learning

The SkinAnalytica platform uses:
  - EfficientNetV2-S (AUC 0.9715), ViT-Large (AUC 0.9887), ConvNeXt-Large (AUC 0.9883)
  - Ensemble MelAUC: 0.9609, pAUC: 0.9316, Sensitivity: 80.1%, Specificity: 95.5%
  - Training data: ISIC 2018 (10,015) + ISIC 2019 (25,331) + ISIC 2020 (33,126) = 68,472 images
  - 7 classes: mel, nv, bcc, akiec, bkl, df, vasc

When you don't have specific data from the platform, answer based on published 
dermatology literature and clearly state your confidence level:
  [HIGH] = well-established clinical fact
  [MEDIUM] = published evidence with some uncertainty  
  [LOW] = expert opinion, limited evidence

Always be concise, clinically accurate, and note when a researcher should 
consult the actual platform data rather than your general knowledge.
"""

# ── Intent classification ─────────────────────────────────────────
DATA_KEYWORDS = [
    "how many","count","last batch","recent","this week","flagged","conflicts",
    "queue","current","status","rate","percentage","show me","list","our model",
    "our ensemble","our results","our auc","drift","calibration","uncertainty"
]

KNOWLEDGE_KEYWORDS = [
    "what is","explain","why","how does","clinical","nccn","guideline",
    "breslow","clark","abcde","isic challenge","leaderboard","compare",
    "tixai","grad-cam","temperature scaling","fitzpatrick","published",
    "literature","research","study","paper"
]


def classify_intent(question: str) -> str:
    """Classify question as 'data' or 'knowledge'."""
    q = question.lower()
    data_score = sum(1 for kw in DATA_KEYWORDS if kw in q)
    know_score = sum(1 for kw in KNOWLEDGE_KEYWORDS if kw in q)
    return "data" if data_score >= know_score else "knowledge"


def match_question_id(question: str) -> Optional[str]:
    """Try to match free-form question to a library question ID."""
    q = question.lower()
    best_score = 0
    best_id    = None
    for lib_q in QUESTION_LIBRARY:
        lib_words = set(lib_q["question"].lower().split())
        q_words   = set(q.split())
        overlap   = len(lib_words & q_words) / max(len(lib_words), 1)
        if overlap > best_score and overlap > 0.4:
            best_score = overlap
            best_id    = lib_q["id"]
    return best_id


def format_data_response(question: str, data: dict) -> str:
    """Turn raw data dict into a readable response."""
    if "error" in data or "note" in data:
        return data.get("error") or data.get("note", "No data available.")

    # Format based on content
    lines = []
    if "count" in data and "days_back" in data:
        lines.append(f"**{data['count']}** melanoma cases flagged in the last {data['days_back']} days.")
        if data.get("scans"):
            lines.append(f"\nMost recent:")
            for s in data["scans"][:5]:
                lines.append(f"  • {s['scan_id']} — {s['pred_class']} ({s.get('confidence',0):.1%}) — {s['timestamp'][:10]}")
    elif "mel_auc" in data:
        lines.append(f"**Ensemble metrics:**")
        lines.append(f"  MelAUC        : {data.get('mel_auc', data.get('mel_auc')):.4f}")
        lines.append(f"  pAUC (TPR≥80%): {data.get('pauc_80', data.get('pauc_80','N/A'))}")
        lines.append(f"  Sensitivity   : {data.get('sensitivity', 0):.1%}")
        lines.append(f"  Specificity   : {data.get('specificity', 0):.1%}")
        lines.append(f"  Temperature T : {data.get('temperature','N/A')}")
    elif "severity" in data:
        icon = {"OK":"✅","MEDIUM":"🟡","HIGH":"🔴"}.get(data["severity"],"⚠")
        lines.append(f"Drift status: {icon} **{data['severity']}**")
        lines.append(f"  KL divergence : {data.get('kl_div','N/A')}")
        lines.append(f"  Conf mean     : {data.get('conf_mean','N/A')}")
        if data.get("flags"):
            lines.append("\n  Flags:")
            for flag in data["flags"]:
                lines.append(f"    ⚠ {flag}")
    elif "by_sex" in data and data["by_sex"]:
        lines.append("**Sex disparity in MelAUC:**")
        for sex, vals in data["by_sex"].items():
            if isinstance(vals, dict):
                auc = vals.get("mel_auc","N/A")
                n   = vals.get("n","?")
                lines.append(f"  {sex:<10}: MelAUC={auc}  (n={n})")
    elif "by_site" in data:
        lines.append("**Confidence by anatomical site (lowest first):**")
        for site, vals in list(data["by_site"].items())[:6]:
            if isinstance(vals, dict):
                auc = vals.get("mel_auc","N/A")
                n   = vals.get("n","?")
                lines.append(f"  {site:<25}: MelAUC={auc}  (n={n})")
    elif "needs_recal" in data:
        icon = "🔴" if data["needs_recal"] else "✅"
        lines.append(f"Calibration: {icon} {data.get('recommendation','')}")
        lines.append(f"  ECE           : {data.get('ece','N/A')}")
        lines.append(f"  Overconfidence: {data.get('overconfidence','N/A'):+.4f}" if isinstance(data.get('overconfidence'), float) else "")
        lines.append(f"  Current T     : {data.get('current_T','N/A')}")
        lines.append(f"  Recommended T : {data.get('recommended_T','N/A')}")
    else:
        lines.append(json.dumps(data, indent=2, default=str)[:500])
    return "\n".join(lines)


class ResearchAssistant:
    """
    SkinAnalytica Research Assistant.
    Handles data-grounded and knowledge-grounded questions.
    Maintains conversation history per session.
    """

    def __init__(self, session_id: str = None, use_claude: bool = True):
        self.session_id  = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.history     = []
        self.use_claude  = use_claude
        self.session_path= os.path.join(OUT_DIR, f"session_{self.session_id}.json")
        self._load_session()

    def _load_session(self):
        if os.path.exists(self.session_path):
            with open(self.session_path, encoding="utf-8") as f:
                data = json.load(f)
            self.history = data.get("history", [])

    def _save_session(self):
        with open(self.session_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": self.session_id,
                "history"   : self.history,
                "updated"   : datetime.now().isoformat(),
            }, f, indent=2, default=str)

    def _call_claude(self, question: str, context: str = "") -> str:
        """Call Claude API for knowledge-grounded questions."""
        try:
            import anthropic
            client   = anthropic.Anthropic()
            messages = []
            # Include last 6 turns for context
            for turn in self.history[-6:]:
                messages.append({"role": "user",      "content": turn["question"]})
                messages.append({"role": "assistant", "content": turn["answer"]})
            user_content = question
            if context:
                user_content = f"{question}\n\nRelevant platform data:\n{context}"
            messages.append({"role": "user", "content": user_content})
            resp = client.messages.create(
                model      = "claude-sonnet-4-6",
                max_tokens = 800,
                system     = SYSTEM_PROMPT,
                messages   = messages,
            )
            return resp.content[0].text
        except ImportError:
            return self._fallback_knowledge(question)
        except Exception as e:
            return f"[Knowledge layer unavailable: {e}]\n\n{self._fallback_knowledge(question)}"

    def _fallback_knowledge(self, question: str) -> str:
        """Fallback answers for common knowledge questions without Claude API."""
        q = question.lower()
        if "breslow" in q:
            return ("[HIGH] Breslow depth >1mm indicates intermediate-thickness melanoma requiring "
                    "sentinel lymph node biopsy (SLNB) per NCCN guidelines. Depth correlates directly "
                    "with 5-year survival: <1mm ~95%, 1-2mm ~80%, >4mm ~50%.")
        if "abcde" in q:
            return ("[HIGH] The ABCDE rule: **A**symmetry, **B**order irregularity, "
                    "**C**olor variation, **D**iameter >6mm, **E**volution/change over time. "
                    "Any positive criterion warrants dermoscopy evaluation.")
        if "pauc" in q or "partial auc" in q:
            return ("[HIGH] pAUC (partial AUC) at TPR≥80% measures performance specifically in the "
                    "clinically relevant high-sensitivity range. The ISIC 2024 challenge uses pAUC "
                    "as its primary metric. SkinAnalytica's pAUC of 0.9316 means the model maintains "
                    "strong discrimination when constrained to catch ≥80% of melanomas.")
        if "temperature scaling" in q:
            return ("[HIGH] Temperature scaling divides model logits by a scalar T before softmax. "
                    "T>1 softens overconfident predictions; T<1 sharpens underconfident ones. "
                    "SkinAnalytica uses T=0.4095 (sharpening) because label smoothing during training "
                    "caused the ensemble to spread probability too evenly across classes.")
        if "grad-cam++" in q or "gradcam" in q:
            return ("[HIGH] Grad-CAM++ improves on standard Grad-CAM by weighting each activation map "
                    "channel using second-order gradients rather than global average pooling. "
                    "This gives more precise localization of small discriminative regions, "
                    "important for detecting early melanoma features like atypical pigment networks.")
        if "tixai" in q:
            return ("[MEDIUM] TIxAI (Trustworthiness Index for XAI) scores 0-1 how well the model's "
                    "attention (Grad-CAM) aligns with the actual lesion region vs background skin. "
                    "Score >0.7 = trustworthy (attention on lesion), <0.4 = suspicious (attention on "
                    "background). SkinAnalytica computes TIxAI using a binary lesion mask from dermoscopy.")
        return ("[MEDIUM] I don't have a pre-built answer for this. "
                "For best results, ensure the Claude API key is configured in your environment "
                "(ANTHROPIC_API_KEY). I can answer from published dermatology literature "
                "and ISIC standards when the API is available.")

    def ask(self, question: str, question_id: str = None) -> dict:
        """
        Ask the research assistant a question.
        Returns answer + metadata.
        """
        intent = classify_intent(question)
        qid    = question_id or match_question_id(question)
        answer = ""
        source = ""
        data   = None

        if intent == "data" and qid and qid in DATA_ROUTES:
            # Data-grounded answer
            data   = query_data(qid)
            answer = format_data_response(question, data)
            source = f"platform_data:{qid}"

            # Also get Claude to explain/contextualise if API available
            if self.use_claude and data and "error" not in data and "note" not in data:
                context  = f"Raw data: {json.dumps(data, default=str)[:400]}"
                enhanced = self._call_claude(
                    f"Briefly explain what this data means for an ISIC researcher: {question}",
                    context=context
                )
                answer = f"{answer}\n\n**Interpretation:**\n{enhanced}"

        elif intent == "data" and (not qid or qid not in DATA_ROUTES):
            # Data question but no matching route — try Claude with context
            metrics = query_data("Q06")
            context = f"Platform metrics: {json.dumps(metrics, default=str)[:400]}"
            answer  = self._call_claude(question, context=context)
            source  = "claude_with_context"

        else:
            # Knowledge-grounded answer
            answer = self._call_claude(question)
            source = "claude_knowledge"

        # Build response
        response = {
            "question"   : question,
            "answer"     : answer,
            "intent"     : intent,
            "question_id": qid,
            "source"     : source,
            "timestamp"  : datetime.now().isoformat(),
        }

        # Save to history
        self.history.append({
            "question": question,
            "answer"  : answer[:500],
            "intent"  : intent,
            "timestamp": response["timestamp"],
        })
        self._save_session()

        return response

    def ask_prebuilt(self, question_id: str) -> dict:
        """Ask a pre-built library question by ID."""
        q = get_question_by_id(question_id)
        if not q:
            return {"error": f"Question ID {question_id} not found"}
        return self.ask(q["question"], question_id=question_id)

    def clear_history(self):
        self.history = []
        self._save_session()

    def get_history(self) -> list:
        return self.history


# ── CLI for quick testing ─────────────────────────────────────────
if __name__ == "__main__":
    print("SkinAnalytica Research Assistant")
    print("Type 'list' to see pre-built questions, 'quit' to exit\n")

    from question_library import list_questions
    assistant = ResearchAssistant(use_claude=True)

    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue
            if q.lower() == "quit":
                break
            if q.lower() == "list":
                print(list_questions())
                continue
            if q.upper().startswith("Q") and q[1:].isdigit():
                result = assistant.ask_prebuilt(q.upper())
            else:
                result = assistant.ask(q)
            print(f"\nAssistant [{result['intent']}]:\n{result['answer']}\n")
        except KeyboardInterrupt:
            break
