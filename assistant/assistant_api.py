"""
SkinAnalytica — assistant_api.py
FastAPI routes for the Research Assistant.
Mount this into SA05_api.py or run standalone on port 8002.
"""

import os, json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE    = os.environ.get("SKINANALYTICA_BASE",
          r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica")

app = FastAPI(
    title      = "SkinAnalytica Research Assistant API",
    description= "Conversational AI for ISIC dermoscopy researchers",
    version    = "1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])

# Lazy-load assistant (one per session_id)
_sessions: dict = {}

def get_assistant(session_id: str):
    if session_id not in _sessions:
        from research_assistant import ResearchAssistant
        _sessions[session_id] = ResearchAssistant(
            session_id=session_id, use_claude=True
        )
    return _sessions[session_id]


# ── Pydantic models ───────────────────────────────────────────────
class AskRequest(BaseModel):
    question   : str
    session_id : str = "default"
    question_id: Optional[str] = None

class PrebuiltRequest(BaseModel):
    question_id: str
    session_id : str = "default"


# ── Routes ────────────────────────────────────────────────────────
@app.get("/assistant/health")
async def assistant_health():
    try:
        import anthropic
        claude_available = True
    except ImportError:
        claude_available = False
    return {
        "status"         : "ok",
        "claude_available": claude_available,
        "active_sessions": len(_sessions),
        "timestamp"      : datetime.now().isoformat(),
    }


@app.get("/assistant/questions")
async def list_questions():
    """List all 20 pre-built questions."""
    from question_library import QUESTION_LIBRARY
    return {
        "questions": QUESTION_LIBRARY,
        "count"    : len(QUESTION_LIBRARY),
    }


@app.post("/assistant/ask")
async def ask(req: AskRequest):
    """Ask a free-form or pre-built question."""
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    assistant = get_assistant(req.session_id)
    result    = assistant.ask(req.question, question_id=req.question_id)
    return result


@app.post("/assistant/ask/prebuilt")
async def ask_prebuilt(req: PrebuiltRequest):
    """Ask a pre-built question by ID (Q01-Q20)."""
    assistant = get_assistant(req.session_id)
    result    = assistant.ask_prebuilt(req.question_id)
    if "error" in result:
        raise HTTPException(404, result["error"])
    return result


@app.get("/assistant/history/{session_id}")
async def get_history(session_id: str):
    """Get conversation history for a session."""
    assistant = get_assistant(session_id)
    return {
        "session_id": session_id,
        "history"   : assistant.get_history(),
        "count"     : len(assistant.get_history()),
    }


@app.delete("/assistant/history/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session."""
    if session_id in _sessions:
        _sessions[session_id].clear_history()
    return {"status": "cleared", "session_id": session_id}


@app.get("/assistant/data/{question_id}")
async def get_data(question_id: str):
    """Directly query the data layer for a question ID."""
    from data_layer import query_data, DATA_ROUTES
    if question_id not in DATA_ROUTES:
        raise HTTPException(404, f"No data route for {question_id}. Valid: {list(DATA_ROUTES.keys())}")
    return query_data(question_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("assistant_api:app", host="0.0.0.0", port=8002,
                reload=False, log_level="info")
