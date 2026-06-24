"""
SkinAnalytica — delivery_api.py
FastAPI routes for Phase 4 delivery layer.
Runs on port 8003 or mounts into main SA05 API.
"""

import os, json
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

BASE    = os.environ.get("SKINANALYTICA_BASE",
          r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica")
OUT_DIR = os.path.join(BASE, "outputs")

app = FastAPI(
    title      = "SkinAnalytica Delivery API",
    description= "Email, PDF, and notification delivery for SkinAnalytica",
    version    = "1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


class EmailQARequest(BaseModel):
    session_name      : str = "batch"
    recipient         : str = "researcher@isic.org"
    theme             : str = "microsoft"
    verification_data : Optional[dict] = None

class EmailAlertRequest(BaseModel):
    title    : str
    message  : str
    severity : str = "medium"
    theme    : str = "microsoft"

class PDFReportRequest(BaseModel):
    session_name  : str = "SkinAnalytica Research Report"
    include_bias  : bool = True
    include_drift : bool = True

class ScanReportRequest(BaseModel):
    scan_id: str

class PushNotificationRequest(BaseModel):
    title   : str
    body    : str
    severity: str = "info"
    url     : str = "http://localhost:8001/docs"


# In-memory notification queue (Phase 4 mock)
_notifications: List[dict] = []


@app.get("/delivery/health")
async def delivery_health():
    return {"status":"ok","version":"1.0.0","timestamp":datetime.now().isoformat()}


@app.post("/delivery/email/qa-report")
async def email_qa_report(req: EmailQARequest, bg: BackgroundTasks):
    """Generate and save QA report email (Microsoft or Google style)."""
    from email_delivery import send_qa_report
    data = req.verification_data or {
        "total_images":0,"flags":{},"conflicts":[],"top_uncertain":[]
    }
    def _send():
        send_qa_report(data, req.session_name, req.recipient, req.theme)
    bg.add_task(_send)
    return {"status":"queued","session_name":req.session_name,
            "theme":req.theme,"timestamp":datetime.now().isoformat()}


@app.post("/delivery/email/weekly-digest")
async def email_weekly_digest(bg: BackgroundTasks, theme: str = "google"):
    """Generate weekly digest email from current metrics."""
    from email_delivery import send_weekly_digest
    import glob

    prod    = os.path.join(BASE,"models","production")
    metrics = {}
    mp = os.path.join(prod,"ensemble","ensemble_metrics.json")
    if os.path.exists(mp):
        with open(mp) as f: metrics = json.load(f)

    drift = {}
    df_files = sorted(glob.glob(os.path.join(OUT_DIR,"drift_reports","*.json")),
                      key=os.path.getmtime, reverse=True)
    if df_files:
        with open(df_files[0]) as f: drift = json.load(f)

    def _send():
        send_weekly_digest(metrics, drift=drift, theme=theme)
    bg.add_task(_send)
    return {"status":"queued","theme":theme}


@app.post("/delivery/email/alert")
async def email_alert(req: EmailAlertRequest, bg: BackgroundTasks):
    """Send an alert email."""
    from email_delivery import send_alert
    def _send():
        send_alert(req.title, req.message, req.severity, req.theme)
    bg.add_task(_send)
    return {"status":"queued","title":req.title,"severity":req.severity}


@app.post("/delivery/report/research")
async def generate_research_report(req: PDFReportRequest):
    """Generate full research report (HTML/PDF)."""
    from pdf_report_generator import generate_research_report as gen
    path = gen(req.session_name, req.include_bias, req.include_drift)
    fname = os.path.basename(path)
    return {"status":"generated","file":fname,"path":path,
            "download_url":f"/delivery/report/download/{fname}"}


@app.post("/delivery/report/scan/{scan_id}")
async def generate_scan_report(scan_id: str):
    """Generate single-scan clinical report."""
    from pdf_report_generator import generate_scan_report
    scan_path = os.path.join(OUT_DIR, "scan_results", f"{scan_id}.json")
    if not os.path.exists(scan_path):
        raise HTTPException(404, f"Scan {scan_id} not found")
    with open(scan_path, encoding="utf-8") as f:
        scan_data = json.load(f)
    path  = generate_scan_report(scan_data)
    fname = os.path.basename(path)
    return {"status":"generated","file":fname,"path":path,
            "download_url":f"/delivery/report/download/{fname}"}


@app.get("/delivery/report/download/{filename}")
async def download_report(filename: str):
    """Download a generated report."""
    path = os.path.join(OUT_DIR, "reports", filename)
    if not os.path.exists(path):
        path = os.path.join(OUT_DIR, "emails", filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"File {filename} not found")
    media_type = "application/pdf" if filename.endswith(".pdf") else "text/html"
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/delivery/emails")
async def list_emails():
    """List generated email files."""
    email_dir = os.path.join(OUT_DIR,"emails")
    if not os.path.exists(email_dir):
        return {"emails":[],"count":0}
    files = sorted(
        [f for f in os.listdir(email_dir) if f.endswith(".html")],
        reverse=True
    )
    return {"emails": files, "count": len(files)}


@app.get("/delivery/emails/{filename}", response_class=HTMLResponse)
async def preview_email(filename: str):
    """Preview a generated email in browser."""
    path = os.path.join(OUT_DIR,"emails",filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"Email {filename} not found")
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ── PWA Push Notifications (mock) ────────────────────────────────
@app.post("/delivery/notify")
async def push_notification(req: PushNotificationRequest):
    """Queue a push notification (stored for PWA polling)."""
    notif = {
        "id"       : f"notif_{len(_notifications)+1}",
        "title"    : req.title,
        "body"     : req.body,
        "severity" : req.severity,
        "url"      : req.url,
        "timestamp": datetime.now().isoformat(),
        "read"     : False,
    }
    _notifications.append(notif)
    return {"status":"queued","notification_id":notif["id"]}


@app.get("/delivery/notifications")
async def get_notifications(unread_only: bool = False):
    """Get all queued notifications (PWA polls this)."""
    notifs = _notifications if not unread_only else [
        n for n in _notifications if not n["read"]
    ]
    return {"notifications":list(reversed(notifs[-50:])),"count":len(notifs)}


@app.patch("/delivery/notifications/{notification_id}/read")
async def mark_read(notification_id: str):
    """Mark a notification as read."""
    for n in _notifications:
        if n["id"] == notification_id:
            n["read"] = True
            return {"status":"marked_read","id":notification_id}
    raise HTTPException(404, f"Notification {notification_id} not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("delivery_api:app", host="0.0.0.0", port=8003,
                reload=False, log_level="info")
