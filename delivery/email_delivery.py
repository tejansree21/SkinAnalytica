"""
SkinAnalytica — email_delivery.py
Email delivery mock — generates HTML emails for Microsoft + Google Mail.
Saves to outputs/emails/ as .html files ready to preview or send via SMTP.
"""

import os, json
from datetime import datetime
from pathlib import Path

BASE    = os.environ.get("SKINANALYTICA_BASE",
          r"C:\Users\tejan\OneDrive\Desktop\drive\SkinAnalytica")
OUT_DIR = os.path.join(BASE, "outputs", "emails")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Email templates ───────────────────────────────────────────────

def _base_html(title: str, body: str, theme: str = "microsoft") -> str:
    """Base HTML email template — Microsoft or Google style."""
    if theme == "microsoft":
        primary = "#0078d4"
        logo    = "SkinAnalytica · Microsoft 365"
        footer  = "Sent via Microsoft Outlook · SkinAnalytica Research Platform"
    else:
        primary = "#1a73e8"
        logo    = "SkinAnalytica · Google Workspace"
        footer  = "Sent via Gmail · SkinAnalytica Research Platform"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  body {{ margin:0; padding:0; font-family: -apple-system, 'Segoe UI', Arial, sans-serif;
          background:#f3f2f1; color:#323130; }}
  .wrapper {{ max-width:600px; margin:24px auto; background:#fff;
              border-radius:4px; overflow:hidden;
              box-shadow:0 2px 8px rgba(0,0,0,0.12); }}
  .header {{ background:{primary}; padding:24px 32px; }}
  .header h1 {{ margin:0; color:#fff; font-size:20px; font-weight:600; }}
  .header p  {{ margin:4px 0 0; color:rgba(255,255,255,0.85); font-size:13px; }}
  .body {{ padding:32px; }}
  .metric-row {{ display:flex; gap:16px; margin:20px 0; }}
  .metric {{ flex:1; background:#f3f2f1; border-radius:4px; padding:16px;
             text-align:center; border-left:4px solid {primary}; }}
  .metric .val {{ font-size:24px; font-weight:700; color:{primary}; }}
  .metric .lbl {{ font-size:12px; color:#605e5c; margin-top:4px; }}
  .section {{ margin:24px 0; }}
  .section h2 {{ font-size:16px; font-weight:600; color:#323130;
                 border-bottom:2px solid {primary}; padding-bottom:8px; }}
  .flag {{ padding:12px 16px; border-radius:4px; margin:8px 0;
           font-size:13px; border-left:4px solid; }}
  .flag-ok     {{ background:#dff6dd; border-color:#107c10; color:#107c10; }}
  .flag-warn   {{ background:#fff4ce; border-color:#d83b01; color:#d83b01; }}
  .flag-danger {{ background:#fde7e9; border-color:#a4262c; color:#a4262c; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; margin:12px 0; }}
  th {{ background:{primary}; color:#fff; padding:10px 12px; text-align:left; }}
  td {{ padding:10px 12px; border-bottom:1px solid #edebe9; }}
  tr:nth-child(even) td {{ background:#f8f8f8; }}
  .btn {{ display:inline-block; background:{primary}; color:#fff;
          padding:12px 24px; border-radius:4px; text-decoration:none;
          font-weight:600; font-size:14px; margin-top:16px; }}
  .footer {{ background:#f3f2f1; padding:16px 32px; font-size:11px;
             color:#605e5c; text-align:center; }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <h1>🔬 {title}</h1>
    <p>{logo} · {datetime.now().strftime("%d %b %Y %H:%M")}</p>
  </div>
  <div class="body">{body}</div>
  <div class="footer">{footer} · Not a medical device · For research use only</div>
</div>
</body>
</html>"""


def send_qa_report(verification_data: dict, session_name: str = "batch",
                   recipient: str = "researcher@isic.org",
                   theme: str = "microsoft") -> str:
    """Generate QA report email from verification agent output."""
    flags  = verification_data.get("flags", {})
    total  = verification_data.get("total_images", 0)
    confl  = flags.get("ANNOTATION_CONFLICT", 0)
    ambig  = flags.get("AMBIGUOUS", 0)
    low_c  = flags.get("LOW_CONFIDENCE", 0)
    ok     = flags.get("OK", 0)

    # Metrics row
    metrics_html = f"""
    <div class="metric-row">
      <div class="metric"><div class="val">{total}</div><div class="lbl">Images Analyzed</div></div>
      <div class="metric"><div class="val">{confl}</div><div class="lbl">Annotation Conflicts</div></div>
      <div class="metric"><div class="val">{ambig}</div><div class="lbl">Ambiguous Cases</div></div>
      <div class="metric"><div class="val">{low_c}</div><div class="lbl">Low Confidence</div></div>
    </div>"""

    # Flags
    flag_html = "<div class='section'><h2>QA Flags</h2>"
    if confl > 0:
        flag_html += f"<div class='flag flag-danger'>⚠ {confl} annotation conflict(s) detected — AI verdict disagrees with human label at high confidence. Review required.</div>"
    if ambig > 0:
        flag_html += f"<div class='flag flag-warn'>⚠ {ambig} ambiguous case(s) — confidence gap &lt;2.0. Consider re-annotation.</div>"
    if low_c > 0:
        flag_html += f"<div class='flag flag-warn'>⚠ {low_c} low-confidence prediction(s) below 65% threshold.</div>"
    if ok > 0 and confl == 0:
        flag_html += f"<div class='flag flag-ok'>✅ {ok} images passed QA with no issues.</div>"
    flag_html += "</div>"

    # Top conflicts table
    conflicts     = verification_data.get("conflicts", [])
    conflict_rows = "".join(
        f"<tr><td>{c.get('image_id','?')}</td><td>{c.get('pred_class','?')}</td>"
        f"<td>{c.get('human_label','?')}</td>"
        f"<td>{c.get('confidence_score','?')}/10</td></tr>"
        for c in conflicts[:10]
    )
    conflict_table = ""
    if conflict_rows:
        conflict_table = f"""
        <div class='section'><h2>Annotation Conflicts (top {min(10,len(conflicts))})</h2>
        <table><tr><th>Image ID</th><th>AI Prediction</th>
        <th>Human Label</th><th>Confidence</th></tr>
        {conflict_rows}</table></div>"""

    # Active learning queue
    al_info = verification_data.get("top_uncertain", [])[:5]
    al_rows = "".join(
        f"<tr><td>{r.get('image_id','?')}</td><td>{r.get('pred_class','?')}</td>"
        f"<td>{r.get('confidence_score','?')}/10</td></tr>"
        for r in al_info
    )
    al_table = ""
    if al_rows:
        al_table = f"""
        <div class='section'><h2>Active Learning Queue (top 5 most uncertain)</h2>
        <table><tr><th>Image ID</th><th>Predicted Class</th><th>Confidence</th></tr>
        {al_rows}</table></div>"""

    body = f"""
    <p>Hello,</p>
    <p>SkinAnalytica has completed QA analysis on batch <strong>{session_name}</strong>.
    Here is your automated verification report.</p>
    {metrics_html}
    {flag_html}
    {conflict_table}
    {al_table}
    <a href="http://localhost:8001/docs" class="btn">View Full Report →</a>
    <p style="margin-top:24px;font-size:12px;color:#605e5c;">
    This report was generated automatically by SkinAnalytica v1.1.0.
    Results are AI-assisted and require clinical review before any diagnostic action.
    </p>"""

    html      = _base_html(f"QA Report: {session_name}", body, theme)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname     = f"qa_report_{session_name}_{timestamp}.html"
    fpath     = os.path.join(OUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ QA report email saved: {fpath}")
    return fpath


def send_weekly_digest(metrics: dict, drift: dict = None,
                       literature: list = None,
                       theme: str = "google") -> str:
    """Generate weekly explainability digest email."""
    mel_auc  = metrics.get("mel_auc", 0)
    pauc     = metrics.get("pauc_80", 0)
    sens     = metrics.get("sensitivity", 0)
    spec     = metrics.get("specificity", 0)

    metrics_html = f"""
    <div class="metric-row">
      <div class="metric"><div class="val">{mel_auc:.4f}</div><div class="lbl">MelAUC</div></div>
      <div class="metric"><div class="val">{pauc:.4f}</div><div class="lbl">pAUC (TPR≥80%)</div></div>
      <div class="metric"><div class="val">{sens:.1%}</div><div class="lbl">Sensitivity</div></div>
      <div class="metric"><div class="val">{spec:.1%}</div><div class="lbl">Specificity</div></div>
    </div>"""

    drift_html = ""
    if drift:
        sev  = drift.get("severity","OK")
        icon = {"OK":"✅","MEDIUM":"🟡","HIGH":"🔴"}.get(sev,"⚠")
        drift_html = f"""
        <div class='section'><h2>Model Drift Status</h2>
        <div class='flag {"flag-ok" if sev=="OK" else "flag-warn" if sev=="MEDIUM" else "flag-danger"}'>
        {icon} Drift severity: <strong>{sev}</strong> · KL divergence: {drift.get('kl_div','N/A')}
        </div></div>"""

    lit_html = ""
    if literature:
        rows = "".join(
            f"<tr><td><a href='{p.get('url','#')}'>{p.get('title','?')[:70]}</a></td>"
            f"<td>{p.get('source','?')}</td><td>{p.get('relevance','?')}</td></tr>"
            for p in literature[:5]
        )
        lit_html = f"""
        <div class='section'><h2>New Literature (this week)</h2>
        <table><tr><th>Title</th><th>Source</th><th>Relevance</th></tr>
        {rows}</table></div>"""

    body = f"""
    <p>Hello,</p>
    <p>Your weekly SkinAnalytica summary for <strong>{datetime.now().strftime('%d %b %Y')}</strong>.</p>
    <div class='section'><h2>Ensemble Performance</h2>{metrics_html}</div>
    {drift_html}
    {lit_html}
    <a href="http://localhost:8002/docs" class="btn">Open Research Assistant →</a>"""

    html  = _base_html("Weekly SkinAnalytica Digest", body, theme)
    fname = f"weekly_digest_{datetime.now().strftime('%Y%m%d')}.html"
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Weekly digest email saved: {fpath}")
    return fpath


def send_alert(title: str, message: str, severity: str = "medium",
               theme: str = "microsoft") -> str:
    """Send a single alert email (drift, conflict, calibration)."""
    color_map = {"low":"flag-ok","medium":"flag-warn","high":"flag-danger"}
    flag_cls  = color_map.get(severity, "flag-warn")
    body = f"""
    <p>Hello,</p>
    <p>SkinAnalytica has detected an alert that requires your attention.</p>
    <div class='flag {flag_cls}'>{message}</div>
    <a href='http://localhost:8002/docs' class='btn'>View Details →</a>"""
    html  = _base_html(f"Alert: {title}", body, theme)
    fname = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    fpath = os.path.join(OUT_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Alert email saved: {fpath}")
    return fpath


if __name__ == "__main__":
    # Demo — generate sample emails
    sample_verification = {
        "total_images": 200, "flags": {
            "OK":150,"AMBIGUOUS":30,"LOW_CONFIDENCE":15,"ANNOTATION_CONFLICT":5
        },
        "conflicts": [
            {"image_id":"ISIC_001","pred_class":"mel","human_label":"nv","confidence_score":8.9},
            {"image_id":"ISIC_002","pred_class":"bcc","human_label":"bkl","confidence_score":7.5},
        ],
        "top_uncertain": [
            {"image_id":"ISIC_003","pred_class":"mel","confidence_score":3.2},
            {"image_id":"ISIC_004","pred_class":"nv","confidence_score":3.8},
        ]
    }
    sample_metrics = {
        "mel_auc":0.9609,"pauc_80":0.9316,"sensitivity":0.801,"specificity":0.955
    }

    qa_path     = send_qa_report(sample_verification, "isic2020_demo", theme="microsoft")
    digest_path = send_weekly_digest(sample_metrics,
                                     drift={"severity":"MEDIUM","kl_div":0.147},
                                     theme="google")
    alert_path  = send_alert("Drift Detected",
                             "Confidence distribution shift detected. KL=0.147. Review recommended.",
                             severity="medium", theme="microsoft")
    print(f"\nGenerated emails:")
    print(f"  {qa_path}")
    print(f"  {digest_path}")
    print(f"  {alert_path}")
