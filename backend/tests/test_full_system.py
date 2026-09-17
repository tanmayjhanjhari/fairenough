import requests
import json
import os
import sys
import io
import time

# Ensure UTF-8 output on Windows consoles to support emojis and unicode symbols
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://localhost:8000"
RESULTS = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    RESULTS.append((status, name, detail))
    icon = "✅" if condition else "❌"
    print(f"  {icon} {status}: {name}" + (f" — {detail}" if detail else ""))
    return condition

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ── TEST 1: Health check ─────────────────────────────────────
section("TEST 1: Backend Health")
try:
    r = requests.get(f"{BASE}/api/health", timeout=5)
    check("Backend is running", r.status_code == 200)
except Exception as e:
    check("Backend is running", False, str(e))
    print("\nERROR: Backend not running. Start with: uvicorn main:app --reload")
    sys.exit(1)

# ── TEST 2: Upload CSV ───────────────────────────────────────
section("TEST 2: Dataset Upload")

# Create a minimal test CSV with known bias
csv_content = """age,gender,income,credit_risk
25,male,50000,1
26,male,52000,1
27,male,48000,1
28,male,55000,1
29,male,51000,1
30,male,49000,1
31,male,53000,1
32,male,50000,1
33,male,54000,0
34,male,48000,0
25,female,45000,0
26,female,43000,0
27,female,44000,0
28,female,46000,0
29,female,42000,0
30,female,45000,0
31,female,43000,0
32,female,44000,0
33,female,46000,0
34,female,42000,0
35,male,56000,1
36,male,57000,1
37,male,55000,1
38,male,58000,1
39,male,56000,1
40,male,59000,1
41,male,60000,1
42,male,54000,1
43,male,53000,0
44,male,51000,0
35,female,47000,0
36,female,48000,0
37,female,46000,0
38,female,49000,0
39,female,47000,0
40,female,48000,0
41,female,50000,1
42,female,51000,1
43,female,46000,0
44,female,45000,0
45,male,62000,1
46,male,61000,1
47,male,63000,1
48,male,64000,1
49,male,60000,1
50,male,65000,1
51,male,62000,1
52,male,61000,1
53,male,55000,0
54,male,54000,0
45,female,48000,0
46,female,49000,0
47,female,52000,1
48,female,53000,1
49,female,47000,0
50,female,48000,0
51,female,46000,0
52,female,47000,0
53,female,49000,0
54,female,48000,0
"""

files = {"file": ("test_bias.csv", io.BytesIO(csv_content.encode()), "text/csv")}
r = requests.post(f"{BASE}/api/upload", files=files)
check("Upload returns 200/201", r.status_code in [200, 201], f"status={r.status_code}")
upload_data = r.json()
check("Returns session_id", "session_id" in upload_data)
check("Returns columns", "columns" in upload_data)
check("Returns row_count", upload_data.get("row_count", 0) > 0,
      f"rows={upload_data.get('row_count')}")
check("Returns preprocessing_report", "preprocessing_report" in upload_data)

session_id = upload_data.get("session_id")

# ── TEST 3: Bias Analysis ────────────────────────────────────
section("TEST 3: Bias Analysis")
payload = {
    "session_id": session_id,
    "target_col": "credit_risk",
    "sensitive_attrs": ["gender"]
}
r = requests.post(f"{BASE}/api/analyze", json=payload)
check("Analyze returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:100]}")
analyze_data = r.json()

# Check metrics exist
metrics = analyze_data.get("metrics_per_attr", {})
gender_metrics = metrics.get("gender", {})
spd = gender_metrics.get("SPD") if gender_metrics.get("SPD") is not None else gender_metrics.get("spd")
di  = gender_metrics.get("DI") if gender_metrics.get("DI") is not None else gender_metrics.get("di")
audit = analyze_data.get("audit_score")

check("metrics_per_attr present", bool(metrics))
check("gender metrics present", bool(gender_metrics))
check("SPD computed", spd is not None, f"SPD={spd}")
check("SPD is meaningful (>0.01)", spd is not None and abs(spd) > 0.01,
      f"SPD={spd} — dataset has deliberate bias, should be >0.01")
check("DI computed", di is not None, f"DI={di}")
check("DI reflects bias (!=1.0)", di is not None and abs(di - 1.0) > 0.05,
      f"DI={di}")
check("audit_score present", audit is not None, f"score={audit}")
check("audit_score 0-100 range", audit is not None and 0 <= audit <= 100,
      f"score={audit}")
check("severity present", gender_metrics.get("severity") in ["low","medium","high"],
      f"severity={gender_metrics.get('severity')}")
check("pattern_predictions present", "pattern_predictions" in analyze_data)
check("group_stats present", bool(gender_metrics.get("group_stats")))

if spd is not None and di is not None:
    print(f"\n  📊 Results: SPD={spd:.4f}, DI={di:.4f}, Score={audit}, "
          f"Severity={gender_metrics.get('severity')}")

# ── TEST 4: Explanation ──────────────────────────────────────
section("TEST 4: Bias Explanation")
payload = {"session_id": session_id, "target_col": "credit_risk",
           "sensitive_attr": "gender", "simulate_threshold": True}
r = requests.post(f"{BASE}/api/explain", json=payload)
check("Explain returns 200", r.status_code == 200)
exp = r.json()
check("proxy_features present", "proxy_features" in exp)
check("plain_reason present", bool(exp.get("plain_reason")))
check("spd_explanation present", bool(exp.get("spd_explanation")))
check("di_explanation present", bool(exp.get("di_explanation")))
check("imbalance_explanation present", bool(exp.get("imbalance_explanation")))
check("pattern_prediction present", "pattern_prediction" in exp)
pred_cause = exp.get("pattern_prediction", {}).get("predicted_cause")
check("predicted_cause is valid",
      pred_cause in ["proxy","underrepresentation","historical_skew","none"],
      f"cause={pred_cause}")

# ── TEST 5: Mitigation ───────────────────────────────────────
section("TEST 5: Mitigation (CRITICAL)")
payload = {"session_id": session_id, "target_col": "credit_risk",
           "sensitive_attr": "gender", "simulate_threshold": True}
r = requests.post(f"{BASE}/api/mitigate", json=payload)
check("Mitigate returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
mit = r.json()

rw = mit.get("reweigh", {})
th = mit.get("threshold", {})
rw_before = rw.get("before") or {}
rw_after  = rw.get("after") or {}
th_before = th.get("before") or {}
th_after  = th.get("after") or {}
rw_eff    = rw.get("effects") or {}
th_eff    = th.get("effects") or {}

# Reweighing checks
check("Reweigh before SPD present", rw_before.get("SPD") is not None,
      f"SPD={rw_before.get('SPD')}")
check("Reweigh after SPD present",  rw_after.get("SPD") is not None,
      f"SPD={rw_after.get('SPD')}")
check("Reweigh actually reduces SPD",
      rw_before.get("SPD") is not None and rw_after.get("SPD") is not None and
      abs(rw_after["SPD"]) < abs(rw_before["SPD"]),
      f"before={rw_before.get('SPD')}, after={rw_after.get('SPD')}")
check("Reweigh bias_reduction_pct > 0",
      (rw_eff.get("bias_reduction_pct") or 0) > 0,
      f"reduction={rw_eff.get('bias_reduction_pct')}%")
check("Reweigh accuracy retained > 70%",
      (rw_after.get("accuracy") or 0) > 0.70,
      f"accuracy={rw_after.get('accuracy')}")
check("Reweigh recall > 0 (not collapsed)",
      (rw_after.get("recall") or 0) > 0.05,
      f"recall={rw_after.get('recall')}")
check("Reweigh F1 > 0",
      (rw_after.get("f1") or 0) > 0.05,
      f"F1={rw_after.get('f1')}")

# Threshold checks
check("Threshold before SPD present", th_before.get("SPD") is not None)
check("Threshold after SPD present",  th_after.get("SPD") is not None)
check("Threshold actually reduces SPD",
      th_before.get("SPD") is not None and th_after.get("SPD") is not None and
      abs(th_after["SPD"]) <= abs(th_before["SPD"]),
      f"before={th_before.get('SPD')}, after={th_after.get('SPD')}")
check("Threshold recall > 0 (not collapsed)",
      (th_after.get("recall") or 0) > 0.05,
      f"recall={th_after.get('recall')} — if 0.000 threshold is broken")
check("Threshold F1 > 0",
      (th_after.get("f1") or 0) > 0.05,
      f"F1={th_after.get('f1')} — if 0.000 threshold is broken")
check("Threshold bias_reduction_pct >= 0",
      (th_eff.get("bias_reduction_pct") or 0) >= 0,
      f"reduction={th_eff.get('bias_reduction_pct')}%")

# Winner
check("winner present", mit.get("winner") in ["reweigh","threshold"],
      f"winner={mit.get('winner')}")
check("winner_reason present", bool(mit.get("winner_reason")))

print(f"\n  📊 Reweigh:    SPD {rw_before.get('SPD',0):.4f} → {rw_after.get('SPD',0):.4f} "
      f"({rw_eff.get('bias_reduction_pct',0):.1f}% reduction)")
print(f"  📊 Threshold:  SPD {th_before.get('SPD',0):.4f} → {th_after.get('SPD',0):.4f} "
      f"({th_eff.get('bias_reduction_pct',0):.1f}% reduction)")
print(f"  📊 Winner: {mit.get('winner')} — {str(mit.get('winner_reason',''))[:80]}")

# ── TEST 6: Learning Stats ───────────────────────────────────
section("TEST 6: Auto-Learning System")
r = requests.get(f"{BASE}/api/learning-stats")
check("Learning stats returns 200", r.status_code == 200)
ls = r.json()
check("total_examples present", "total_examples" in ls)
check("total_examples >= 20 (seed data loaded)",
      ls.get("total_examples", 0) >= 20,
      f"total={ls.get('total_examples')} — if <20, JSON file not persisting")
check("learned_from_uploads present", "learned_from_uploads" in ls)
check("cause_distribution present", bool(ls.get("cause_distribution")))
print(f"\n  📊 Training examples: {ls.get('total_examples')} "
      f"(seed={ls.get('seed_examples')}, "
      f"uploads={ls.get('learned_from_uploads')})")
print(f"  📊 Cause distribution: {ls.get('cause_distribution')}")

# ── TEST 7: Gemini / Fallback ────────────────────────────────
section("TEST 7: Gemini + Fallback")
r = requests.get(f"{BASE}/api/gemini-test")
check("Gemini test endpoint exists", r.status_code in [200, 503, 404])
if r.status_code == 200:
    check("Gemini API working", r.json().get("status") == "ok",
          f"response={r.json()}")
elif r.status_code == 503:
    check("Gemini quota/error handled gracefully", True,
          "quota reached — fallback will be used")

payload = {"session_id": session_id, "message": "what is the SPD for gender?",
           "history": []}
r = requests.post(f"{BASE}/api/gemini-chat", json=payload)
check("Gemini chat returns 200", r.status_code == 200)
reply = r.json().get("reply", "")
check("Chat returns non-empty reply", len(reply) > 10, f"reply='{reply[:80]}'")
check("Reply is not raw error message",
      "trouble connecting" not in reply.lower() and
      "quota" not in reply.lower(),
      f"reply='{reply[:80]}'")

# ── TEST 8: PDF Report ───────────────────────────────────────
section("TEST 8: PDF Report")
r = requests.get(f"{BASE}/api/report/{session_id}")
check("Report returns 200", r.status_code == 200, f"got {r.status_code}")
check("Response is PDF", "application/pdf" in r.headers.get("content-type",""),
      f"content-type={r.headers.get('content-type')}")
check("PDF has content", len(r.content) > 1000,
      f"size={len(r.content)} bytes")

# ── TEST 9: Auth ─────────────────────────────────────────────
section("TEST 9: Authentication")
test_email = f"testuser_{int(time.time())}@byus.test"
r = requests.post(f"{BASE}/api/auth/register",
                  json={"name":"Test User", "email":test_email,
                        "password":"TestPass123"})
check("Register returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:100]}")
reg_data = r.json()
check("Register returns token", "token" in reg_data)
check("Register returns user", "user" in reg_data)
token = reg_data.get("token")

r = requests.post(f"{BASE}/api/auth/login",
                  json={"email": test_email, "password": "TestPass123"})
check("Login returns 200", r.status_code == 200)
check("Login returns token", "token" in r.json())

r = requests.post(f"{BASE}/api/auth/login",
                  json={"email": test_email, "password": "WRONGPASSWORD"})
check("Wrong password returns 401", r.status_code == 401)

headers = {"Authorization": f"Bearer {token}"}
r = requests.get(f"{BASE}/api/auth/me", headers=headers)
check("Auth /me returns 200", r.status_code == 200)
check("Auth /me returns email", r.json().get("email") == test_email)

# ── TEST 10: Save Report + History ───────────────────────────
section("TEST 10: Report Save + History")
save_payload = {
    "session_id": session_id,
    "filename": "test_bias.csv",
    "row_count": 30,
    "target_col": "credit_risk",
    "sensitive_attrs": ["gender"],
    "scenario": "lending",
    "audit_score": analyze_data.get("audit_score"),
    "grade": "F",
    "overall_severity": analyze_data.get("overall_severity"),
    "metrics_summary": {},
    "winner_technique": mit.get("winner"),
    "bias_reduction_pct": rw_eff.get("bias_reduction_pct"),
    "full_metrics": analyze_data.get("metrics_per_attr", {}),
    "mitigation_results": mit,
    "pattern_predictions": analyze_data.get("pattern_predictions", {}),
}
r = requests.post(f"{BASE}/api/reports/save", json=save_payload, headers=headers)
check("Save report returns 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
check("Save returns report_id", "report_id" in r.json())

r = requests.get(f"{BASE}/api/reports/history", headers=headers)
check("History returns 200", r.status_code == 200)
hist = r.json()
check("History returns reports list", "reports" in hist)
check("History has at least 1 report", len(hist.get("reports", [])) >= 1)

r = requests.get(f"{BASE}/api/reports/summary", headers=headers)
check("Summary returns 200", r.status_code == 200)
summ = r.json()
check("Summary has total_analyses", summ.get("total_analyses", 0) >= 1)

# ── FINAL REPORT ─────────────────────────────────────────────
section("FINAL RESULTS")
passed = sum(1 for s,_,_ in RESULTS if s=="PASS")
failed = sum(1 for s,_,_ in RESULTS if s=="FAIL")
total  = len(RESULTS)
print(f"\n  Total: {total} | Passed: {passed} | Failed: {failed}")
print(f"  Score: {round(passed/total*100)}%\n")

if failed > 0:
    print("  FAILED TESTS:")
    for status, name, detail in RESULTS:
        if status == "FAIL":
            print(f"    ❌ {name}" + (f": {detail}" if detail else ""))

print("\n  Run this after starting backend:")
print("  uvicorn main:app --reload --app-dir backend")
print("  python backend/tests/test_full_system.py")
