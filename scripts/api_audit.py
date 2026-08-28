#!/usr/bin/env python3
"""Exhaustive live API audit. Hits every endpoint with good and bad input."""
import json, sys, urllib.request, urllib.error, urllib.parse, io, uuid

BASE = "http://127.0.0.1:8000/api/v1"
results = []

def call(method, path, params=None, body=None, files=None, expect=200, note=""):
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    data, headers = None, {"Accept": "application/json"}
    if files:
        boundary = uuid.uuid4().hex
        buf = io.BytesIO()
        for key, (fname, content) in files.items():
            buf.write(f"--{boundary}\r\n".encode())
            buf.write(f'Content-Disposition: form-data; name="{key}"; filename="{fname}"\r\n'.encode())
            buf.write(b"Content-Type: application/octet-stream\r\n\r\n")
            buf.write(content if isinstance(content, bytes) else content.encode())
            buf.write(b"\r\n")
        for key, value in (body or {}).items():
            buf.write(f"--{boundary}\r\n".encode())
            buf.write(f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
        buf.write(f"--{boundary}--\r\n".encode())
        data = buf.getvalue()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            status, raw = r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read().decode()
    except Exception as e:
        results.append({"m": method, "p": path, "note": note, "status": "ERR",
                        "ok": False, "detail": f"{type(e).__name__}: {e}"})
        return None

    try:
        payload = json.loads(raw)
    except Exception:
        payload = None

    ok = status == expect
    summary = ""
    if isinstance(payload, dict):
        if "available" in payload:
            d = payload.get("data")
            n = len(d) if isinstance(d, (list, dict)) else "-"
            summary = f"available={payload['available']} data={type(d).__name__}({n})"
        else:
            summary = ",".join(list(payload)[:5])
    results.append({"m": method, "p": path, "note": note, "status": status,
                    "ok": ok, "detail": summary or raw[:100]})
    return payload

# ---------------------------------------------------------------- system
call("GET", "/health/", note="liveness")
call("GET", "/system/", note="hardware + registry + profiles")

# ------------------------------------------------------------- dashboard
call("GET", "/dashboard/summary/", note="defaults")
call("GET", "/dashboard/summary/", {"horizon": 7, "level": "destination"}, note="h=7")
call("GET", "/dashboard/summary/", {"horizon": 999}, note="horizon capped")
call("GET", "/dashboard/summary/", {"horizon": "abc"}, note="bad horizon -> default")
call("GET", "/dashboard/summary/", {"level": "nonsense"}, note="bad level -> fallback")
call("GET", "/dashboard/summary/", {"level": "market"}, note="market level")
call("GET", "/dashboard/summary/", {"level": "listing"}, note="listing level")
call("GET", "/dashboard/summary/", {"level": "category"}, note="category level")
call("GET", "/dashboard/summary/", {"run_id": "does-not-exist"}, note="unknown run -> no data")

# ------------------------------------------------------------- forecasts
call("GET", "/forecasts/", {"level": "destination"}, note="rows")
call("GET", "/forecasts/", {"level": "destination", "id": "kish", "horizon": 5}, note="filtered")
call("GET", "/forecasts/", {"level": "destination", "id": "no_such_entity"}, note="unknown entity")
call("GET", "/forecasts/", {"level": "destination", "limit": 3}, note="limit")
call("GET", "/forecasts/", {"start_date": "2026-09-01", "end_date": "2026-09-10"}, note="date range")
call("GET", "/forecasts/", {"start_date": "not-a-date"}, note="bad date ignored")
call("GET", "/forecasts/timeseries/", {"level": "destination"}, note="all destinations")
call("GET", "/forecasts/timeseries/", {"level": "destination", "id": "kish"}, note="one destination")
call("GET", "/forecasts/timeseries/", {"level": "listing"}, note="listing level")
call("GET", "/forecasts/timeseries/", {"level": "market"}, note="market level")
call("GET", "/forecasts/drivers/", note="shap groups")
call("GET", "/forecasts/peaks/", note="peaks")
call("GET", "/forecasts/peaks/", {"limit": 3, "type": "peak"}, note="filtered peaks")
call("GET", "/forecasts/overview/", {"level": "destination"}, note="overview table")
call("GET", "/forecasts/overview/", {"level": "category"}, note="overview category")
call("GET", "/forecasts/heatmap/", {"level": "destination", "top_n": 5}, note="heatmap")
call("GET", "/forecasts/narrative/", {"level": "destination"}, note="market narrative")
call("GET", "/forecasts/narrative/", {"level": "destination", "id": "kish"}, note="entity narrative")

# ------------------------------------------------------------- anomalies
call("GET", "/anomalies/", note="all")
call("GET", "/anomalies/", {"severity": "high"}, note="high only")
call("GET", "/anomalies/", {"limit": 5}, note="limited")
call("GET", "/anomalies/", {"id": "kish"}, note="by entity")

# ---------------------------------------------------------------- models
call("GET", "/models/", note="registry + trained")
call("GET", "/models/leaderboard/", note="leaderboard")

# ------------------------------------------------------------- backtests
call("GET", "/backtests/", {"level": "destination"}, note="series")
call("GET", "/backtests/", {"level": "destination", "id": "kish"}, note="one entity")
call("GET", "/backtests/metrics/", note="metrics")

# -------------------------------------------------------------- insights
call("GET", "/insights/", note="structured")
call("GET", "/insights/opportunities/", note="opportunities")
call("GET", "/insights/data-quality/", note="quality")

# ------------------------------------------------------------- scenarios
call("GET", "/scenarios/options/", note="adjustable levers")
call("POST", "/scenarios/simulate/", body={"level": "destination", "entity_id": "kish",
     "adjustments": [{"column": "price", "change_pct": -15}]}, note="price cut")
call("POST", "/scenarios/simulate/", body={"level": "destination",
     "adjustments": [{"column": "is_promotion", "value": 1, "mode": "absolute"}]}, note="promo on")
call("POST", "/scenarios/simulate/", body={"adjustments": []}, note="no adjustments")
call("POST", "/scenarios/simulate/", body={"adjustments": [{"column": "bogus", "change_pct": 5}]},
     note="unknown covariate rejected")
call("POST", "/scenarios/simulate/", body={"adjustments": "not-a-list"}, expect=400,
     note="malformed adjustments")
call("POST", "/scenarios/simulate/", body={"level": "listing", "entity_id": "acc_0001",
     "adjustments": [{"column": "price", "change_pct": 20}]}, note="listing level")

# --------------------------------------------------------------- datasets
call("GET", "/datasets/", note="list")
call("GET", "/datasets/contract/", note="active contract")
call("POST", "/datasets/profile/", body={"path": "data/synthetic/daily_demand.csv"}, note="profile by path")
call("POST", "/datasets/profile/", body={}, expect=400, note="missing args")
call("POST", "/datasets/profile/", body={"path": "../../etc/passwd"}, expect=400, note="traversal blocked")
call("POST", "/datasets/upload/", files={"file": ("t.txt", "nope")}, expect=400, note="bad file type")
call("POST", "/datasets/upload/", body={}, expect=400, note="no file, no path (json)")
call("POST", "/datasets/map/", body={"dataset_id": 1}, expect=400, note="incomplete mapping")
call("POST", "/datasets/validate/", body={"dataset_id": 999999}, expect=404, note="unknown dataset")
call("POST", "/datasets/upload/", body={"path": "data/synthetic/destinations.csv"}, note="register local path via json")

# --------------------------------------------------------------- training
call("GET", "/training/", note="runs + profiles")
call("GET", "/training/nope/", expect=404, note="unknown run")
call("POST", "/training/run/", body={"metric": "bogus"}, expect=400, note="bad metric")
call("POST", "/training/run/", body={"dataset_id": 999999}, expect=404, note="unknown dataset")
call("GET", "/experiments/", note="experiments")

# ------------------------------------------------------------------ print
bad = [r for r in results if not r["ok"]]
print(f"{'':2} {'M':<5} {'ENDPOINT':<34} {'NOTE':<28} {'ST':<5} DETAIL")
for r in results:
    mark = "  " if r["ok"] else "!!"
    print(f"{mark} {r['m']:<5} {r['p']:<34} {r['note']:<28} {str(r['status']):<5} {r['detail'][:60]}")
print(f"\n{len(results)} calls, {len(bad)} unexpected")
sys.exit(1 if bad else 0)
