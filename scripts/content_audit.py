#!/usr/bin/env python3
"""Semantic audit: are the numbers actually right, not just present?"""
import json, urllib.request, urllib.parse

BASE = "http://127.0.0.1:8000/api/v1"
issues = []

def get(path, **params):
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=120) as r:
        return json.loads(r.read().decode())

def post(path, body):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())

def check(name, condition, detail=""):
    print(f"{'  OK ' if condition else '  !! '}{name}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        issues.append(f"{name}: {detail}")

print("=== hierarchy coherence: levels must add up")
totals = {}
for level in ("listing", "destination", "category", "market"):
    rows = get("/forecasts/", level=level, horizon=30, limit=100000)["data"]
    totals[level] = sum(r["forecast"] for r in rows)
base = totals["listing"]
for level, value in totals.items():
    check(f"{level} total == listing total", abs(value - base) < max(base * 1e-6, 0.01),
          f"{value:,.2f} vs {base:,.2f}")

print("\n=== timeseries: history and forecast must not overlap")
ts = get("/forecasts/timeseries/", level="destination", horizon=30)["data"]
split = ts["forecast_start"]
future = [p for p in ts["series"] if p["ds"] >= split]
past = [p for p in ts["series"] if p["ds"] < split]
check("forecast points all after the split", all("forecast" in p for p in future), f"{len(future)} pts")
check("no actuals in the future", not any("actual" in p for p in future))
check("history has actuals", any("actual" in p for p in past), f"{len(past)} pts")
check("interval brackets the point forecast",
      all(p.get("lower", 0) <= p["forecast"] <= p.get("upper", 1e18) for p in future if "forecast" in p))

print("\n=== overview: change % must match the underlying numbers")
rows = get("/forecasts/overview/", level="destination", horizon=30)["data"]
bad = []
for r in rows:
    if r["current_demand"]:
        expected = (r["forecast_total"] - r["current_demand"]) / r["current_demand"] * 100
        if abs(expected - r["change_pct"]) > 0.05:
            bad.append((r["entity_id"], expected, r["change_pct"]))
check("change_pct is consistent", not bad, str(bad[:2]))
check("overview total == timeseries total",
      abs(sum(r["forecast_total"] for r in rows) - ts["totals"]["forecast"]) < 1,
      f"{sum(r['forecast_total'] for r in rows):,.1f} vs {ts['totals']['forecast']:,.1f}")

print("\n=== dashboard KPIs must agree with the forecast rows")
summary = get("/dashboard/summary/", level="destination", horizon=30)["data"]
check("summary total == timeseries total",
      abs(summary["forecast_total"] - ts["totals"]["forecast"]) < 1,
      f"{summary['forecast_total']:,.1f} vs {ts['totals']['forecast']:,.1f}")
check("lower <= total <= upper",
      summary["forecast_lower"] <= summary["forecast_total"] <= summary["forecast_upper"])
check("confidence label is valid", summary["confidence"]["label"] in {"high", "medium", "low"},
      summary["confidence"]["label"])

print("\n=== horizon must actually change the answer")
h7 = get("/dashboard/summary/", level="destination", horizon=7)["data"]["forecast_total"]
h30 = get("/dashboard/summary/", level="destination", horizon=30)["data"]["forecast_total"]
check("30d total > 7d total", h30 > h7, f"{h30:,.0f} vs {h7:,.0f}")
check("roughly proportional", 3.0 < h30 / h7 < 6.0, f"ratio {h30/h7:.2f}")

print("\n=== entity filter must actually filter")
all_ts = get("/forecasts/timeseries/", level="destination", horizon=30)["data"]
kish = get("/forecasts/timeseries/", level="destination", id="kish", horizon=30)["data"]
check("entity total < market total", kish["totals"]["forecast"] < all_ts["totals"]["forecast"],
      f"{kish['totals']['forecast']:,.0f} vs {all_ts['totals']['forecast']:,.0f}")
check("entity label is Persian", kish["label"] != "kish", kish["label"])

print("\n=== leaderboard must be ranked and self-consistent")
lb = get("/models/leaderboard/")["data"]
values = [r["primary_value"] for r in lb["leaderboard"]]
check("ranked ascending (lower WAPE better)", values == sorted(values))
champion = [r for r in lb["leaderboard"] if r["is_champion"]]
check("exactly one champion", len(champion) == 1)
check("champion is not a baseline", not champion[0]["is_baseline"], champion[0]["model"])
baseline = [r for r in lb["leaderboard"] if r["is_baseline"]][0]
imp = (baseline["primary_value"] - champion[0]["primary_value"]) / baseline["primary_value"]
check("improvement matches the numbers", abs(imp - champion[0]["improvement_vs_baseline"]) < 5e-5,
      f"{imp:.4f} vs {champion[0]['improvement_vs_baseline']}")

print("\n=== backtest coverage must be near nominal")
bm = get("/backtests/metrics/")["data"]
for row in bm["coverage_by_horizon"]:
    check(f"coverage {row['bucket']} within 10pp of nominal",
          abs(row["observed_coverage"] - row["nominal_coverage"]) < 0.10,
          f"{row['observed_coverage']:.1%} vs {row['nominal_coverage']:.0%}")

print("\n=== drivers must sum to ~1 and carry directions")
dr = get("/forecasts/drivers/")["data"]
total_share = sum(g["contribution_share"] for g in dr["groups"])
check("group shares sum to ~1", abs(total_share - 1.0) < 0.02, f"{total_share:.4f}")
check("directions present", all(g["direction"] in {"positive","negative","neutral"} for g in dr["groups"]))
check("price dependence curves exist", len(dr["price_dependence"]) > 0, str(len(dr["price_dependence"])))

print("\n=== scenario engine must respond in the right direction")
baseline_run = post("/scenarios/simulate/", {"level": "destination", "entity_id": "kish",
                                             "adjustments": []})["data"]
check("empty scenario == baseline", abs(baseline_run["impact_pct"] or 0) < 1e-6,
      str(baseline_run["impact_pct"]))
cut = post("/scenarios/simulate/", {"level": "destination", "entity_id": "kish",
           "adjustments": [{"column": "price", "change_pct": -20}]})["data"]
rise = post("/scenarios/simulate/", {"level": "destination", "entity_id": "kish",
            "adjustments": [{"column": "price", "change_pct": 20}]})["data"]
check("price cut raises demand", cut["impact_pct"] > 0, f"{cut['impact_pct']:+.2f}%")
check("price rise lowers demand", rise["impact_pct"] < 0, f"{rise['impact_pct']:+.2f}%")
check("baselines identical across runs",
      abs(cut["baseline_total"] - rise["baseline_total"]) < 1e-6)
promo = post("/scenarios/simulate/", {"level": "destination", "entity_id": "kish",
             "adjustments": [{"column": "is_promotion", "value": 1, "mode": "absolute"}]})["data"]
check("promotion raises demand", promo["impact_pct"] > 0, f"{promo['impact_pct']:+.2f}%")
rejected = post("/scenarios/simulate/", {"adjustments": [{"column": "bogus", "change_pct": 5}]})["data"]
check("unknown covariate is reported as rejected", len(rejected["rejected"]) == 1)

print("\n=== narrative must be grounded in the supplied facts")
nar = get("/forecasts/narrative/", level="destination", id="kish", horizon=30)["data"]
check("narrative is non-empty Persian", len(nar["text"]) > 40 and any("؀" <= c <= "ۿ" for c in nar["text"]))
check("narrative reports grounded", nar["grounded"] is True)
facts = nar["facts"]
check("facts carry the forecast total", facts.get("forecast_total") is not None)
market_nar = get("/forecasts/narrative/", level="destination", horizon=30)["data"]
check("entity and market narratives differ", nar["text"] != market_nar["text"])

print("\n=== anomalies and peaks must be internally consistent")
an = get("/anomalies/", limit=50)["data"]
check("anomaly types valid", all(a["type"] in {"spike","drop"} for a in an))
check("severity valid", all(a["severity"] in {"high","medium","low"} for a in an))
check("spikes have positive score", all(a["score"] > 0 for a in an if a["type"] == "spike"))
check("drops have negative score", all(a["score"] < 0 for a in an if a["type"] == "drop"))
pk = get("/forecasts/peaks/", limit=10)["data"]
check("peaks are positive changes", all(p["expected_change_pct"] > 0 for p in pk["peaks"]))
check("troughs are negative changes", all(p["expected_change_pct"] < 0 for p in pk["troughs"]))
check("peaks have Persian labels", all(p["label"] for p in pk["peaks"]))

print(f"\n{'ALL CHECKS PASSED' if not issues else f'{len(issues)} ISSUES'}")
for i in issues:
    print(" !", i)
