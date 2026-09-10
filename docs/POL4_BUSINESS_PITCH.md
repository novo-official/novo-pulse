# Novo Pulse — buyer hypothesis, pilot and jury pitch

The proposition is an **earlier, auditable destination review**, using a forecast
of remaining search interest. The first commercial hypothesis is reduced analyst
planning effort and better-timed reviews. Revenue lift remains unproven.

## Who uses it, who buys it, and what changes

| Role | Current workflow hypothesis | Proposed workflow | Evidence to collect |
|---|---|---|---|
| Destination growth analyst | Manually compares recent searches across cities | Reviews a ranked city queue, pickup support and peak check-in dates | Minutes per completed review, corrections, usefulness |
| Destination growth lead | Allocates team attention from backward-looking summaries | Approves a weekly review cohort and records resulting decisions | Adoption, review completion, decision reversals |
| Analytics/engineering owner | Maintains ad hoc queries and dashboards | Runs a versioned batch with validated CSVs and explicit freshness | Runtime, failure rate, reproducibility, support hours |

These are interview hypotheses, not verified customer interviews or traction.
The growth lead is the proposed sponsor; actual budget ownership must be
confirmed. The initial alternative is the team's spreadsheet/query workflow
and the supplied pickup baseline. No unsupported competitor capability claim is
needed.

The useful differentiation is the two-clock formulation, observed-demand floor,
uncertainty disclosure, source traceability, and the measured decision not to
cluster. A longer list of model libraries is not a customer benefit.

## The review loop

1. Analyst opens the historical snapshot and confirms its cutoff and coverage.
2. Filters a province and selects a destination. The brief shows search volume,
   observation share, pickup ratio and the peak check-in date.
3. Records a content or campaign-calendar review. Search interest is enough to
   prioritize investigation; it is insufficient to approve spend or infer a
   shortage of accommodation.
4. Joins available inventory, conversion and contribution margin before proposing
   a commercial action. An owner signs off on the experimental decision.
5. Logs whether the recommendation was used, changed or rejected and why.

The app exports all 321 destination briefs as a CSV. The current dashboard is
read-only; durable decision logging and booking integration belong to the pilot,
not an already deployed capability.

## Four-week pilot proposal

The following thresholds are **proposed acceptance criteria**, to be agreed
before the pilot. They are not observed outcomes.

| Stage | Owner and deliverable | Acceptance gate |
|---|---|---|
| Week 1: define and instrument | Sponsor + analyst choose a fixed city cohort and record baseline planning time | Common definitions; data rights; event/decision timestamps; no outcome-selected cohort |
| Week 2: shadow | Analyst uses the queue alongside current planning | At least 90% of planned reviews completed; no budget changes based solely on search forecasts |
| Weeks 3–4: evaluate | Analytics lead preregisters a randomized or otherwise defensible comparison | Proposed ≥20% median review-time reduction; no increase in correction/reversal rate; complete audit trail |
| Commercial follow-up | Sponsor reviews measured value and analyst feedback | Pay only if validated labor value covers recurring fees and support costs; otherwise iterate or stop |

The pilot duration is a workflow proposal. It is **not** a promise that four weeks
provide enough samples to establish booking uplift. Commercial outcome testing
needs a power calculation based on baseline variance and a prespecified minimum
worthwhile effect. If volume is inadequate, report inconclusive and extend the
measurement window without selecting favorable outcomes.

For campaign effects, randomize at a grain that limits cross-city or time spillover,
record exposure and treatment assignment, keep a contemporaneous holdout, and
predefine the primary metric. Do not infer campaign lift from forecast accuracy.

Guardrails: gross-margin deterioration, cancellations, customer complaints,
recommendation reversals, missing-data rate and forecast freshness. If those
fields are unavailable, that commercial gate is unmeasurable and cannot pass.

## Proposed commercial offer and unit economics

Start with a fixed-scope evaluation and integration engagement, followed by a
monthly workspace subscription hypothesis. Price discovery belongs in sponsor
interviews. There is no validated willingness-to-pay, signed pilot, revenue,
TAM or market-share estimate in this repository.

The evidence-room calculator uses user-entered values:

```text
monthly labor value = (current hours/week − pilot hours/week) × loaded hourly cost × 52/12
monthly net value   = monthly labor value − recurring service and operating costs
first-year value    = 12 × monthly net value − one-time integration costs
```

All costs must use the same currency. These are scenarios, not realized ROI.
A negative value is shown as negative; it is not clamped away. Do not multiply
search forecasts by a made-up conversion rate or average booking price.

Supplier-side cost model: measured batch compute + storage + observability +
support hours + integration amortization. The local CPU run demonstrates
feasibility, not a cloud price quote or a production SLA. Recurring fees must
cover those measured costs and buyer value must exceed the fee.

Commercial validation questions for a real sponsor:

- What decision currently takes the most analyst time, and how often?
- Which inputs cause you to distrust or override a destination recommendation?
- Who owns the budget, and what result would justify paying for a pilot?
- Can bookings, conversion, inventory and margin be joined at city × check-in
  with availability timestamps and a stable city mapping?
- What security, retention and deployment constraints apply to the approved data?

## Minimum data extension

| Dataset | Required grain and fields | Enables |
|---|---|---|
| Bookings | booking ID, city, check-in, created_at, cancelled_at, net value | Conversion and booking outcomes, correctly timestamped |
| Search exposure | city, check-in, log date, sessions or eligible exposures | A defensible conversion denominator; current aggregate counts are insufficient |
| Inventory | listing/city, stay date, available units, snapshot_at | Observed availability; supply-gap analysis needs censoring treatment |
| Margin and pricing | booking/listing, net contribution, currency, effective_at | Unit economics and contribution outcome |
| Campaigns | exposure/treatment assignment, city/date, spend, recorded_at | Controlled lift measurement |
| Event calendar | event dates by year, geography, publication date, verified source, ±1-day lunar uncertainty | Scheduled-event experiments without hindsight |

Use data available at the simulated decision time. Never join a booking's later
cancellation or a revised calendar as if it were known at an earlier origin.

## Seven-minute presentation + seven-minute Q&A

The confirmed format is **14 minutes total: 7 minutes presenting and 7 minutes
for questions**. Finish the prepared presentation at 7:00. Rehearse the live
interaction once, then keep the offline deck open as a fallback.

| Clock | Slide / action | Message and evidence |
|---|---|---|
| 0:00–0:45 | 1 · Problem | “Search interest arrives over time. A destination team needs an earlier review signal, with a clear distinction between observed and forecast searches.” |
| 0:45–1:30 | 2 · Formulation | Explain the two clocks and `final = observed + remaining`. State 321 cities, seven provinces, 30 check-in dates. Show the validated 9,630-row export. |
| 1:30–2:45 | 3 · Accuracy and honesty | “WAPE is 0.147927 against 0.219991. The war-overlap fold stays in that score; excluding it gives 0.10615, diagnostic only. Five-fold uncertainty is wide.” |
| 2:45–3:40 | 4 · Clustering judgment | Explain the aggregated-city control and mostly mechanical gain. “We keep city-level submission rows. A 0.06% blend improvement does not pass our 1% practical gate.” |
| 3:40–5:10 | 5 · One decision, demonstrated | Open `/jury`, choose a province and destination, point to observation support and the decision gate, export the review CSV. Do not tour every chart. |
| 5:10–6:20 | 6 · Pilot and buyer hypothesis | Analyst workflow, proposed growth-lead sponsor, four-week pilot, proposed ≥20% review-time improvement. Commercial lift and willingness-to-pay remain unproven. |
| 6:20–7:00 | 7 · Trust and ask | Trainset hashes recovered, limitations disclosed, challengers measured separately. Ask for an operator-sponsored pilot with timestamped booking/inventory/cost data. Stop at 7:00. |

For the **seven-minute Q&A**, answer the question directly in 20–40 seconds,
then offer the relevant appendix or live evidence if needed. Keep the detailed
challenger table, history-leakage test, fold bootstrap interval, source hashes,
calibration guard and pilot economics ready. Avoid opening new experiments or
training jobs during judging.

The generated `artifacts/pol4/pitch.html` is a self-contained offline deck. Use
arrow keys to advance, `N` for speaker notes and `P` to print all slides. The
`/jury` page is the interactive evidence appendix. All numbers must come from
the generated evidence packet, including the shock diagnostic.

## Hard questions with defensible answers

**“What business improvement have you proven?”** Forecast improvement against a
strong baseline; no commercial uplift yet. We have an explicit pilot to measure
workflow value and a data contract for commercial validation.

**“Why not deploy the lowest score in the new table?”** The challenger matrix is
exploratory. Specifications and cluster height were chosen using prior evidence;
promotion needs a prespecified acceptance rule and a fresh closed holdout.

**“Is the history leakage resolved?”** The new origin-history implementation uses
only completed outcomes at each row's own forecast date and is adversarially
tested. It is an isolated challenger; the submitted model and its historical
score retain the fold-history limitation. The direct calibrated comparison is
0.147927 for the original setup and 0.151430 for origin-specific city history,
on the same five folds. Pickup curves remain fold-fitted.

**“Do your error bars guarantee per-city coverage?”** No. The headline interval
is uncertainty of pooled WAPE. Experimental prediction bands have forward-only
coverage diagnostics and are not deployed on the submission. Temporal dependence
and shocks limit the coverage interpretation.

**“Can this win?”** The project is stronger when its evidence, product workflow
and commercial hypotheses are inspectable. Rank remains the judges' decision.
