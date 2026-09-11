# چارت‌های پیشنهادی ارائه Novo Pulse

تمام تصاویر این سند با `matplotlib` و از artifactهای واقعی پروژه ساخته شده‌اند. متن داخل تصاویر English است تا اصطلاحات فنی دقیق بمانند و روی سیستم ارائه به مشکل Persian text shaping نخورند.

## خلاصه اولویت‌ها

| Priority | Chart | Chart type | سؤال بیزینسی |
|---:|---|---|---|
| **P0 — 1** | Demand Formation | `Stacked Area + Line` | چه مقدار تقاضا دیده شده و چه مقدار هنوز شکل نگرفته است؟ |
| **P0 — 2** | Destination Priority | `Pareto Chart` | تیم باید بررسی را از کدام مقصدها شروع کند؟ |
| **P0 — 3** | Lead-time Risk | `Bar + Line / Dual Axis` | در هر فاصله تا سفر، forecast چقدر قابل اتکاست؟ |
| **P0 — 4** | Temporal Validation & Shock | `Grouped Bar Chart` | آیا مدل واقعاً از baseline بهتر است و در شوک چه می‌شود؟ |
| **P1 — 5** | Growth & Supply Opportunities | `Bubble Scatter / Quadrant Matrix` | برای Growth و Host Growth کدام مقصدها ارزش بررسی دارند؟ |
| **P1 — 6** | Demand-regime Accuracy | `Bar + Line / Dual Axis` | مدل روی Low، High و Peak demand چگونه عمل می‌کند؟ |
| **P2 — Appendix** | Clustering Gain Decomposition | `Bar + Stacked Gain Bar` | آیا کاهش WAPE واقعاً از مدل clustered آمده است؟ |

اگر فقط سه تصویر فرصت دارید، شماره‌های ۱، ۳ و ۴ را نشان دهید. اگر هدف ارائه بیشتر business-oriented است، شماره ۲ را نیز حتماً اضافه کنید.

---

## P0 — 1. Demand Formation

**Chart type:** `Stacked Area Chart + Line Chart`

![Demand Formation](../artifacts/pol4/presentation_charts/01_demand_formation.png)

### خیلی ساده چیست؟

- بخش نارنجی: `Observed demand`؛ چیزی که تا `cutoff` واقعاً ثبت شده است.
- بخش آبی: `Forecast remaining`؛ چیزی که مدل انتظار دارد بعداً اضافه شود.
- خط مشکی: `Final demand forecast`؛ جمع دو بخش قبلی.
- خطوط عمودی: شب `Peak` و شب `Low`.

فاصله آبی `Model Error` یا `Confidence Interval` نیست؛ همان `Predicted Remaining Demand` است. روی grid فعلی ۳۲.۲٪ تقاضای نهایی مشاهده و ۶۷.۸٪ هنوز forecast شده است. طبیعی است که این فاصله در Lead timeهای دور بزرگ‌تر باشد.

### چرا این شکل مناسب است؟

چون مسئله اصلی پروژه composition تقاضاست. یک خط ساده فقط forecast نهایی را نشان می‌دهد، ولی `Stacked Area` مرز بین fact و prediction را شفاف می‌کند. داور در چند ثانیه formulation زیر را می‌بیند:

```text
Final demand = Observed demand + Forecast remaining
```

### متن پیشنهادی ارائه

> «قسمت نارنجی جست‌وجوهایی است که تا روز تصمیم واقعاً دیده‌ایم. قسمت آبی تقاضایی است که هنوز شکل نگرفته و مدل آن را forecast می‌کند. خط مشکی خروجی نهایی است. بنابراین مدل چیزی را که مشاهده شده حذف نمی‌کند و فقط بخش نامعلوم را تخمین می‌زند.»

### Business impact

- مشخص‌کردن شب‌های مهم برای بررسی campaign calendar
- تفکیک demand قطعی از demand پیش‌بینی‌شده
- جلوگیری از تصمیم‌گیری روی یک forecast بدون context

### Data source

```text
artifacts/pol4/results_calibrated_named.csv
```

---

## P0 — 2. Destination Priority

**Chart type:** `Pareto Chart = Bar Chart + Cumulative Share Line`

![Destination Priority](../artifacts/pol4/presentation_charts/02_city_priority_pareto.png)

### خیلی ساده چیست؟

- ستون‌ها: forecast تقاضای ۳۰ شب برای هر شهر
- خط نارنجی: سهم تجمعی شهرها از کل demand پنل
- درصد بالای ستون‌های اول: سهم مستقل هر شهر

خط نارنجی باید صعودی باشد، چون سهم هر شهر به مجموع شهرهای قبلی اضافه می‌شود. مخرج آن Demand هر ۳۲۱ شهر است؛ به همین دلیل خط Top 15 در ۷۷.۸٪ تمام می‌شود و فقط در صورت نمایش همه شهرها به ۱۰۰٪ می‌رسد.

### چرا این شکل مناسب است؟

`Ranking Bar Chart` می‌گوید چه شهری بزرگ‌تر است؛ `Cumulative Share` نشان می‌دهد تمرکز تقاضا چقدر زیاد است. این دقیقاً منطق `Pareto prioritisation` است: آیا بخش بزرگی از فرصت در تعداد محدودی شهر جمع شده است؟

در خروجی فعلی، ۱۰ شهر اول حدود **69.6%** از کل forecast demand را تشکیل می‌دهند.

### متن پیشنهادی ارائه

> «هدف ما این نیست که تیم Growth هر ۳۲۱ شهر را هم‌زمان بررسی کند. این نمودار نشان می‌دهد حدود ۷۰ درصد forecast در ۱۰ شهر اول متمرکز است. پس مدل یک review queue می‌سازد و ظرفیت محدود تیم را روی مقصدهای پراثر متمرکز می‌کند.»

### Business impact

- `Budget allocation` برای مرحله بررسی، نه approval خودکار هزینه
- تمرکز analyst time روی مقصدهای پراثر
- انتخاب cohort مناسب برای campaign یا content review
- کاهش پراکندگی تصمیم‌گیری میان ۳۲۱ شهر

### Data source

```text
artifacts/pol4/results_calibrated_named.csv → city aggregation
```

---

## P0 — 3. Lead-time Risk

**Chart type:** `Bar Chart + Line Chart with Dual Axis`

![Lead-time Risk](../artifacts/pol4/presentation_charts/04_lead_time_risk.png)

### خیلی ساده چیست؟

- ستون‌های آبی: `Historical WAPE` از `D-1` تا `D-30`
- خط نارنجی: سهمی از forecast نهایی که روی target grid فعلی مشاهده شده است
- محدوده نارنجی کم‌رنگ: ناحیه پرریسک‌تر برای planning

### چرا این شکل مناسب است؟

دو اتفاق هم‌زمان رخ می‌دهد: هرچه check-in دورتر باشد observation کمتری داریم و معمولاً error بیشتر می‌شود. `Dual Axis` ارتباط این دو را روی یک timeline نشان می‌دهد.

این chart نباید به‌صورت «رابطه کاملاً monotonic» تفسیر شود؛ composition روزها و foldها متفاوت است، اما trend کلی واضح است.

### متن پیشنهادی ارائه

> «اعتبار forecast یک عدد ثابت نیست و به Lead time وابسته است. در D-1، WAPE حدود ۲.۲ درصد است؛ در D-30 به ۳۷.۵ درصد می‌رسد. بنابراین near-term forecast برای execution مناسب‌تر است، ولی far-term forecast را به‌عنوان prioritisation signal استفاده می‌کنیم، نه commit قطعی بودجه.»

### Business impact

- تعریف سطح اعتماد متفاوت برای tactical و planning decisions
- زمان‌بندی مناسب campaign review
- جلوگیری از overcommit روی horizonهای دور
- امکان ساخت `decision policy` بر اساس Lead time

### Data source

```text
Historical WAPE: artifacts/pol4/backtest_metrics_phase2.json
Observed share: artifacts/pol4/results_calibrated_named.csv
```

---

## P0 — 4. Temporal Validation & Shock

**Chart type:** `Grouped Bar Chart`

![Temporal Validation and Shock](../artifacts/pol4/presentation_charts/06_temporal_validation_shock.png)

### خیلی ساده چیست؟

- ستون خاکستری: `Pickup baseline WAPE`
- ستون آبی: `Submitted model WAPE`
- ستون نارنجی: fold هم‌پوشان با جنگ
- هر گروه: یک `rolling-origin fold` تاریخی

### چرا این شکل مناسب است؟

یک average metric ممکن است instability را مخفی کند. `Grouped Bar` هم مقایسه model/baseline و هم تفاوت regimeهای زمانی را مستقیم نشان می‌دهد.

این نمودار قوی‌ترین evidence برای پاسخ به سؤال‌های `generalisation`، `overfit` و `shock robustness` است.

### متن پیشنهادی ارائه

> «مدل روی هر پنج پنجره زمانی از Pickup baseline بهتر است. اما fold می با شروع جنگ هم‌پوشانی دارد و ۴۶.۳ درصد کل absolute error را ساخته است. ما این fold را حذف نکردیم؛ WAPE رسمی ۱۴.۷۹ درصد شامل جنگ است. عدد ۱۰.۶۱ درصد بدون آن فقط sensitivity diagnostic است.»

### Business impact

- نمایش `regime risk` و جلوگیری از اعتماد بیش از حد
- نشان‌دادن robustness مدل نسبت به baseline
- امکان تعریف human review در شرایط shock
- افزایش اعتماد داور به‌خاطر عدم حذف نتیجه نامطلوب

### نکته مهم

جنگ به‌عنوان feature وارد مدل نشده است. این رویداد unexpected بوده و استفاده از آن به‌صورت `known-future feature` اشتباه است. نمودار فقط تحلیل تشخیصی است، نه `causal impact estimate`.

### Data source

```text
artifacts/pol4/backtest_metrics_phase2.json
artifacts/pol4/jury_evidence.json
Shock dates: recorded UN DPPA source
```

---

## P1 — 5. Growth & Supply Opportunities

**Chart type:** `Bubble Scatter Plot + Quadrant Matrix`

![Growth and Supply Opportunities](../artifacts/pol4/presentation_charts/03_growth_supply_opportunities.png)

### خیلی ساده چیست؟

دو نگاه متفاوت به یک forecast است:

#### Growth review

- محور X: `Final demand forecast` با `log scale`
- محور Y: `Pickup / historical expectation`
- بالا-راست: تقاضای زیاد و pickup سریع‌تر از انتظار

#### Supply review

- محور X: `Final demand forecast` با `log scale`
- محور Y: `Remaining share of forecast`
- بالا-راست: حجم زیاد و سهم بیشتری از demand که هنوز شکل نگرفته است

در هر دو نمودار اندازه حباب برابر `Forecast remaining demand` است.

### چرا این شکل مناسب است؟

Ranking فقط «بزرگ‌ترین شهر» را نشان می‌دهد. `Quadrant Matrix` هم scale و هم momentum/risk را وارد تصمیم می‌کند. `Log scale` نیز لازم است چون demand شهرها بسیار `right-skewed` است و بدون آن شهرهای کوچک روی هم می‌افتند.

### متن پیشنهادی ارائه

> «سمت چپ برای Growth است: بالا-راست شهرهایی هستند که هم حجم بیشتری دارند و هم pickup آن‌ها سریع‌تر از انتظار تاریخی است. سمت راست برای Supply review است: شهرهایی که demand بالا و بخش پیش‌بینی‌نشده بیشتری دارند. این‌ها trigger بررسی هستند؛ نه approval خودکار campaign و نه اثبات کمبود عرضه.»

### Business impact

- `Marketing/Growth`: اولویت بررسی campaign calendar و content
- `Host Growth`: اولویت بررسی inventory و جذب میزبان
- ترکیب magnitude و momentum در یک decision view
- شناسایی شهرهای بزرگ با رفتار غیرعادی نسبت به history

### محدودیت بیزینسی

برای تبدیل این signal به تصمیم واقعی باید داده‌های زیر join شوند:

```text
Bookings · Conversion · Available inventory · Margin · Campaign cost
```

### Data source

```text
artifacts/pol4/results_calibrated_named.csv
artifacts/pol4/city_momentum.parquet
```

---

## P1 — 6. Demand-regime Accuracy

**Chart type:** `Bar Chart + Line Chart with Dual Axis`

![Demand-regime Accuracy](../artifacts/pol4/presentation_charts/05_demand_regime_accuracy.png)

### خیلی ساده چیست؟

- ستون‌ها: `WAPE` برای پنج سطح demand
- خط قرمز: `Normalised Bias`
- label روی ستون: `WAPE` و `MAE`
- پنج سطح: Low، Q50–75، Q75–90، High و Peak

### چرا این شکل مناسب است؟

یک `overall WAPE` نمی‌گوید مدل در کدام segment مشکل دارد. همچنین WAPE و MAE دو داستان متفاوت دارند:

- در Low demand مخرج کوچک است؛ WAPE بزرگ ولی absolute error کوچک است.
- در Peak demand WAPE بهتر است؛ اما absolute exposure بزرگ‌تر است.

### متن پیشنهادی ارائه

> «در نیمه Low-demand، WAPE حدود ۵۵.۵ درصد است اما MAE فقط حدود دو search است. در Peak یک درصد بالا، WAPE به ۱۲.۶ درصد می‌رسد ولی MAE نزدیک شش هزار search است. پس برای impact بیزینسی، WAPE را همیشه کنار Bias، MAE، sample size و actual volume می‌خوانیم.»

### Business impact

- انتخاب metric مناسب برای cityهای کوچک و بزرگ
- مشاهده ریسک `under-forecast` در Peak demand
- تعریف guardrail متفاوت بر اساس demand segment
- جلوگیری از تصمیم اشتباه بر اساس یک average metric

### Data source

```text
artifacts/pol4/backtest_metrics_phase2.json
Five-fold out-of-fold predictions
```

---

## ترتیب پیشنهادی نمایش در ارائه

برای یک روایت business-first:

1. `Demand Formation` — مسئله چیست؟
2. `Destination Priority` — خروجی چه تصمیمی می‌سازد؟
3. `Growth & Supply Opportunities` — use case چیست؟
4. `Lead-time Risk` — چه زمانی می‌توان اعتماد کرد؟
5. `Demand-regime Accuracy` — روی چه segmentهایی باید محتاط بود؟
6. `Temporal Validation & Shock` — evidence و صداقت ارزیابی چیست؟

برای یک روایت technical-first، نمودار `Temporal Validation & Shock` را بلافاصله بعد از `Demand Formation` نمایش دهید.

---

## P2 — Appendix. Clustering Gain Decomposition

**Chart type:** `WAPE Comparison + Stacked Gain Decomposition`

![Clustering Gain Decomposition](../artifacts/pol4/presentation_charts/07_clustering_gain_decomposition.png)

### خیلی ساده چیست؟

- سه ستون: `City-level`، همان predictionها پس از `Aggregation control` و مدل واقعی `Clustered`
- نوار سمت راست: تفکیک بهبود ظاهری به `Mechanical aggregation` و `Actual modelling`
- تصمیم: حفظ مدل `City-level`

### چرا در Appendix است؟

Clustering یک capability محصول نهایی نیست؛ یک hypothesis فنی برای کمک به `Sparse Cities` بود. نمودار نشان می‌دهد ۸۹.۸٪ کاهش ظاهری خطا قبل از یادگیری مدل جدید و صرفاً به‌دلیل aggregation رخ داده است. `Relative modelling gain` فقط ۰.۲۱٪ بود و از gate دو درصدی عبور نکرد.

### متن پیشنهادی ارائه

> «Clustering را برای pooling شهرهای کم‌داده تست کردیم. WAPE ظاهراً بهتر شد، اما با یک aggregation control نشان دادیم نزدیک ۹۰ درصد این سود مکانیکی است. بهبود واقعی مدل فقط ۰.۲۱ درصد بود؛ بنابراین complexity و از‌دست‌رفتن granularity شهر را نپذیرفتیم.»

### محدودیت مقایسه

این یک `three-fold uncalibrated cluster sweep` است. اعداد آن فقط باید داخل همان sweep مقایسه شوند و مستقیماً با WAPE پنج-fold calibrated مدل نهایی مقایسه نشوند.

### Data source

```text
artifacts/pol4/jury_evidence.json
artifacts/pol4_cluster/run_summary.json
```

## بازتولید تصاویر

```bash
.venv/bin/python scripts/pol4_presentation_charts.py
```

خروجی‌ها در مسیر زیر ساخته می‌شوند:

```text
artifacts/pol4/presentation_charts/
```
