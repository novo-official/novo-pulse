# گزارش فنی نهایی Novo Pulse — مسئله Pol 4

تاریخ گزارش: ۱۴۰۵/۰۶/۲۰ (2026-09-11)  
وضعیت: گزارش مبتنی بر artifactهای موجود؛ مدل یا خروجی مسابقه در زمان تهیه این گزارش دوباره train نشده است.

## ۱. نتیجه اجرایی

مدل اصلی پروژه **City-level calibrated two-band LightGBM با ۶۶ فیچر** است. این مدل تقاضای نهایی را مستقیم پیش‌بینی نمی‌کند؛ ابتدا «تقاضای باقی‌مانده» را پیش‌بینی می‌کند و سپس آن را به تقاضای مشاهده‌شده تا cutoff اضافه می‌کند:

```text
remaining_target = max(final_demand - observed_so_far, 0)
final_prediction = observed_so_far + max(predicted_remaining, 0)
```

در نتیجه مدل هیچ‌وقت نمی‌تواند کمتر از جست‌وجوهای ثبت‌شده تا زمان تصمیم پیش‌بینی کند.

نتیجه‌ی walk-forward روی پنج پنجره شبیه‌سازی‌شده:

| روش | WAPE | Normalised bias | نقش |
|---|---:|---:|---|
| Pickup baseline | 21.9991% | -14.9974% | baseline بیزینسی و fallback |
| همان LightGBM پیش از calibration | 15.5080% | -9.8551% | raw variant برای اندازه‌گیری اثر calibration |
| **مدل اصلی کالیبره‌شده** | **14.7927%** | **-6.8189%** | champion و سازنده `results.csv` |

مدل اصلی نسبت به pickup baseline حدود **32.76% کاهش نسبی WAPE** دارد. Calibration نیز نسبت به نسخه خام، WAPE را 0.7153 واحد درصد کاهش داده و bias را 3.0362 واحد درصد به صفر نزدیک کرده است.

مدل clustered یک challenger آزمایشی است، نه مدل نهایی. بهترین سطح ظاهراً WAPE سه-fold را از 12.0168% به 11.7699% رساند، اما فقط 0.2105% بهبود نسبی به خود مدل‌سازی نسبت داده شد و عمده تفاوت از جمع‌زدن شهرها و خنثی‌شدن خطاها بود. چون gate پروژه حداقل 2% **modelling gain** بود، clustering رد و خروجی city-level حفظ شد.

## ۲. صورت مسئله و دو ساعت زمانی

هر رکورد خام تعداد جست‌وجو برای یک `city × checkin` در یک `log_date` است. بنابراین دو زمان داریم:

- `log_date`: جست‌وجو چه زمانی ثبت شده است.
- `checkin`: کاربر برای چه تاریخی قصد سفر داشته است.
- `lead time = checkin - log_date`: چند روز پیش از اقامت جست‌وجو انجام شده است.

در cutoff مسابقه، بخشی از تقاضای هر check-in دیده شده و بخشی هنوز در آینده ثبت خواهد شد. مسئله واقعی پیش‌بینی همین بخش باقی‌مانده است، نه بازسازی بخش معلوم.

- cutoff: `2025-11-21`
- بازه هدف: `2025-11-22` تا `2025-12-21`
- خروجی: ۳۲۱ شهر × ۳۰ شب = ۹٬۶۳۰ ردیف
- معنای demand: تعداد search؛ نه booking، revenue یا occupancy

## ۳. داده و کنترل کیفیت

| فایل | تعداد ردیف داده | کاربرد |
|---|---:|---|
| `search_data.csv` | 3,298,564 | تاریخچه بسته‌شده برای train و backtest |
| `evaluation.csv` | 65,411 | جست‌وجوهای قابل مشاهده برای grid نهایی |
| `cities.csv` | 321 | شهر، استان و مختصات |
| `city_code_mapping.csv` | 321 | نام خوانای شهرها |

کنترل‌های انجام‌شده:

- duplicate روی `(log_date, city, checkin)` وجود ندارد.
- `log_date > checkin` و lead time خارج از ۰ تا ۵۹ وجود ندارد.
- تاریخ‌های check-in در search و evaluation overlap ندارند.
- هیچ داده‌ای پس از cutoff وارد featureهای grid نهایی نشده است.
- تمام ۳۲۱ شهر mapping و سابقه دارند.
- hash چهار ورودی در `input_manifest.json` ثبت شده است.

داده فقط هفت استان را پوشش می‌دهد؛ بنابراین اصطلاح صحیح **in-panel / هفت‌استانی** است و نباید آن را تقاضای کل ایران نامید.

## ۴. چرا اندازه trainset با فایل خام فرق دارد؟

فایل search، eventهای sparse را می‌شمارد؛ trainset ردیف‌های supervised روی grid dense را:

```text
321 cities × 752 check-in days × 11 sampled horizons = 2,655,312 rows
```

یازده horizon آموزشی عبارت‌اند از:

```text
1, 2, 3, 5, 7, 10, 14, 18, 21, 25, 30
```

در trainset محدودیت تصادفی ردیف نداریم:

```text
train_window_days = 752
city_history_days = 752
max_train_rows = None
sampled = false
```

هر ردیف train یک وضعیت تاریخی واقعی را شبیه‌سازی می‌کند: «اگر h روز تا check-in باقی مانده بود، چه چیزهایی می‌دانستیم و بعداً چه مقدار search اضافه شد؟»

## ۵. ساخت featureها

مدل اصلی ۶۶ feature در ۹ خانواده دارد:

| خانواده | هدف |
|---|---|
| Base | حجم مشاهده‌شده و فاصله تا check-in |
| Pickup | مقدار جست‌وجوی تازه در پنجره‌های ۱، ۳، ۷ و ۱۴ روزه |
| Velocity | سرعت، acceleration و نسبت پنجره‌های pickup |
| Activity | تعداد روز فعال، فاصله اولین/آخرین search و پراکندگی فعالیت |
| Curve | درصد تاریخی تکمیل تقاضا و تخمین pickup baseline |
| Calendar | روز هفته، weekend، روز ماه و ماه/روز جلالی |
| City history | mean، median، quantileها، volatility و الگوی weekday شهر |
| In-panel market | حرکت کل هفت استان و سهم شهر از پنل |
| Province | حرکت استان و سهم شهر از استان |

`city_code` و `province_code` در LightGBM categorical هستند، نه عدد ترتیبی.

بیشترین gain ثبت‌شده متعلق به این featureهاست:

| Feature | سهم gain |
|---|---:|
| `city_weekday_mean` | 46.77% |
| `city_hist_mean` | 15.81% |
| `city_hist_median` | 13.73% |
| `pickup_baseline_remaining` | 6.58% |
| `city_hist_p75` | 5.01% |

روی grid واقعی ۹٬۶۳۰ ردیفی، schema دقیقاً ۶۶ feature است، مقدار infinite نداریم و nullهای سه feature activity فقط برای ردیف‌هایی هستند که هنوز هیچ search مشاهده‌شده‌ای ندارند.

### محدودیت City History

مدل submit‌شده city-history را در cutoff هر fold محاسبه و برای originهای آموزشی همان fold ثابت می‌کند. این روش از آینده‌ی fold عبور نمی‌کند و train/serve consistent است، اما برای یک ردیف آموزشی قدیمی می‌تواند summary روزهای بعدی داخل همان fold را ببیند.

نسخه origin-specific جداگانه آزمایش شد:

| نسخه ۶۶ فیچری | Raw WAPE | Calibration همسان |
|---|---:|---:|
| history ثابت در cutoff fold | 15.5080% | **14.7927%** |
| history ساخته‌شده در origin هر ردیف | 15.6905% | 15.1430% |

نسخه origin-specific از نظر مفهومی تمیزتر بود ولی روی شواهد موجود بهتر نشد؛ به همین دلیل بدون holdout جدید جایگزین مدل submit‌شده نشد. این موضوع باید به‌عنوان محدودیت روش، صادقانه بیان شود.

## ۶. انتخاب مدل

### ۶.۱ Pickup baseline

Baseline با استفاده از completion curve تاریخی تخمین می‌زند چه سهمی از تقاضا تا هر lead time دیده شده است. اگر حمایت city کافی نباشد به province و سپس global in-panel fallback می‌کند. این روش سریع، قابل توضیح و fallback عملیاتی است، ولی تعامل‌های پیچیده، شتاب اخیر و regimeهای شهر را خوب مدل نمی‌کند.

### ۶.۲ LightGBM روی remaining demand

LightGBM با objective از نوع L1 آموزش داده شد؛ چون معیار اصلی WAPE بر مجموع absolute error تکیه دارد. target به دلیل heavy-tail با `log1p` تبدیل و در inference با `expm1` برگردانده می‌شود. آزمایش ثبت‌شده نشان داد log1p در این formulation، WAPE و عملکرد top-demand را بهتر کرده است.

به‌جای یک مدل برای همه horizonها، دو booster داریم:

- D-1 تا D-14: شرایط نزدیک به check-in و مشاهده بیشتر
- D-15 تا D-30: عدم قطعیت و remaining demand بیشتر

دو band مصالحه‌ای میان تخصصی‌شدن horizon و کم‌شدن sample هر مدل بود. چهار band pooled score بهتری نشان داده بود، اما روی seasonal analogue آذر افت تعمیم بیشتری داشت.

### ۶.۳ Calibration محافظت‌شده

نسخه خام under-forecast داشت. سه روش calibration فقط با out-of-fold predictionها مقایسه شدند:

| روش | WAPE | Bias | Safe |
|---|---:|---:|---|
| `horizon_shrunk` | 14.9292% | -7.1386% | بله |
| **`guarded_bias_horizon`** | **14.7927%** | **-6.8189%** | بله |
| `bias_horizon_shrunk` | 17.2189% | -1.3367% | خیر |

نزدیک‌ترین bias به صفر الزاماً بهترین مدل نیست. روش سوم bias را بسیار کم کرد، ولی WAPE را خراب کرد و رد شد. روش نهایی برای هر horizon bucket یک ضریب محدود و shrink‌شده اعمال می‌کند و فقط وقتی پذیرفته می‌شود که:

- pooled WAPE از raw بدتر نشود؛
- قدرمطلق bias کمتر شود؛
- هیچ fold بیش از یک واحد درصد WAPE آسیب نبیند.

ضرایب نهایی از 1.00 تا 1.1025 هستند؛ یعنی calibration اصلاح محدود است، نه افزایش آزادانه‌ی همه پیش‌بینی‌ها.

## ۷. پروتکل ارزیابی و ریسک overfit

پنج rolling-origin fold هر کدام دقیقاً مسئله‌ی ۳۲۱ شهر × ۳۰ روز را شبیه‌سازی می‌کنند:

| Cutoff | WAPE مدل اصلی | Bias |
|---|---:|---:|
| 2024-11-21 | 12.1684% | +0.7964% |
| 2025-05-21 | 27.2169% | -21.2095% |
| 2025-08-21 | 11.4597% | +0.5171% |
| 2025-09-22 | 9.2630% | -4.3833% |
| 2025-10-22 | 9.5324% | -5.9185% |

- مدل در هر پنج fold از pickup baseline بهتر است.
- WAPE تجمیعی: 14.7927%
- prequential calibration WAPE: 14.8618%
- last-fold holdout: 9.5324% در برابر baseline برابر 21.6650%
- bootstrap CI برای WAPE تجمیعی: 9.7554% تا 22.1850%

Raw train WAPE برابر 6.8722% و validation WAPE برابر 15.5080% است؛ شکاف 8.6358 واحد درصد یک **هشدار overfit یا distribution shift** است، نه اثبات قطعی overfit. نزدیک‌بودن prequential score به headline score امیدوارکننده است، ولی فقط پنج fold داریم و model specification داخل یک nested outer loop انتخاب نشده است.

## ۸. جنگ دوازده‌روزه؛ metric و منبع

رویداد ثبت‌شده در artifact پروژه **جنگ ایران و اسرائیل در ژوئن ۲۰۲۵** است، نه جنگ ایران و آمریکا. بازه مرجع `2025-06-13` تا `2025-06-24` از منبع ثبت‌شده‌ی United Nations DPPA گرفته شده است:

```text
https://dppa.un.org/en/node/101504
```

fold با cutoff برابر `2025-05-21`، بازه هدف `2025-05-22` تا `2025-06-20` دارد؛ بنابراین هشت روز انتهایی target آن با شروع جنگ overlap دارد.

شواهد:

- WAPE این fold: **27.2169%**؛ بدترین fold
- Normalised bias: **-21.2095%**؛ under-forecast شدید
- سهم این fold از کل absolute error پنج fold: **46.30%**
- WAPE همه foldها: **14.7927%**
- WAPE بدون این fold: **10.6147%**

فرمول سهم خطا:

```text
shock error share = sum(abs(error) in shock-overlap fold)
                   / sum(abs(error) in all folds)
```

عدد «بدون fold» فقط sensitivity diagnostic است و score رسمی نیست؛ score رسمی جنگ را حذف نمی‌کند. event study همچنین volume را در دو clock `checkin` و `log_date` برای pre/during/post، با weekday-adjusted ratio و matched-Jalali-year ratio توصیف می‌کند. این تحلیل **علّی نیست**: جنگ treatment از پیش معلوم برای مدل نبود و control group تمیزی نیز وجود ندارد.

هیچ war flag یا event/holiday feature وارد مدل اصلی نشده است. این تصمیم درست است چون شوک ناگهانی در inference از قبل معلوم نیست. برای تعطیلات قابل برنامه‌ریزی، calendar رسمی سال‌به‌سال، geography، زمان انتشار تقویم و عدم قطعیت ±۱ روز برای تاریخ قمری لازم است؛ تا زمانی که منبع معتبر تکمیل نشده نباید تاریخ ساخته شود.

## ۹. تحلیل High demand و Low demand

backtest با quantileهای تقاضای واقعی به پنج regime تقسیم شده است:

| بازه | ردیف | WAPE | MAE | Bias |
|---|---:|---:|---:|---:|
| Low، نیمه پایین | 24,171 | 55.5263% | 1.5874 | -1.1085% |
| Q50–75 | 11,948 | 28.0303% | 21.5366 | -7.6837% |
| Q75–90 | 7,217 | 20.1255% | 151.5783 | -5.4407% |
| High، Q90–99 | 4,329 | 14.9842% | 1,216.2912 | -6.6523% |
| Peak، یک درصد بالا | 485 | 12.5775% | 5,933.3018 | -7.3847% |

WAPE در low-demand بزرگ است چون مخرج demand کوچک است؛ MAE فقط 1.59 search است. پس برای شهرهای کم‌تقاضا WAPE به‌تنهایی می‌تواند گمراه‌کننده باشد. برای peakها WAPE پایین‌تر است، اما MAE مطلق و under-forecast از نظر تجاری مهم‌تر می‌شود.

برش مستقل پرترافیک:

| برش | WAPE | Bias |
|---|---:|---:|
| Top 1% | 13.4241% | -6.9470% |
| Top 5% | 13.8793% | -7.2931% |
| Top 10% | 14.1698% | -7.1086% |

برای تصمیم بیزینسی باید WAPE، Bias، MAE، تعداد ردیف و actual volume کنار هم خوانده شوند؛ «دقت درصدی خوب» در peak به معنی کوچک‌بودن خطای مطلق بودجه یا ظرفیت نیست.

## ۱۰. بررسی dimensionهای مختلف

ابعاد ارزیابی موجود:

- زمان: fold و cutoff
- lead time: روزبه‌روز D-1 تا D-30 و پنج bucket
- demand regime: Low، میانی، High و Peak
- geography: استان
- calendar: روز هفته
- observation state: دارای search مشاهده‌شده یا بدون مشاهده
- entity/date در inference: شهر، استان و check-in
- stability: تغییر forecast میان snapshotهای D-30، D-21، D-14، D-7، D-3 و D-1

نمونه insightها:

- بهترین WAPE استانی: Hormozgan با 7.0663%؛ ضعیف‌ترین: Alborz با 22.9488%.
- ردیف‌های بدون observation، WAPE برابر 64.6098% و bias برابر -41.0453% دارند؛ cold-start/zero-observation مهم‌ترین segment ریسکی است، هرچند حجم کل آن‌ها کم است.
- میان weekdayها، جمعه WAPE برابر 17.4809% دارد و شنبه 10.8237%؛ تقویم روی دقت اثر قابل مشاهده دارد.

این breakdownها برای پیدا کردن failure mode هستند، نه انتخاب post-hoc عدد مطلوب برای ارائه.

## ۱۱. بررسی دقیق Lead time

WAPE برای هر یک از ۳۰ روز جداگانه ذخیره شده و علاوه بر آن برای خوانایی در پنج bucket جمع می‌شود:

| Lead-time bucket | WAPE | Bias |
|---|---:|---:|
| D-1 تا D-3 | 2.9215% | -1.0399% |
| D-4 تا D-7 | 6.0656% | -0.6652% |
| D-8 تا D-14 | 10.7822% | -1.2869% |
| D-15 تا D-21 | 12.4440% | -2.7873% |
| D-22 تا D-30 | 26.3073% | -18.0883% |

نقاط منتخب روزبه‌روز:

| روز | WAPE | Bias |
|---|---:|---:|
| D-1 | 2.2002% | -1.1119% |
| D-3 | 3.7240% | -0.4307% |
| D-7 | 7.5668% | -1.7336% |
| D-14 | 14.6001% | -7.0648% |
| D-21 | 11.1247% | -3.3024% |
| D-30 | 37.5336% | -29.0269% |

رابطه کاملاً monotonic نیست، چون روز هفته، فصل، fold و ترکیب demand تغییر می‌کند؛ اما الگوی کلی واضح است: هرچه از check-in دورتر شویم، observation کمتر و uncertainty و under-forecast بیشتر می‌شود. بنابراین خروجی D-22 تا D-30 باید برای prioritisation استفاده شود، نه commit قطعی بودجه یا ظرفیت.

## ۱۲. آزمایش Clustering

هدف clustering کمک به شهرهای sparse با pooling شهرهای مشابه بود. similarity از رفتار demand، pickup، تقویم، geography و profile تاریخی ساخته شد و dendrogram در چند cut height ارزیابی شد.

- ۳۲۱ شهر در سطح منتخب آزمایش به ۱۰۹ virtual city تبدیل شدند.
- trainset clustered: `109 × 752 × 11 = 901,648` ردیف و ۶۹ feature
- سه feature اضافه: اندازه cluster، حداقل حجم member و بیشترین سهم member
- شهرهای بزرگ با threshold سهم 0.1% محافظت شدند.

اما WAPE روی aggregate row ذاتاً می‌تواند بهتر دیده شود، چون خطای مثبت و منفی شهرها هنگام جمع‌شدن خنثی می‌شوند. برای همین gain به دو بخش تقسیم شد:

- mechanical gain: سودی که فقط از aggregation metric حاصل می‌شود.
- modelling gain: چیزی که بعد از کنترل aggregation واقعاً از مدل خوشه‌ای باقی می‌ماند.

بهترین سطح ظاهری حدود ۱۱۰ گروه داشت:

- total relative gain: 2.0546%
- modelling relative gain: فقط 0.2105%
- gate لازم: 2% modelling gain

پس مدل city انتخاب شد. این «رد آگاهانه‌ی complexity» بخش مهم storytelling است: پروژه clustering را صرفاً چون پیچیده‌تر بود promote نکرد.

## ۱۳. خروجی نهایی

فایل رسمی:

```text
artifacts/pol4/results.csv
```

مشخصات:

- schema: `cluster_code, checkin, predicted_demand`
- ردیف: 9,630
- شهر: 321
- تاریخ: 30
- بازه: `2025-11-22` تا `2025-12-21`
- مجموع بعد از integer rounding: 8,352,768
- duplicate key: صفر
- مقدار منفی یا non-finite: صفر
- observed-demand floor: برقرار
- SHA-256: `a0b31f3feb659b8c1838384a31db31c68d62de51ca7227d70ba3f0223528be75`

`results.csv` و `results_calibrated.csv` byte-identical هستند. فایل `results_named.csv` companion خوانا با نام شهر و استان است، اما schema مسابقه را ندارد و فایل submit نیست.

## ۱۴. Insightهای قابل دفاع

- تقاضای ۳۰ شب آینده در چند شهر متمرکز است؛ Tehran، Qeshm و Kish در صدر forecast قرار دارند.
- مدل در near-term بسیار قابل اتکاتر است؛ D-1 تا D-7 برای اقدام tactical مناسب‌تر از D-22 تا D-30 است.
- ریسک اصلی under-forecast در horizonهای دور، zero-observation و regime shock دیده می‌شود.
- clustering برای این داده ارزش مدل‌سازی کافی نساخت؛ city-level granularity برای تصمیم و submission حفظ شد.
- مدل روی peak-demand درصد خطای قابل قبول‌تری دارد، اما خطای مطلق peak همچنان بزرگ و مهم است.
- جنگ نشان داد averaging کل می‌تواند regime risk را پنهان کند؛ باید fold-level performance کنار headline metric نمایش داده شود.

## ۱۵. محدودیت‌ها

- فقط پنج fold زمانی داریم؛ uncertainty گسترده است.
- target مسابقه label ندارد و تنها test واقعاً unseen است.
- model selection به‌صورت nested داخل outer folds انجام نشده است.
- city-history مدل اصلی محدودیت within-training-origin دارد.
- search معادل booking یا revenue نیست.
- داده inventory، conversion، margin و campaign exposure وجود ندارد؛ supply gap و ROI قابل اثبات نیست.
- holiday/event calendar معتبر و versioned وارد مدل نشده است.
- جنگ فقط diagnostic است و causal effect اندازه‌گیری نشده است.
- ۸۲ شهر روی grid نهایی هیچ observation ندارند و ۷۸ شهر پس از rounding مجموع forecast صفر دارند؛ خروجی شهرهای بسیار sparse باید با احتیاط استفاده شود.

## ۱۶. Storytelling پیشنهادی برای داور

روایت اصلی بهتر است این ترتیب را داشته باشد:

1. در روز cutoff بخشی از search را دیده‌ایم؛ سؤال این است که تا check-in چقدر دیگر اضافه می‌شود.
2. یک pickup baseline شفاف ساختیم تا مدل پیچیده مجبور باشد واقعاً آن را شکست دهد.
3. target را remaining demand تعریف کردیم تا observation از بین نرود و leakage زمانی را ساختاری بستیم.
4. LightGBM دو-band با ۶۶ feature، WAPE را از 22.00% به 15.51% رساند.
5. calibration محافظت‌شده، نه calibration تهاجمی، نتیجه را به 14.79% و bias را به -6.82% رساند.
6. نتیجه را فقط aggregate گزارش نکردیم؛ Low/High/Peak، استان، weekday، observation state و D-1 تا D-30 را جدا دیدیم.
7. بدترین fold را پنهان نکردیم: overlap با جنگ 46.3% کل absolute error را ساخته است. آن را در score نگه داشتیم و فقط sensitivity را جدا نشان دادیم.
8. clustering را آزمایش کردیم؛ سود ظاهری عمدتاً mechanical بود، پس مدل ساده‌تر و قابل اقدام city-level را نگه داشتیم.
9. خروجی نهایی دقیقاً ۹٬۶۳۰ ردیف معتبر دارد؛ اما search signal است و برای تصمیم مالی باید booking، inventory، conversion و margin به آن متصل شود.

پیام استراتژیک: **ارزش پروژه فقط یک WAPE بهتر نیست؛ ارزش آن یک forecast زودتر، قابل audit و segment-aware است که می‌گوید کجا به مدل اعتماد کنیم و کجا نیاز به بررسی انسانی یا داده بیشتر داریم.**

## ۱۷. بازتولید و اعتبارسنجی

```bash
# اجرای کامل train، backtest، calibration و ساخت results.csv
make pol4

# inference از bundle ذخیره‌شده، بدون train مجدد
make pol4-predict

# تست‌های Pol 4
make test-pol4

# بازسازی trainset ثبت‌شده
make pol4-recover-trainset

# event study و evidence مربوط به جنگ
make pol4-event-study

# آزمایش clustering
make pol4-cluster
```

منابع اصلی این گزارش:

- `artifacts/pol4/run_summary.json`
- `artifacts/pol4/model_card.json`
- `artifacts/pol4/trainset_manifest.json`
- `artifacts/pol4/backtest_metrics_phase2.json`
- `artifacts/pol4/validation_audit.json`
- `artifacts/pol4/shock_event_study.json`
- `artifacts/pol4/jury_evidence.json`
- `artifacts/pol4_cluster/run_summary.json`

