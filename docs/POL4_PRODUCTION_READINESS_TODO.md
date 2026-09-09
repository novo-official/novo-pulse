# Pol4 Hackathon Readiness

دامنه این سند فقط آماده بودن داده، الگوریتم، مدل و خروجی قابل ارائه در هکاتون است. استقرار production، زیرساخت ابری و عملیات بلندمدت عمدا خارج از scope هستند.

## داده و provenance

- [x] تنها منابع خام pipeline این چهار فایل هستند:
  - `data/raw/pol4/search_data.csv`
  - `data/raw/pol4/evaluation.csv`
  - `data/raw/pol4/cities.csv`
  - `data/raw/pol4/city_code_mapping.csv`
- [x] loader قرارداد ستون ها، تاریخ ها، مقادیر، duplicateها، پوشش شهرها و بازه lead-time را کنترل می کند.
- [x] `input_manifest.json` برای هر چهار ورودی مسیر، تعداد ردیف، schema، حجم و SHA-256 ثبت می کند.
- [x] digest مشترک چهار ورودی در trainset، model bundle و `run_summary.json` ثبت می شود.
- [x] هیچ fallback در package اختصاصی Pol4 به `data/uploads/` یا داده نمونه وجود ندارد.
- [x] تست pipeline دقیقا چهار source مجاز را کنترل می کند.

## Trainset

- [x] trainset از `search_data.csv` و featureهای جغرافیایی `cities.csv` ساخته می شود.
- [x] target هر ردیف به صورت `max(final - observed, 0)` ساخته و قبل از ذخیره اعتبارسنجی می شود.
- [x] کنترل cutoff و leakage موجود است؛ feature افق `h` فقط داده قابل مشاهده تا `T-h` را می بیند.
- [x] ۶۶ feature با نام، ترتیب و dtype مشخص ذخیره می شوند.
- [x] trainset به سه فایل Zstandard Parquet تفکیک شده است:
  - `artifacts/pol4/trainset/features.parquet`
  - `artifacts/pol4/trainset/target.parquet`
  - `artifacts/pol4/trainset/meta.parquet`
- [x] metadata شامل `city_code`, `checkin`, `horizon`, `observed` و `final` است.
- [x] manifest شامل cutoff، seed، horizonها، feature groups، schema، تعداد قبل و بعد sampling و checksum فایل هاست.
- [x] نوشتن هر فایل اتمیک است.
- [x] loader مستقل قبل از استفاده checksum، schema و تعداد ردیف ها را بررسی می کند.
- [x] گزینه `--reuse-trainset` فقط trainset منطبق با input digest و training contract را می پذیرد.
- [x] تست round-trip و تشخیص trainset خراب وجود دارد.

## الگوریتم و مدل

- [x] انتخاب champion با walk-forward backtest پنج fold انجام می شود.
- [x] baseline، calibrationهای global/horizon و LightGBM در گزارش مقایسه می شوند.
- [x] مدل هدف `remaining_demand` را با objective نوع L1 و تبدیل `log1p` پیش بینی می کند.
- [x] floor منطقی `predicted_final >= observed_so_far` تضمین شده است.
- [x] champion شامل دو مدل horizon است:
  - LightGBM برای `D-1 ... D-14`
  - LightGBM برای `D-15 ... D-30`
- [x] هر Booster با فرمت بومی LightGBM ذخیره می شود:
  - `artifacts/pol4/model_bundle/lightgbm_h1_14.txt`
  - `artifacts/pol4/model_bundle/lightgbm_h15_30.txt`
- [x] `model_spec.json` شامل bandها، پارامترها، log transform، seed، feature order و categorical features است.
- [x] baseline، pickup curves و zero-observation prior در `baseline_state.joblib` ذخیره می شوند.
- [x] state نوع joblib فقط بعد از تایید checksum load می شود.
- [x] `FittedChampion.save()` و `FittedChampion.load()` پیاده سازی شده اند.
- [x] model bundle دارای manifest و checksum مستقل است.
- [x] bundle ناقص، دستکاری شده یا دارای feature contract ناسازگار fail-fast می شود.
- [x] تست prediction parity قبل و بعد از serialization وجود دارد.

## Inference و خروجی هکاتون

- [x] `python -m ml.pol4.inference` مدل ذخیره شده را بدون train مجدد اجرا می کند.
- [x] دستور `make pol4-predict` برای اجرای ساده همان مسیر وجود دارد.
- [x] digest چهار CSV باید با model bundle برابر باشد؛ در غیر این صورت inference متوقف می شود.
- [x] inference مدل load شده دوباره با validator رسمی ۹٬۶۳۰ ردیف کنترل می شود.
- [x] خروجی کددار و نام دار تولید می شود.
- [x] pipeline اصلی پس از ذخیره مدل، آن را در همان run دوباره load می کند و parity را با tolerance سخت بررسی می کند.
- [x] `run_summary.json` مسیر و checksum trainset/model bundle و نتیجه parity را ثبت می کند.

## تست و مستندات هکاتون

- [x] unit test serialization و checksum مدل
- [x] unit test materialization و checksum trainset
- [x] integration test مسیر raw CSV تا trainset، bundle و prediction بدون train
- [x] leakage test برای داده های post-cutoff
- [x] determinism test با seed ثابت
- [x] اعتبارسنجی کامل submission شامل ستون ها، تاریخ ها، شهرها، duplicate، NaN و مقادیر منفی
- [x] README شامل مسیر چهار ورودی، artifactهای جدید و command inference است.
- [x] `model_card.json` تعریف target، الگوریتم، validation، invariants و محدودیت های تفسیر مدل را ثبت می کند.
- [x] پس از آخرین تغییر کد، pipeline کامل روی چهار CSV اصلی دوباره اجرا و artifactهای نهایی ثبت شد.
- [x] `make pol4-predict` روی bundle نهایی اجرا شد و خروجی آن با `results.csv` hash یکسان دارد.
- [x] suite کامل backend و تست، typecheck، lint و build فرانت اند اجرا و تایید شد.

## خارج از scope این هکاتون

موارد زیر برای محصول production واقعی مفیدند، اما طبق تصمیم فعلی وارد فرایند این تحویل نمی شوند:

- Gunicorn، reverse proxy، TLS و hardening شبکه
- PostgreSQL/Redis و چند worker
- CI/CD، blue-green deployment و model registry ابری
- object storage، retention و rollback چند release
- monitoring دائمی، alerting و drift service آنلاین
- authentication، rate limiting، SLA/SLO و disaster recovery

## Definition of Done هکاتون

- [x] منشأ تمام داده ها تا hash چهار CSV قابل ردیابی است.
- [x] trainset آماده، audit پذیر و قابل load مجدد است.
- [x] هر دو LightGBM serialize و checksum-protected هستند.
- [x] مدل در process جدید بدون fit قابل استفاده است.
- [x] خروجی مدل load شده از validator مسابقه عبور می کند.
- [x] artifact نهایی روی دیتای اصلی پس از آخرین تغییرات بازتولید و تایید شد.
