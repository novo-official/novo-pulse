"""Competition-day data shapes.

The hackathon dataset had not been released when this was written, so the
ingestion layer is tested against the shapes an Iranian accommodation
marketplace actually exports: Jalali dates, Persian column headers, a raw
booking log with no demand column, and the demand / accommodation / booking
data split across separate files.

Every one of these silently destroys a run if it is not handled: Jalali dates
parse to NaT, Persian headers score zero against English keyword tables, a
booking log has nothing to sum, and unjoined files lose every covariate.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from ml.contract import ENTITY, TARGET, TS, DataContract, JoinSpec
from ml.data.adapter import DataAdapter
from ml.data.dates import (
    detect_calendar,
    jalali_to_gregorian,
    parse_timestamps,
    to_jalali_string,
    _jalali_to_gregorian_fallback,
)
from ml.data.profiler import detect_transactional, profile_dataset


# --------------------------------------------------------------- calendars
def test_jalali_dates_are_detected_and_converted():
    series = pd.Series(["1403/05/12", "1403/05/13", "1403/05/14"])
    detection = detect_calendar(series)
    assert detection.calendar == "jalali"

    parsed, _ = parse_timestamps(series)
    assert parsed.iloc[0] == pd.Timestamp("2024-08-02")
    assert parsed.notna().all()


def test_persian_digits_parse():
    parsed, detection = parse_timestamps(pd.Series(["۱۴۰۳/۰۵/۱۲", "۱۴۰۳/۰۵/۱۳"]))
    assert detection.calendar == "jalali"
    assert parsed.iloc[0] == pd.Timestamp("2024-08-02")


def test_gregorian_dates_are_left_alone():
    series = pd.Series(["2024-08-02", "2024-08-03"])
    parsed, detection = parse_timestamps(series)
    assert detection.calendar == "gregorian"
    assert parsed.iloc[0] == pd.Timestamp("2024-08-02")


def test_an_already_parsed_column_is_not_reinterpreted():
    """A join or a Parquet reader may hand us real datetimes.

    Re-reading those as Jalali would shift every date by roughly 621 years.
    """
    series = pd.Series(pd.to_datetime(["2024-08-02", "2024-08-03"]))
    parsed, detection = parse_timestamps(series, calendar="jalali")
    assert detection.calendar == "gregorian"
    assert parsed.iloc[0] == pd.Timestamp("2024-08-02")


def test_forcing_the_calendar_overrides_detection():
    # 1350 is a valid Jalali year and never a plausible Gregorian one, so the
    # explicit setting is what is being checked, not the guess.
    forced, _ = parse_timestamps(pd.Series(["1350/01/01"]), calendar="jalali")
    assert forced.iloc[0].year == 1971


def test_the_pure_python_fallback_matches_jdatetime():
    """`jdatetime` is optional; the fallback must agree with it where present."""
    jdatetime = pytest.importorskip("jdatetime")
    values = pd.Series(["1403/01/01", "1400/06/31", "1399/12/30", "1402/11/05"])
    assert (jalali_to_gregorian(values) == _jalali_to_gregorian_fallback(values)).all()
    assert jdatetime  # the import is the point


def test_bad_rows_become_nat_rather_than_raising():
    parsed, _ = parse_timestamps(pd.Series(["1403/05/12", "not a date", "", "1403/13/99"]))
    assert parsed.notna().sum() == 1


def test_jalali_display_round_trips():
    assert to_jalali_string(pd.Timestamp("2024-08-02")) == "1403-05-12"


# ------------------------------------------------------- Persian headers
def test_persian_column_names_are_recognised(tmp_path):
    frame = pd.DataFrame(
        {
            "تاریخ_رزرو": list(pd.date_range("2024-01-01", periods=40).strftime("%Y-%m-%d")) * 3,
            "کد_اقامتگاه": ["ACC-1"] * 40 + ["ACC-2"] * 40 + ["ACC-3"] * 40,
            "شهر": ["تهران"] * 40 + ["مشهد"] * 40 + ["شیراز"] * 40,
            "نوع_اقامتگاه": ["ویلا"] * 40 + ["هتل"] * 40 + ["آپارتمان"] * 40,
            "تعداد_رزرو": np.arange(120) % 7,
        }
    )
    path = tmp_path / "fa.csv"
    frame.to_csv(path, index=False)

    suggested = profile_dataset(path)["suggested_schema"]
    assert suggested["timestamp"] == "تاریخ_رزرو"
    assert suggested["target"] == "تعداد_رزرو"
    assert suggested["entity_id"] == "کد_اقامتگاه"
    assert suggested["destination"] == "شهر"
    assert suggested["category"] == "نوع_اقامتگاه"


def test_english_column_names_still_win_after_the_persian_additions(tmp_path):
    frame = pd.DataFrame(
        {
            "date": list(pd.date_range("2024-01-01", periods=40).strftime("%Y-%m-%d")) * 3,
            "accommodation_id": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
            "city": ["tehran"] * 40 + ["mashhad"] * 40 + ["shiraz"] * 40,
            "booking_count": np.arange(120) % 5,
        }
    )
    path = tmp_path / "en.csv"
    frame.to_csv(path, index=False)

    suggested = profile_dataset(path)["suggested_schema"]
    assert suggested["timestamp"] == "date"
    assert suggested["target"] == "booking_count"
    assert suggested["entity_id"] == "accommodation_id"


# ------------------------------------------------ transactional booking log
def _booking_log(n_days: int = 120, n_listings: int = 4, seed: int = 5) -> pd.DataFrame:
    """One row per booking, no demand column - the raw shape of a booking table."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        date = dt.date(2024, 1, 1) + dt.timedelta(days=day)
        for listing in range(n_listings):
            for _ in range(int(rng.poisson(2.0 + listing))):
                rows.append(
                    {
                        "تاریخ_رزرو": date.isoformat(),
                        "کد_اقامتگاه": f"ACC-{listing}",
                        "شهر": f"CITY-{listing % 2}",
                        "مبلغ": int(rng.integers(1_000_000, 9_000_000)),
                    }
                )
    return pd.DataFrame(rows)


def test_a_booking_log_is_detected_as_transactional():
    frame = _booking_log()
    result = detect_transactional(frame, "تاریخ_رزرو", "کد_اقامتگاه")
    assert result["transactional"] is True
    assert result["rows_per_period"] > 1.5


def test_the_profiler_suggests_counting_rows_for_a_booking_log(tmp_path):
    path = tmp_path / "bookings.csv"
    _booking_log().to_csv(path, index=False)

    suggested = profile_dataset(path)["suggested_schema"]
    assert suggested["aggregation"] == "count"
    # "مبلغ" is a plausible-looking numeric column; counting rows is the demand,
    # and summing rials would be measuring something else entirely.
    assert suggested["target"] is None
    assert suggested["integer"] is True


def _count_contract(path, **overrides) -> DataContract:
    raw = {
        "dataset": {"name": "bookings", "path": str(path)},
        "schema": {
            "timestamp": "تاریخ_رزرو",
            "target": None,
            "entity_id": "کد_اقامتگاه",
            "frequency": "D",
            "aggregation": "count",
            "calendar": "auto",
        },
        "hierarchy": {"destination": "شهر"},
        "features": {"future": [], "historical": [], "static": []},
        "target_options": {"non_negative": True, "integer": True},
    }
    for section, values in overrides.items():
        raw[section].update(values)
    return DataContract.from_dict(raw)


def test_count_aggregation_preserves_every_booking(tmp_path):
    frame = _booking_log()
    path = tmp_path / "bookings.csv"
    frame.to_csv(path, index=False)

    panel = DataAdapter(_count_contract(path)).build()
    # Gap filling adds zero-demand days, so the total must be unchanged.
    assert panel.frame[TARGET].sum() == pytest.approx(float(len(frame)))
    assert not panel.frame.duplicated(subset=[ENTITY, TS]).any()
    assert any("aggregation='count'" in note for note in panel.notes)


def test_count_aggregation_works_on_jalali_dates(tmp_path):
    jdatetime = pytest.importorskip("jdatetime")
    frame = _booking_log()
    frame["تاریخ_رزرو"] = [
        "{0.year:04d}/{0.month:02d}/{0.day:02d}".format(
            jdatetime.date.fromgregorian(date=dt.date.fromisoformat(value))
        )
        for value in frame["تاریخ_رزرو"]
    ]
    path = tmp_path / "bookings_jalali.csv"
    frame.to_csv(path, index=False)

    panel = DataAdapter(_count_contract(path)).build()
    assert panel.frame[TARGET].sum() == pytest.approx(float(len(frame)))
    assert any("Jalali" in note for note in panel.notes)


def test_a_target_is_still_required_without_count_aggregation(tmp_path):
    path = tmp_path / "bookings.csv"
    _booking_log().to_csv(path, index=False)
    contract = _count_contract(path, schema={"aggregation": "sum"})

    with pytest.raises(ValueError, match="no target column"):
        DataAdapter(contract).build()


# ------------------------------------------------------------------ joins
def test_side_tables_are_merged_and_reported(tmp_path):
    main = _booking_log()
    main_path = tmp_path / "bookings.csv"
    main.to_csv(main_path, index=False)

    listings = pd.DataFrame(
        {
            "کد_اقامتگاه": sorted(main["کد_اقامتگاه"].unique()),
            "ظرفیت": [4, 6, 8, 10],
        }
    )
    side_path = tmp_path / "listings.csv"
    listings.to_csv(side_path, index=False)

    contract = _count_contract(main_path)
    contract.joins = [JoinSpec(path=str(side_path), on="کد_اقامتگاه")]
    contract.static_features = ["ظرفیت"]

    adapter = DataAdapter(contract)
    panel = adapter.build()
    assert "ظرفیت" in panel.frame.columns
    assert panel.frame["ظرفیت"].notna().all()
    assert any("100% of rows matched" in note for note in panel.notes)


def test_a_join_across_two_calendars_still_matches(tmp_path):
    """The booking file may be Jalali while the calendar file is Gregorian."""
    jdatetime = pytest.importorskip("jdatetime")
    main = _booking_log(n_days=60, n_listings=2)
    gregorian_days = sorted(main["تاریخ_رزرو"].unique())
    main["تاریخ_رزرو"] = [
        "{0.year:04d}/{0.month:02d}/{0.day:02d}".format(
            jdatetime.date.fromgregorian(date=dt.date.fromisoformat(value))
        )
        for value in main["تاریخ_رزرو"]
    ]
    main_path = tmp_path / "bookings_jalali.csv"
    main.to_csv(main_path, index=False)

    holidays = pd.DataFrame(
        {"تاریخ_رزرو": gregorian_days, "تعطیل_رسمی": [i % 7 == 4 for i in range(len(gregorian_days))]}
    )
    side_path = tmp_path / "holidays_gregorian.csv"
    holidays.to_csv(side_path, index=False)

    contract = _count_contract(main_path)
    contract.joins = [JoinSpec(path=str(side_path), on="تاریخ_رزرو")]
    contract.future_features = ["تعطیل_رسمی"]

    panel = DataAdapter(contract).build()
    assert "تعطیل_رسمی" in panel.frame.columns
    assert any("both were converted before matching" in note for note in panel.notes)
    assert any("100% of rows matched" in note for note in panel.notes)


def test_a_join_that_matches_nothing_says_so(tmp_path):
    main = _booking_log(n_days=30, n_listings=2)
    main_path = tmp_path / "bookings.csv"
    main.to_csv(main_path, index=False)

    unrelated = pd.DataFrame({"کد_اقامتگاه": ["OTHER-1", "OTHER-2"], "ظرفیت": [3, 4]})
    side_path = tmp_path / "unrelated.csv"
    unrelated.to_csv(side_path, index=False)

    contract = _count_contract(main_path)
    contract.joins = [JoinSpec(path=str(side_path), on="کد_اقامتگاه")]

    panel = DataAdapter(contract).build()
    assert any("no row matched" in note for note in panel.notes)


def test_a_duplicated_side_table_cannot_multiply_rows(tmp_path):
    main = _booking_log(n_days=30, n_listings=2)
    main_path = tmp_path / "bookings.csv"
    main.to_csv(main_path, index=False)

    codes = sorted(main["کد_اقامتگاه"].unique())
    # The same listing twice - a real export hazard, and a silent row explosion.
    duplicated = pd.DataFrame({"کد_اقامتگاه": codes + codes, "ظرفیت": [4, 6, 40, 60]})
    side_path = tmp_path / "dupes.csv"
    duplicated.to_csv(side_path, index=False)

    contract = _count_contract(main_path)
    contract.joins = [JoinSpec(path=str(side_path), on="کد_اقامتگاه")]

    adapter = DataAdapter(contract)
    raw = adapter.load_raw()
    assert len(raw) == len(main), "a join must never add rows to the main frame"
    assert any("duplicate row" in note for note in adapter.join_notes)


def test_a_missing_side_file_is_a_note_not_a_crash(tmp_path):
    main_path = tmp_path / "bookings.csv"
    _booking_log(n_days=30, n_listings=2).to_csv(main_path, index=False)

    contract = _count_contract(main_path)
    contract.joins = [JoinSpec(path=str(tmp_path / "nope.csv"), on="کد_اقامتگاه")]

    panel = DataAdapter(contract).build()
    assert any("file not found" in note for note in panel.notes)
