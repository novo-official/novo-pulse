"""Realistic synthetic dataset for the Iranian stay/tourism market.

Demand is *not* random noise. It is drawn from an explicit latent demand
function so that every effect the models are asked to discover genuinely exists
in the data:

    latent_demand = base
                  x trend
                  x weekly_seasonality
                  x yearly_seasonality
                  x holiday_effect
                  x destination_effect
                  x accommodation_effect
                  x event_effect
                  x price_elasticity
                  x promotion_effect
                  + noise

Observed bookings are then *censored* by the available capacity:

    observed_bookings = min(latent_demand, available_capacity)

which reproduces the real-world problem that a sold-out listing understates
true market demand.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import SYNTHETIC_DATA_DIR
from .holidays import build_calendar

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Market definition
# --------------------------------------------------------------------------
# name, province, type, popularity, winter_pull, summer_pull, weekend_pull,
# holiday_pull
DESTINATIONS: list[tuple[str, str, str, str, float, float, float, float, float]] = [
    ("tehran", "تهران", "Tehran", "metro", 1.35, 1.00, 0.95, 0.90, 0.85),
    ("mashhad", "مشهد", "Khorasan Razavi", "pilgrimage", 1.30, 1.05, 1.00, 1.05, 1.75),
    ("shiraz", "شیراز", "Fars", "historical", 1.05, 1.00, 0.90, 1.15, 1.35),
    ("isfahan", "اصفهان", "Isfahan", "historical", 1.10, 0.98, 0.92, 1.18, 1.40),
    ("kish", "کیش", "Hormozgan", "island", 1.00, 1.55, 0.62, 1.25, 1.45),
    ("qeshm", "قشم", "Hormozgan", "island", 0.80, 1.45, 0.68, 1.20, 1.30),
    ("rasht", "رشت", "Gilan", "north", 0.85, 0.85, 1.30, 1.35, 1.20),
    ("ramsar", "رامسر", "Mazandaran", "north", 0.75, 0.80, 1.40, 1.45, 1.25),
    ("tabriz", "تبریز", "East Azerbaijan", "historical", 0.70, 0.90, 1.10, 1.05, 1.15),
    ("yazd", "یزد", "Yazd", "desert", 0.65, 1.20, 0.75, 1.10, 1.25),
    ("chabahar", "چابهار", "Sistan & Baluchestan", "coastal", 0.45, 1.25, 0.70, 1.05, 1.10),
    ("bandar_abbas", "بندرعباس", "Hormozgan", "coastal", 0.55, 1.30, 0.65, 1.00, 1.05),
]

# category, share, capacity range, price multiplier, weekend sensitivity,
# price elasticity, seasonality amplitude
CATEGORIES: list[tuple[str, float, tuple[int, int], float, float, float, float]] = [
    ("hotel", 0.30, (18, 90), 1.00, 0.85, -1.10, 0.55),
    ("apartment", 0.22, (2, 8), 0.72, 1.05, -1.45, 0.75),
    ("villa", 0.20, (4, 16), 1.35, 1.45, -1.25, 0.95),
    ("eco_lodge", 0.13, (3, 12), 0.62, 1.25, -1.60, 1.05),
    ("boutique_hotel", 0.09, (6, 24), 1.20, 1.00, -1.20, 0.70),
    ("suite", 0.06, (2, 6), 0.85, 1.10, -1.40, 0.80),
]

CATEGORY_FA = {
    "hotel": "هتل",
    "apartment": "آپارتمان",
    "villa": "ویلا",
    "eco_lodge": "اقامتگاه بوم‌گردی",
    "boutique_hotel": "هتل بوتیک",
    "suite": "سوئیت",
}


@dataclass
class SyntheticConfig:
    days: int = 900
    end_date: dt.date | None = None
    n_accommodations: int = 140
    seed: int = RANDOM_SEED
    output_dir: Path | None = None
    noise_scale: float = 0.16
    promotion_rate: float = 0.09


# --------------------------------------------------------------------------
def generate(config: SyntheticConfig | None = None) -> dict[str, pd.DataFrame]:
    """Generate the full synthetic warehouse and return it as DataFrames."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    end = cfg.end_date or (dt.date.today() - dt.timedelta(days=1))
    start = end - dt.timedelta(days=cfg.days - 1)

    destinations = _build_destinations()
    accommodations = _build_accommodations(cfg, destinations, rng)
    calendar = build_calendar(start, end)
    demand = _build_daily_demand(cfg, accommodations, destinations, calendar, rng)
    bookings = _build_bookings(demand, rng)
    searches = _build_searches(demand)

    return {
        "destinations": destinations,
        "accommodations": accommodations,
        "calendar": calendar,
        "daily_demand": demand,
        "bookings": bookings,
        "searches": searches,
    }


def generate_and_save(config: SyntheticConfig | None = None) -> dict[str, Path]:
    cfg = config or SyntheticConfig()
    out_dir = cfg.output_dir or SYNTHETIC_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = generate(cfg)
    written: dict[str, Path] = {}
    for name, frame in tables.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
    return written


# --------------------------------------------------------------------------
def _build_destinations() -> pd.DataFrame:
    rows = []
    for key, name_fa, province, dtype, pop, winter, summer, weekend, holiday in DESTINATIONS:
        rows.append(
            {
                "destination_id": key,
                "destination_name": name_fa,
                "destination_name_en": key.replace("_", " ").title(),
                "province": province,
                "destination_type": dtype,
                "popularity_index": pop,
                "winter_pull": winter,
                "summer_pull": summer,
                "weekend_pull": weekend,
                "holiday_pull": holiday,
            }
        )
    return pd.DataFrame(rows)


def _build_accommodations(
    cfg: SyntheticConfig, destinations: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    weights = destinations["popularity_index"].to_numpy()
    weights = weights / weights.sum()
    cat_names = [c[0] for c in CATEGORIES]
    cat_share = np.array([c[1] for c in CATEGORIES])
    cat_share = cat_share / cat_share.sum()
    cat_lookup = {c[0]: c for c in CATEGORIES}

    rows = []
    for i in range(cfg.n_accommodations):
        dest_idx = rng.choice(len(destinations), p=weights)
        dest = destinations.iloc[dest_idx]
        category = str(rng.choice(cat_names, p=cat_share))
        _, _, cap_range, price_mult, weekend_sens, elasticity, season_amp = cat_lookup[category]

        capacity = int(rng.integers(cap_range[0], cap_range[1] + 1))
        bedrooms = max(1, int(np.ceil(capacity / rng.uniform(1.8, 3.2))))
        rating = float(np.clip(rng.normal(4.2, 0.45), 2.6, 5.0).round(1))
        quality = (rating - 2.6) / 2.4  # 0..1

        base_price = float(
            np.round(
                rng.lognormal(mean=np.log(1_650_000), sigma=0.35)
                * price_mult
                * (0.75 + 0.55 * quality)
                * (0.8 + 0.4 * dest["popularity_index"]),
                -4,
            )
        )
        # Listings enter the marketplace at different times -> cold-start cases.
        onboarding_offset = int(rng.choice([0, 0, 0, 0, 0, 30, 90, 180, 400, 700], size=1)[0])

        rows.append(
            {
                "accommodation_id": f"acc_{i:04d}",
                "destination_id": dest["destination_id"],
                "category": category,
                "category_fa": CATEGORY_FA[category],
                "capacity": capacity,
                "bedrooms": bedrooms,
                "rating": rating,
                "base_price": base_price,
                "quality_index": round(quality, 3),
                "weekend_sensitivity": weekend_sens,
                "price_elasticity": round(elasticity * rng.uniform(0.85, 1.15), 3),
                "season_amplitude": round(season_amp * rng.uniform(0.85, 1.15), 3),
                "intrinsic_appeal": round(float(rng.lognormal(0.0, 0.28)), 4),
                "onboarding_offset_days": onboarding_offset,
            }
        )
    return pd.DataFrame(rows)


def _build_daily_demand(
    cfg: SyntheticConfig,
    accommodations: pd.DataFrame,
    destinations: pd.DataFrame,
    calendar: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    dest_meta = destinations.set_index("destination_id")
    n_days = len(calendar)
    day_index = np.arange(n_days)
    dates = calendar["date"].to_numpy()
    doy = calendar["date"].dt.dayofyear.to_numpy()
    dow = calendar["day_of_week"].to_numpy()
    is_weekend = calendar["is_weekend"].to_numpy()
    is_holiday = calendar["is_holiday"].to_numpy()
    days_to_holiday = calendar["days_to_holiday"].to_numpy()
    event_dests = calendar["event_destinations"].to_numpy()

    # Market-wide upward trend plus a slow macro cycle.
    trend = 1.0 + 0.00042 * day_index
    macro = 1.0 + 0.06 * np.sin(2 * np.pi * day_index / 365.25 + 0.7)

    # Weekly shape: Wed-Fri are the travel peak in Iran (Mon=0).
    weekly_base = np.array([0.82, 0.80, 0.92, 1.32, 1.45, 1.05, 0.88])

    frames = []
    for row in accommodations.itertuples(index=False):
        dest = dest_meta.loc[row.destination_id]

        # ---- yearly seasonality, destination flavoured -------------------
        winter_weight = 0.5 * (1 + np.cos(2 * np.pi * (doy - 15) / 365.25))
        summer_weight = 1.0 - winter_weight
        seasonal_pull = winter_weight * dest["winter_pull"] + summer_weight * dest["summer_pull"]
        yearly = 1.0 + row.season_amplitude * 0.90 * (seasonal_pull - 1.0)

        # Northern destinations spike in late spring/summer weekends.
        if dest["destination_type"] == "north":
            yearly *= 1.0 + 0.18 * np.exp(-((doy - 200) ** 2) / (2 * 45.0**2))
        if dest["destination_type"] == "island":
            yearly *= 1.0 + 0.22 * np.exp(-((((doy + 182) % 365) - 182) ** 2) / (2 * 55.0**2))

        # ---- weekly ------------------------------------------------------
        weekly = 1.0 + row.weekend_sensitivity * (weekly_base[dow] - 1.0)
        weekly *= 1.0 + 0.12 * is_weekend * (dest["weekend_pull"] - 1.0) * 5

        # ---- holidays and events ----------------------------------------
        holiday = 1.0 + is_holiday * (dest["holiday_pull"] - 1.0)
        # Anticipation ramp: demand builds in the days before a holiday.
        holiday *= 1.0 + 0.35 * np.exp(-days_to_holiday / 4.0) * (dest["holiday_pull"] - 1.0)
        has_event = np.array(
            [row.destination_id in (e.split("|") if e else []) for e in event_dests],
            dtype=float,
        )
        event = 1.0 + 0.34 * has_event

        # ---- price path --------------------------------------------------
        # Operators raise prices into peak season and on weekends.
        expected_pressure = yearly * weekly * holiday
        price_noise = rng.normal(0, 0.045, n_days)
        # Operators track demand pressure but only partially - if price moved
        # one-for-one with demand the elasticity signal would cancel itself out.
        price_factor = 0.88 + 0.14 * np.clip(expected_pressure, 0.4, 2.6) + price_noise
        promo = (rng.random(n_days) < cfg.promotion_rate).astype(float)
        # Promotions come with a real discount.
        discount = promo * rng.uniform(0.10, 0.30, n_days)
        price = np.round(row.base_price * price_factor * (1 - discount), -3)
        price = np.maximum(price, row.base_price * 0.45)

        relative_price = price / row.base_price
        price_effect = np.power(np.maximum(relative_price, 0.2), row.price_elasticity)
        promotion_effect = 1.0 + 0.22 * promo

        # ---- assemble latent demand -------------------------------------
        base = 0.26 * row.capacity * dest["popularity_index"] * row.intrinsic_appeal
        base *= 0.7 + 0.6 * row.quality_index
        latent = (
            base
            * trend
            * macro
            * yearly
            * weekly
            * holiday
            * event
            * price_effect
            * promotion_effect
        )
        latent = latent * np.exp(rng.normal(0, cfg.noise_scale, n_days) - cfg.noise_scale**2 / 2)
        latent = np.maximum(latent, 0.0)

        # ---- supply side + censoring ------------------------------------
        # Availability shrinks when demand has been running hot.
        availability_ratio = np.clip(rng.beta(6.0, 2.2, n_days), 0.25, 1.0)
        available_capacity = np.maximum(1, np.round(row.capacity * availability_ratio))
        observed = np.minimum(latent, available_capacity)
        bookings_obs = rng.poisson(np.maximum(observed, 0.01))
        bookings_obs = np.minimum(bookings_obs, available_capacity).astype(int)

        # ---- funnel ------------------------------------------------------
        # Searches lead bookings: they reflect latent (uncensored) interest.
        search = rng.poisson(np.maximum(latent * rng.uniform(7.5, 12.5), 0.5))
        view = rng.binomial(search, np.clip(rng.normal(0.34, 0.05), 0.08, 0.85))

        length_of_stay = np.round(np.clip(rng.normal(2.4, 0.9, n_days), 1, 14), 1)
        reserved_nights = np.round(bookings_obs * length_of_stay).astype(int)
        lead_time = np.round(np.clip(rng.normal(18, 9, n_days), 0, 120)).astype(int)

        active = day_index >= row.onboarding_offset_days
        frame = pd.DataFrame(
            {
                "date": dates,
                "accommodation_id": row.accommodation_id,
                "destination_id": row.destination_id,
                "category": row.category,
                "price": price,
                "base_price": row.base_price,
                "relative_price": np.round(relative_price, 4),
                "discount_percent": np.round(discount * 100, 2),
                "is_promotion": promo.astype(int),
                # Festivals are published months ahead, so this is legitimately
                # a known-future covariate rather than leakage.
                "is_event": has_event.astype(int),
                "capacity": row.capacity,
                "available_capacity": available_capacity.astype(int),
                "search_count": search,
                "view_count": view,
                "booking_count": bookings_obs,
                "reserved_nights": reserved_nights,
                "avg_lead_time": lead_time,
                "avg_length_of_stay": length_of_stay,
                "latent_demand": np.round(latent, 3),
                "is_censored": (latent > available_capacity).astype(int),
            }
        )
        frames.append(frame.loc[active])

    demand = pd.concat(frames, ignore_index=True)
    demand["occupancy_rate"] = np.round(
        demand["booking_count"] / demand["available_capacity"].clip(lower=1), 4
    )
    demand["revenue"] = (demand["booking_count"] * demand["price"]).astype("int64")
    demand["conversion_rate"] = np.round(
        demand["booking_count"] / demand["search_count"].clip(lower=1), 5
    )
    return demand.sort_values(["date", "accommodation_id"]).reset_index(drop=True)


def _build_bookings(demand: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Explode the daily aggregate into individual booking records."""
    positive = demand.loc[demand["booking_count"] > 0]
    # Keep the transaction table to a sane size for a demo repository.
    if len(positive) > 60_000:
        positive = positive.sample(60_000, random_state=RANDOM_SEED).sort_values("date")
    repeats = positive["booking_count"].to_numpy()
    idx = np.repeat(positive.index.to_numpy(), repeats)
    expanded = demand.loc[idx]
    n = len(expanded)
    lead = rng.integers(0, 90, n)
    nights = np.maximum(1, rng.poisson(2.2, n))
    return pd.DataFrame(
        {
            "booking_id": [f"bk_{i:07d}" for i in range(n)],
            "booking_date": expanded["date"].to_numpy(),
            "checkin_date": expanded["date"].to_numpy() + pd.to_timedelta(lead, unit="D"),
            "accommodation_id": expanded["accommodation_id"].to_numpy(),
            "destination_id": expanded["destination_id"].to_numpy(),
            "nights": nights,
            "guests": np.maximum(1, rng.poisson(2.6, n)),
            "price_per_night": expanded["price"].to_numpy(),
            "lead_time_days": lead,
        }
    )


def _build_searches(demand: pd.DataFrame) -> pd.DataFrame:
    """Destination-level daily search/view/booking funnel."""
    grouped = (
        demand.groupby(["date", "destination_id"], as_index=False)
        .agg(
            search_count=("search_count", "sum"),
            view_count=("view_count", "sum"),
            booking_count=("booking_count", "sum"),
        )
    )
    grouped["conversion_rate"] = (
        grouped["booking_count"] / grouped["search_count"].clip(lower=1)
    ).round(5)
    return grouped
