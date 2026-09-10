"""Turn a cluster assignment into a `Pol4Data` of virtual cities.

This is the whole architectural decision in one module. Clustering is a
*data-aggregation step*, not a second model: it collapses a group of sparse
cities into one pseudo-city carrying a `cluster_code` where a `city_code` used
to be, and every module downstream - `pickup.py`, `baseline.py`, `features.py`,
`dataset.py`, `champion.py`, `backtest.py` - runs on the result without knowing
whether a row is one real city or five merged ones.

The alternative, a bespoke cluster model with its own feature logic, would
confound the only question worth asking. If the clustered arm scored better,
there would be no way to tell whether clustering helped or whether the second
pipeline was simply built better. Here both arms are the same code, and the
identity assignment reproduces the original panel byte for byte - which
`tests/test_pol4_clustering.py` asserts.

**Why the raw log is summed rather than the features averaged.** A feature is
only aggregatable if it is linear in the log, and most of them are not:

    observed_total, pickup_*d        sum the member logs, then read the feature
    city_hist_std/p75/p90/volatility recompute from the pooled daily series
    city_weekend_ratio               recompute - a cluster's weekly shape is not
                                     the mean of its members' weekly shapes
    max_daily_search                 recompute - five bursty cities whose bursts
                                     fall on different days pool to a smoother
                                     series, while a shared event (Nowruz) pools
                                     to a sharper one
    market_*                         unchanged; already in-panel
    province_*                       unchanged; clusters never cross a province
    lat / long                       volume-weighted centroid of the members

Summing the log first and recomputing everything from it gets all of those
right by construction, because none of them is ever computed twice.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .clustering import ClusterAssignment
from .loader import (
    CHECKIN,
    CITY,
    CITY_NAME,
    DTC,
    LOG_DATE,
    PROVINCE,
    PROVINCE_NAME,
    SEARCHES,
    Pol4Data,
)

#: The virtual-city attributes `features.py` reads for the cluster feature group.
CLUSTER_ATTRIBUTES = (
    "n_cities_in_cluster",
    "cluster_min_member_volume",
    "cluster_max_member_share",
)


def aggregate_events(events: pd.DataFrame, mapping: pd.Series) -> pd.DataFrame:
    """Sum a search log over cluster members, keeping the two time axes intact.

    Grouping on `(log_date, cluster_code, checkin)` preserves exactly the
    structure `loader.py` guarantees - one row per pair per log date - so the
    pair tensor built from this frame is the pooled daily series, and every
    pickup, velocity and acceleration feature is read off it rather than
    averaged from the members' own values.
    """
    if events.empty:
        return events.assign(**{CITY: events[CITY]})

    clusters = mapping.reindex(events[CITY].to_numpy()).to_numpy()
    if pd.isna(clusters).any():
        missing = sorted(set(events[CITY].to_numpy()[pd.isna(clusters)]))[:5]
        raise ValueError(f"cluster assignment does not cover city_code(s) {missing}")

    out = (
        events.assign(**{CITY: clusters.astype("int32")})
        .groupby([LOG_DATE, CITY, CHECKIN], as_index=False, observed=True)[SEARCHES]
        .sum()
    )
    out[DTC] = (out[CHECKIN] - out[LOG_DATE]).dt.days.astype("int16")
    return out.loc[:, [LOG_DATE, CITY, CHECKIN, SEARCHES, DTC]]


def _cluster_names(data: Pol4Data, assignment: ClusterAssignment) -> pd.DataFrame:
    """Readable labels for virtual cities: the largest member, plus a count."""
    members = assignment.frame.sort_values(["cluster_code", "volume"], ascending=[True, False])
    names = data.city_name_of.reindex(members[CITY].to_numpy()).to_numpy()
    members = members.assign(**{CITY_NAME: names})
    lead = members.groupby("cluster_code", sort=True).first()
    sizes = members.groupby("cluster_code", sort=True).size()
    label = np.where(
        sizes.to_numpy() > 1,
        lead[CITY_NAME].astype(str).to_numpy() + " +" + (sizes.to_numpy() - 1).astype(str),
        lead[CITY_NAME].astype(str).to_numpy(),
    )
    province_names = data.province_name_of.reindex(lead[CITY].to_numpy()).to_numpy()
    return pd.DataFrame(
        {
            "cluster_code": lead.index.to_numpy(),
            CITY_NAME: label,
            PROVINCE_NAME: province_names,
        }
    )


def aggregate_data(data: Pol4Data, assignment: ClusterAssignment) -> Pol4Data:
    """The competition datasets re-expressed over the assignment's virtual cities.

    The returned object is an ordinary `Pol4Data`: `city_code` holds the
    `cluster_code`, `cities` holds one row per submitted group with its centroid
    and its cluster-shape attributes, and every method - `history_before`,
    `observed_at`, `final_demand`, `province_of` - answers at the panel grain
    the submission is scored on.
    """
    mapping = assignment.mapping()
    unknown = sorted(set(data.cities[CITY]) - set(mapping.index))
    if unknown:
        raise ValueError(f"cluster assignment is missing city_code(s) {unknown[:5]}")

    clusters = assignment.clusters.rename(columns={"cluster_code": CITY})
    columns = [CITY, PROVINCE, "lat", "long", *CLUSTER_ATTRIBUTES]
    cities = clusters.loc[:, columns].copy()
    cities[CITY] = cities[CITY].astype("int32")
    cities[PROVINCE] = cities[PROVINCE].astype("int32")

    if data.has_names:
        names = _cluster_names(data, assignment).rename(columns={"cluster_code": CITY})
        cities = cities.merge(names, on=CITY, how="left")

    if assignment.is_identity:
        # Nothing to sum: keep the original frames so the identity arm is
        # provably the untouched panel, not a re-derivation of it.
        search, evaluation = data.search, data.evaluation
    else:
        search = aggregate_events(data.search, mapping)
        evaluation = aggregate_events(data.evaluation, mapping)

    return Pol4Data(
        search=search,
        evaluation=evaluation,
        cities=cities.sort_values(CITY, ignore_index=True),
        findings=list(data.findings),
    )


def city_volume_shares(assignment: ClusterAssignment) -> pd.Series:
    """Each city's share of its cluster's historical volume, from train data only.

    Used to push a cluster-level prediction back down to its member cities for
    the shrinkage blend. Members of a cluster with no history at all split it
    evenly, which is the only defensible answer when nothing distinguishes them.
    """
    frame = assignment.frame
    totals = frame.groupby("cluster_code")["volume"].transform("sum").to_numpy()
    sizes = frame.groupby("cluster_code")["volume"].transform("size").to_numpy()
    volumes = frame["volume"].to_numpy(dtype=np.float64)
    share = np.where(totals > 0, volumes / np.maximum(totals, 1e-9), 1.0 / sizes)
    return pd.Series(share, index=pd.Index(frame[CITY].to_numpy(), name=CITY), name="share")
