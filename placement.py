"""Anchoring against receipt placement, as a two-way design.

Two things differ between F&O's *Per period* and SAP's period lot sizing, and
they are usually discussed as if they were one thing.

  ANCHORING. Where the bucket boundary sits. SAP "groups several requirements
  within a time interval together to form a lot", the interval being a day, a
  week, a month, a posting period or a planning calendar entry. F&O's period
  "starts with the first demand of the item", so its boundary floats.

  PLACEMENT. Where the receipt sits inside the bucket. SAP "sets the
  availability date for period lot-sizing procedures to the first requirements
  date of the period. However, you can also define that the availability date
  is at the beginning or end of the period." F&O has one position: "The order
  is planned for the first day of the period."

An earlier version of the companion Infor LN analysis compared a
calendar-anchored bucket with a receipt on the boundary against a floating
bucket with a receipt at first demand, and concluded anchoring was expensive.
That comparison moved both variables at once, so it could not say which one
was responsible. This module moves them one at a time.

The design is 2 x 3, minus the cell that does not exist:

                    first requirement   period start   period end
    floating              yes           same as first      yes
    anchored              yes               yes            yes

A floating bucket opens at the first demand, so its start IS its first
requirement and the two coincide. That cell is run once and labelled.

H2 reads down a placement column: anchored against floating at the same
placement.
H3 reads across an anchoring row: the three placements at the same anchoring.

Placement is swept across every weekday anchor as well, because a weekly bucket
that starts on the wrong day of the week is a different rule, and the firm in
this data does not trade on Saturdays.
"""
import numpy as np
import pandas as pd

import config as C
import data as D
import engine as E
import rolling as R
import sap as S

FLOATING, ANCHORED = "floating", "anchored"


def configs(buckets=None, anchors=(0,)):
    """The two-way grid.

    A floating bucket at PERIOD_START is not generated: it is the same rule as
    a floating bucket at FIRST_REQUIREMENT, because the bucket opens at the
    first demand. Generating it would put a duplicate column in the design and
    make the placement effect look smaller than it is.
    """
    buckets = C.SAP_BUCKETS if buckets is None else buckets
    out = []
    for L in buckets:
        out.append({"name": f"floating {L}d first", "kind": "float", "L": L,
                    "anchoring": FLOATING, "placement": S.FIRST_REQUIREMENT,
                    "anchor": 0, "who": "Both"})
        for anchor in anchors:
            for pl in S.PLACEMENTS:
                out.append({"name": f"anchored {L}d {pl} a{anchor}",
                            "kind": "sap_cal", "L": L, "placement": pl,
                            "anchor": anchor, "anchoring": ANCHORED,
                            "who": "SAP only"})
    return out


def run(daily, profile, prices, case="Seasonal wholesaler", anchors=(0,),
        log=print):
    """Replay the grid in one business case.

    One case, not five. The question is about the rule's mechanics rather than
    about the business, and holding the lead time and forecast error fixed is
    what lets the two-way comparison be read as a two-way comparison.
    """
    spec = C.CASES[case]
    sc = {"lead_time": spec["lead_time"], "replan_every": C.REPLAN_EVERY,
          "visibility": C.VISIBILITY_DAYS, "noise": spec["noise"],
          "opening_stock_days": C.OPENING_STOCK_DAYS, "coverage_fence": None}
    pool = D.sample(profile, daily)
    pool = pool[pool["StockCode"].isin(prices.index)]
    rhythm = weekday_rhythm(daily, pool["StockCode"])
    cfgs = configs(anchors=anchors)
    rows = []
    for _, it in pool.iterrows():
        code = it["StockCode"]
        a = D.series(daily, code)
        if a.mean() <= 0:
            continue
        price = float(prices[code])
        h = S.holding_per_unit_day(price, C.STORAGE_PCT)
        for cfg in cfgs:
            m = R.simulate(a, cfg, sc,
                           seed=R.stable_seed(code, cfg["name"], case))
            rows.append({"code": code, "quadrant": it["quadrant"],
                         "config": cfg["name"], "L": cfg["L"],
                         "anchoring": cfg["anchoring"],
                         "placement": cfg["placement"],
                         "anchor": cfg["anchor"],
                         "rhythm": float(rhythm[code]),
                         "cost": C.ORDER_COST * m["n_orders"] + h * m["unit_days"],
                         **m})
        log(f"  {code}") if False else None
    return pd.DataFrame(rows)


# --------------------------------------------------------------- the rhythm

def weekday_rhythm(daily, codes):
    """How concentrated an item's demand is on one weekday.

    The share of an item's total demand falling on its own busiest weekday.
    Flat across the six days this firm trades would be 1/6 = 0.167; everything
    on one day would be 1.0.

    This is the variable H2 is really about. A calendar bucket and a floating
    window of the same length are the same object on demand with no weekly
    rhythm, and different objects on demand with one. Splitting the catalogue
    on a measured rhythm isolates that, where comparing two different firms
    would confound it with everything else about the second firm.
    """
    sub = daily[[c for c in codes if c in daily.columns]]
    g = sub.groupby(sub.index.dayofweek).sum()
    tot = g.sum()
    share = (g.max() / tot.replace(0, np.nan))
    return share.fillna(0.0)


# ------------------------------------------------------------- the readings

def effect_of_anchoring(raw):
    """H2. Anchored against floating, holding placement at first requirement.

    Both arms place the receipt at the first requirement inside the bucket, so
    the only thing that differs is where the boundary falls.
    """
    sub = raw[raw["placement"] == S.FIRST_REQUIREMENT]
    piv = sub.pivot_table(index=["L", "code"], columns="anchoring",
                          values=["cost", "fill", "cover_days",
                                  "orders_per_year"])
    out = []
    for L, g in piv.groupby(level="L"):
        row = {"L": L}
        for metric in ("cost", "fill", "cover_days", "orders_per_year"):
            a = g[(metric, ANCHORED)]
            f = g[(metric, FLOATING)]
            row[f"{metric}_floating"] = float(f.median())
            row[f"{metric}_anchored"] = float(a.median())
            row[f"{metric}_pct"] = 100.0 * float((a - f).median() / f.median())
        out.append(row)
    return pd.DataFrame(out)


def effect_of_placement(raw):
    """H3. The three receipt positions, holding the bucket anchored.

    Read against `first`, which is the position SAP ships by default and the
    only position F&O offers.
    """
    sub = raw[raw["anchoring"] == ANCHORED]
    piv = sub.pivot_table(index=["L", "code"], columns="placement",
                          values=["cost", "fill", "cover_days",
                                  "orders_per_year"])
    out = []
    for L, g in piv.groupby(level="L"):
        for pl in (S.PERIOD_START, S.PERIOD_END):
            row = {"L": L, "placement": pl}
            for metric in ("cost", "fill", "cover_days", "orders_per_year"):
                base = g[(metric, S.FIRST_REQUIREMENT)]
                alt = g[(metric, pl)]
                row[f"{metric}_first"] = float(base.median())
                row[f"{metric}_{pl}"] = float(alt.median())
                row[f"{metric}_pct"] = 100.0 * float(
                    (alt - base).median() / base.median())
            out.append(row)
    return pd.DataFrame(out)


def anchoring_by_rhythm(raw, bins=(0.167, 0.25, 0.30, 0.40, 1.01)):
    """H2, against the variable it should depend on.

    If anchoring only matters where demand carries a calendar rhythm, the
    anchored-minus-floating gap should widen across these bins. If it is flat,
    anchoring is not about rhythm and the hypothesis is wrong.
    """
    sub = raw[raw["placement"] == S.FIRST_REQUIREMENT].copy()
    sub["band"] = pd.cut(sub["rhythm"], bins=list(bins), include_lowest=True)
    piv = sub.pivot_table(index=["band", "L", "code"], columns="anchoring",
                          values="cost", observed=True)
    piv = piv.dropna()
    rel = (piv[ANCHORED] - piv[FLOATING]) / piv[FLOATING] * 100.0
    return rel.groupby(level=["band", "L"], observed=True).median().unstack("L")
