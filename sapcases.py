"""The rolling test: F&O coverage codes against SAP lot-sizing procedures.

Section 3 of this repository's static work showed SAP's Part Period Balancing
within 9 per cent of the Wagner and Whitin optimum where the best single F&O
coverage period was 43 to 51 per cent above it. That was computed under perfect
foresight in a single pass, which is the condition the lot-sizing literature
says flatters an optimising heuristic.

Blackburn and Millen (1980) found that under a rolling schedule the simpler
Silver-Meal heuristic can beat the Wagner and Whitin algorithm outright, so
being close to the optimum in one pass predicts very little. Wemmerlov (1989)
found forecast error compresses the differences between procedures further.

This module runs the same rules under rolling regeneration with forecast error,
which is what an implementation actually does, and reports the physical
outcomes alongside the cost the optimum procedures are minimising.

Two things are measured that the static work could not measure:

  - service, because a rule that holds less stock can pay for it in fill rate,
    and cost alone hides that;
  - what happens when the assumed order cost is wrong. F&O's coverage period
    never reads a cost, so it cannot be wrong about one. SAP's optimum
    procedures read three material master fields, and two of them are
    estimates. That asymmetry is the point of run_misspecified().
"""
import numpy as np
import pandas as pd

import config as C
import data as D
import rolling as R
import sap as S

# Which product documents each rule. The article's mapping table is built from
# this column, so it is data rather than prose.
FNO, SAPO, BOTH = "F&O only", "SAP only", "Both"


def configs(price, K=None, storage_pct=None):
    """Every rule under test, for one item at one price.

    Price enters because SAP's optimum procedures cost storage as a percentage
    of the material price. It is the item's own median unit price from the
    order book, so the setup-to-holding ratio varies across the catalogue the
    way it does in a real material master.
    """
    K = C.ORDER_COST if K is None else K
    pct = C.STORAGE_PCT if storage_pct is None else storage_pct

    c = [{"name": "Per requirement", "kind": "l4l", "L": 1, "who": BOTH}]
    for L in C.PERIODS:
        c.append({"name": f"Per period {L}d", "kind": "float", "L": L,
                  "who": BOTH})
    c.append({"name": "Min/Max", "kind": "minmax", "L": 1, "who": BOTH})

    # Optimum lot sizing. No F&O coverage code reads a cost.
    for code in ("SP", "WI", "DY", "GR"):
        c.append({"name": f"{code} {S.OPTIMUM_NAMES[code]}", "kind": "sap_opt",
                  "code": code, "price": price, "storage_pct": pct, "K": K,
                  "L": 1, "who": SAPO})

    # Calendar buckets, at the receipt position SAP ships by default. The other
    # two positions are the subject of placement.py and are not run here, so
    # this table stays readable.
    for L in (7, 28):
        c.append({"name": f"Calendar bucket {L}d", "kind": "sap_cal", "L": L,
                  "placement": S.FIRST_REQUIREMENT, "who": SAPO})

    # Short-term and long-term lot size. A fine bucket near, a coarse one far.
    for far in C.SHORT_LONG_FAR:
        c.append({"name": f"Short 7d / long {far}d", "kind": "sap_sl",
                  "near": {"kind": "sap_cal", "L": 7},
                  "far": {"kind": "sap_cal", "L": far},
                  "periods": C.SHORT_LONG_PERIODS, "L": 7, "who": SAPO})
    return c


def _bind(cfg, lead_time, mean_daily):
    """Fill in the per-item parameters a rule needs before it can run."""
    cfg = dict(cfg)
    if cfg["kind"] == "minmax":
        # The minimum must cover the lead time or the rule is a strawman.
        cfg["mn"] = lead_time * mean_daily
        cfg["mx"] = (lead_time + 28) * mean_daily
    return cfg


def realised_cost(m, price, K, storage_pct=None):
    """Setup plus holding actually incurred over the measured window.

    This is the quantity SAP's optimum procedures exist to minimise, so it is
    what they have to be judged on. Holding is priced off the item's own unit
    price, the same basis the procedures use to choose their lots.
    """
    pct = C.STORAGE_PCT if storage_pct is None else storage_pct
    h = S.holding_per_unit_day(price, pct)
    return K * m["n_orders"] + h * m["unit_days"]


def run(daily, profile, prices, log=print):
    """Every rule in every business case, under rolling regeneration."""
    pool = D.sample(profile, daily)
    pool = pool[pool["StockCode"].isin(prices.index)]
    rows = []
    for case, spec in C.CASES.items():
        items = pool if spec["classes"] is None else \
            pool[pool["quadrant"].isin(spec["classes"])]
        sc = {"lead_time": spec["lead_time"], "replan_every": C.REPLAN_EVERY,
              "visibility": C.VISIBILITY_DAYS, "noise": spec["noise"],
              "opening_stock_days": C.OPENING_STOCK_DAYS,
              "coverage_fence": None}
        for _, it in items.iterrows():
            code = it["StockCode"]
            a = D.series(daily, code)
            md = float(a.mean())
            if md <= 0:
                continue
            price = float(prices[code])
            for base in configs(price):
                cfg = _bind(base, spec["lead_time"], md)
                m = R.simulate(a, cfg, sc,
                               seed=R.stable_seed(code, cfg["name"], case))
                rows.append({"case": case, "config": cfg["name"],
                             "who": cfg["who"], "code": code, "price": price,
                             "quadrant": it["quadrant"],
                             "cost": realised_cost(m, price, C.ORDER_COST),
                             **m})
        log(f"  {case}: {len(items)} items")
    return pd.DataFrame(rows)


def run_misspecified(daily, profile, prices, log=print):
    """What it costs SAP's optimum procedures to be given the wrong order cost.

    The rule is run believing the order cost is `assumed`, and the outcome is
    then priced at the `true` cost. F&O's coverage period is run unchanged in
    the same conditions, because it never reads a cost and therefore cannot be
    misinformed about one. That asymmetry is the finding this function exists
    to produce, in either direction.

    One business case is used rather than five, because the question is about
    the cost inputs and not about the business.
    """
    pool = D.sample(profile, daily)
    pool = pool[pool["StockCode"].isin(prices.index)]
    spec = C.CASES["Seasonal wholesaler"]
    sc = {"lead_time": spec["lead_time"], "replan_every": C.REPLAN_EVERY,
          "visibility": C.VISIBILITY_DAYS, "noise": spec["noise"],
          "opening_stock_days": C.OPENING_STOCK_DAYS, "coverage_fence": None}
    true_K = C.ORDER_COST
    factors = (1.0,) + tuple(C.MISSPECIFY)
    rows = []
    for _, it in pool.iterrows():
        code = it["StockCode"]
        a = D.series(daily, code)
        md = float(a.mean())
        if md <= 0:
            continue
        price = float(prices[code])
        for f in factors:
            assumed = true_K * f
            for cfgs in (configs(price, K=assumed),):
                for base in cfgs:
                    if base["who"] == SAPO and base["kind"] != "sap_opt":
                        continue          # only the cost-reading rules
                    if base["kind"] not in ("sap_opt", "float"):
                        continue
                    cfg = _bind(base, spec["lead_time"], md)
                    m = R.simulate(a, cfg, sc,
                                   seed=R.stable_seed(code, cfg["name"], "miss"))
                    rows.append({"factor": f, "assumed_K": assumed,
                                 "config": cfg["name"], "who": cfg["who"],
                                 "code": code,
                                 "cost": realised_cost(m, price, true_K),
                                 **m})
        log(f"  misspecification: {code}") if False else None
    return pd.DataFrame(rows)


def summarise(raw):
    """Per case and rule: the physical outcomes and the cost, averaged."""
    return raw.groupby(["case", "config", "who"], as_index=False).agg(
        fill=("fill", "mean"),
        cover_days=("cover_days", "mean"),
        orders_per_year=("orders_per_year", "mean"),
        cost=("cost", "mean"))


def cost_index(agg):
    """Cost as a multiple of the cheapest rule in the same business case."""
    out = agg.copy()
    out["cost_index"] = out.groupby("case")["cost"].transform(
        lambda s: s / s.min())
    return out
