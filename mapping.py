"""The deliverable: what an SAP lot-size key costs when it lands in F&O.

A material arriving from S/4HANA carries a lot-sizing procedure in the MRP 1
view. F&O has a coverage code. This module answers, for each SAP key, which
coverage code reproduces it, and where none does, what the best available
substitute costs in stock, orders and service.

Three kinds of answer, and the table says which applies:

    EXACT       the same mechanism exists in F&O and produces the same plan
    APPROXIMATE a coverage code gets close, and the cost of the difference is
                measured here
    NONE        F&O has no rule in that family at all

The mapping is computed rather than asserted. Every "approximate" row is the
cheapest F&O rule that also clears the service floor, chosen from the same
replay the rest of the article uses, so the cost in the table is a measured
cost and not a judgement.
"""
import numpy as np
import pandas as pd

import config as C
import data as D
import engine as E
import rolling as R
import sap as S

FLOOR = 0.95        # a substitute that fails service is not a substitute

EXACT, APPROX, NONE = "Exact", "Approximate", "No equivalent"

# SAP's lot-sizing procedures, by family, with the F&O coverage code that
# corresponds where one does. Keys are SAP's material master indicators.
SAP_KEYS = [
    # ---- static
    dict(key="EX", name="Lot-for-lot order quantity", family="Static",
         kind="l4l", fno="Per requirement", status=EXACT),
    dict(key="FX", name="Fixed lot size", family="Static",
         kind="foq", fno=None, status=NONE),
    dict(key="HB", name="Replenish to maximum stock level", family="Static",
         kind="minmax", fno="Min/Max", status=EXACT),
    # ---- period
    dict(key="TB", name="Daily lot size", family="Period",
         kind="sap_cal", L=1, fno="Per period 1d", status=EXACT),
    dict(key="WB", name="Weekly lot size", family="Period",
         kind="sap_cal", L=7, fno=None, status=APPROX),
    dict(key="MB", name="Monthly lot size", family="Period",
         kind="sap_cal", L=28, fno=None, status=APPROX),
    # ---- optimum
    dict(key="SP", name="Part Period Balancing", family="Optimum",
         kind="sap_opt", code="SP", fno=None, status=NONE),
    dict(key="WI", name="Sliding Economic Lot Size", family="Optimum",
         kind="sap_opt", code="WI", fno=None, status=NONE),
    dict(key="DY", name="Dynamic Planning Calculation", family="Optimum",
         kind="sap_opt", code="DY", fno=None, status=NONE),
    dict(key="GR", name="Groff Reorder Procedure", family="Optimum",
         kind="sap_opt", code="GR", fno=None, status=NONE),
]

# The F&O rules a migration can actually choose between.
FNO_GRID = (3, 5, 7, 10, 14, 21, 28, 42, 56, 84, 112)


def _cfg_for(spec, price, lead_time, mean_daily):
    """Build the runnable configuration for one SAP key."""
    k = spec["kind"]
    if k == "l4l":
        return {"name": spec["key"], "kind": "l4l", "L": 1}
    if k == "foq":
        # SAP's fixed lot size. Expressed as 28 days of mean demand so it is
        # comparable across items of very different volume, the same treatment
        # the LN article gave Fixed Order Quantity.
        return {"name": spec["key"], "kind": "foq",
                "Q": max(28 * mean_daily, 1.0), "L": 28}
    if k == "minmax":
        return {"name": spec["key"], "kind": "minmax", "L": 1,
                "mn": lead_time * mean_daily,
                "mx": (lead_time + 28) * mean_daily}
    if k == "sap_cal":
        return {"name": spec["key"], "kind": "sap_cal", "L": spec["L"],
                "placement": S.FIRST_REQUIREMENT, "anchor": 0}
    if k == "sap_opt":
        return {"name": spec["key"], "kind": "sap_opt", "code": spec["code"],
                "price": price, "storage_pct": C.STORAGE_PCT,
                "K": C.ORDER_COST, "L": 1}
    raise ValueError(k)


def run(daily, profile, prices, case="Seasonal wholesaler", log=print):
    """Replay every SAP key and every candidate F&O coverage period."""
    spec = C.CASES[case]
    sc = {"lead_time": spec["lead_time"], "replan_every": C.REPLAN_EVERY,
          "visibility": C.VISIBILITY_DAYS, "noise": spec["noise"],
          "opening_stock_days": C.OPENING_STOCK_DAYS, "coverage_fence": None}
    pool = D.sample(profile, daily)
    pool = pool[pool["StockCode"].isin(prices.index)]
    rows = []
    for _, it in pool.iterrows():
        code = it["StockCode"]
        a = D.series(daily, code)
        md = float(a.mean())
        if md <= 0:
            continue
        price = float(prices[code])
        h = S.holding_per_unit_day(price, C.STORAGE_PCT)

        def record(name, side, cfg):
            m = R.simulate(a, cfg, sc, seed=R.stable_seed(code, name, case))
            rows.append({"code": code, "quadrant": it["quadrant"],
                         "name": name, "side": side,
                         "cost": C.ORDER_COST * m["n_orders"] + h * m["unit_days"],
                         **m})

        for s in SAP_KEYS:
            record(s["key"], "SAP", _cfg_for(s, price, spec["lead_time"], md))
        for L in FNO_GRID:
            record(f"Per period {L}d", "F&O",
                   {"name": f"Per period {L}d", "kind": "float", "L": L})
        record("Per requirement", "F&O", {"name": "Per requirement",
                                          "kind": "l4l", "L": 1})
        record("Min/Max", "F&O", {"name": "Min/Max", "kind": "minmax", "L": 1,
                                  "mn": spec["lead_time"] * md,
                                  "mx": (spec["lead_time"] + 28) * md})
    return pd.DataFrame(rows)


def table(raw):
    """One row per SAP lot-size key: the best F&O substitute and what it costs.

    The substitute is chosen per item, then averaged, because a migration sets
    a coverage code per item and not one for the catalogue. The row also
    reports the single best catalogue-wide coverage period, because that is
    what a migration under time pressure will actually do.
    """
    ok = raw[raw.fill >= FLOOR]
    fno = ok[ok.side == "F&O"]
    per_item = fno.loc[fno.groupby("code")["cost"].idxmin()].set_index("code")
    catalogue = fno.groupby("name")["cost"].mean()
    best_cat_name = catalogue.idxmin()
    cat_item = ok[(ok.name == best_cat_name)].set_index("code")

    out = []
    for s in SAP_KEYS:
        src = ok[ok.name == s["key"]].set_index("code")
        idx = src.index.intersection(per_item.index).intersection(cat_item.index)
        if len(idx) == 0:
            continue
        base = src.loc[idx, "cost"]
        tuned = per_item.loc[idx, "cost"]
        flat = cat_item.loc[idx, "cost"]
        out.append({
            "SAP key": s["key"], "SAP procedure": s["name"],
            "family": s["family"], "status": s["status"],
            "F&O equivalent": s["fno"] or "none",
            "items": len(idx),
            "fill_sap": float(src.loc[idx, "fill"].mean()),
            "cost_sap": float(base.mean()),
            "best substitute, tuned per item": float(tuned.mean()),
            "tuned vs SAP %": 100.0 * float((tuned.mean() - base.mean()) / base.mean()),
            f"one setting ({best_cat_name})": float(flat.mean()),
            "one setting vs SAP %": 100.0 * float((flat.mean() - base.mean()) / base.mean()),
            "most common tuned choice": per_item.loc[idx, "name"].mode().iloc[0],
        })
    return pd.DataFrame(out), best_cat_name
