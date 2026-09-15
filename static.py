"""Perfect foresight, single pass, against the Wagner and Whitin optimum.

This is the condition the lot-sizing literature says flatters an optimising
heuristic, so it is run first and reported as a baseline rather than as a
result. Section 4.1 of the article is the rolling version, which is the one
that decides anything.

Two things come out of here:

  - how close each SAP optimum procedure gets to the exact optimum, which is
    the sanity check on the reimplementations beyond the vendor examples;
  - how much of F&O's gap is the missing optimiser and how much is one
    coverage period applied across a mixed catalogue.
"""
import numpy as np
import pandas as pd

import config as C
import data as D
import engine as E
import sap as S

# Coverage periods swept. Wider than config.PERIODS because the point here is
# to find the best one, not to test the four the previous article used.
GRID = (3, 5, 7, 10, 14, 21, 28, 42, 56, 84, 112, 168, 252)


def run(daily, profile, prices, log=print):
    """Every rule against the exact optimum, at each assumed order cost."""
    sample = D.sample(profile, daily)
    sample = sample[sample["StockCode"].isin(prices.index)]
    rows = []
    for K in C.ORDER_COST_SWEEP:
        for _, it in sample.iterrows():
            code = it["StockCode"]
            d = D.series(daily, code)
            if d.sum() <= 0:
                continue
            price = float(prices[code])
            h = S.holding_per_unit_day(price, C.STORAGE_PCT)
            if h <= 0:
                continue
            opt = E.wagner_whitin(d, K, h)
            if opt <= 0:
                continue
            r = {"K": K, "code": code, "quadrant": it["quadrant"]}
            for L in GRID:
                r[f"L{L}"] = E.cost_of(E.period_floating(d, L), d, K, h) / opt
            for c, fn in S.OPTIMUM.items():
                r[c] = E.cost_of(fn(d, price, C.STORAGE_PCT, K), d, K, h) / opt
            rows.append(r)
        log(f"  order cost {K:g}: {len(sample)} items")
    return pd.DataFrame(rows)


def best_period_by_cost(df):
    """The best single coverage period at each assumed order cost.

    This is the table behind the statement that the right coverage period
    depends on a cost F&O has no field for.
    """
    cols = [c for c in df.columns if c.startswith("L")]
    out = []
    for K, g in df.groupby("K"):
        med = g[cols].median()
        out.append({"K": K,
                    "best_period_days": int(med.idxmin()[1:]),
                    "cost_multiple": float(med.min()),
                    "tuned": float(g[cols].min(axis=1).median()),
                    "SP": float(g["SP"].median())})
    return pd.DataFrame(out)
