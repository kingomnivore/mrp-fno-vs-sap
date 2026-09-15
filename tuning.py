"""What tuning the coverage period is worth, and which tuning rule to use.

The gap between F&O and SAP's Part Period Balancing is measured against one
coverage period applied across the whole catalogue. Most of it closes if the
period is set per item. The question this module answers is what a planner can
actually set it from.

Three ways of choosing, in increasing order of realism:

  one setting     the best single period for the catalogue
  demand class    the median best period for the item's class
  economic order  derived from the item's own price and demand rate
  quantity

and one that is not a method at all:

  hindsight       each item's best period, chosen knowing the outcome

The hindsight column is reported because it bounds the other three. It is
labelled as a bound in the article and must never be presented as achievable.
"""
import numpy as np
import pandas as pd

import config as C
import data as D
import rolling as R
import sap as S

GRID = (3, 5, 7, 10, 14, 21, 28, 42, 56, 84, 112)
FLOOR = 0.95


def run(daily, profile, prices, log=print):
    """Replay every coverage period in the grid, in every business case."""
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
            if a.mean() <= 0:
                continue
            price = float(prices[code])
            h = S.holding_per_unit_day(price, C.STORAGE_PCT)
            for L in GRID:
                cfg = {"name": f"Per period {L}d", "kind": "float", "L": L}
                m = R.simulate(a, cfg, sc,
                               seed=R.stable_seed(code, cfg["name"], case))
                rows.append({"case": case, "code": code,
                             "quadrant": it["quadrant"], "L": L,
                             "price": price,
                             "cost": C.ORDER_COST * m["n_orders"] + h * m["unit_days"],
                             **m})
        log(f"  {case}: {len(items)} items x {len(GRID)} periods")
    return pd.DataFrame(rows)


def _implied_cycle(price, rate, K=None):
    """EOQ cycle length in days for one item.

    sqrt(2K / (h * d)). Needs the item's price and demand rate, which an
    implementation has, and an order cost, which F&O has no field for. That
    missing field is the request in Section 5.3.
    """
    K = C.ORDER_COST if K is None else K
    h = S.holding_per_unit_day(price, C.STORAGE_PCT)
    if h <= 0 or rate <= 0:
        return np.nan
    return float(np.sqrt(2.0 * K / (h * rate)))


def assignment_rules(grid, daily, prices, sp_cost):
    """Compare the ways of choosing a coverage period, per business case.

    sp_cost is a Series of Part Period Balancing cost per (case, code), which
    is the comparator every column is measured against.
    """
    g = grid.copy()
    g.loc[g.fill < FLOOR, "cost"] = np.nan
    rate = daily.mean(axis=0)
    grid_days = sorted(g.L.unique())

    # The demand-class rule, derived on the pooled population so it is a rule
    # and not a per-case fit.
    piv_all = g.pivot_table(index=["quadrant", "code"], columns="L", values="cost")
    best_all = piv_all.idxmin(axis=1).dropna()
    by_class = best_all.groupby(best_all.index.get_level_values("quadrant")).median()
    by_class = {k: min(grid_days, key=lambda x: abs(x - v))
                for k, v in by_class.items()}

    rows = []
    for case, s in g.groupby("case"):
        piv = s.pivot_table(index="code", columns="L", values="cost")
        quad = s.groupby("code")["quadrant"].first()
        one = piv.mean()
        flat = piv[one.idxmin()]
        hindsight = piv.min(axis=1)

        cls, eoq = {}, {}
        for code in piv.index:
            L = by_class.get(quad[code])
            if L in piv.columns:
                cls[code] = piv.loc[code, L]
            T = _implied_cycle(float(prices[code]), float(rate[code]))
            if not np.isnan(T):
                Le = min(grid_days, key=lambda x: abs(x - T))
                if Le in piv.columns:
                    eoq[code] = piv.loc[code, Le]
        cls, eoq = pd.Series(cls), pd.Series(eoq)

        sp = sp_cost.xs(case) if case in sp_cost.index.get_level_values(0) else pd.Series(dtype=float)
        idx = (hindsight.dropna().index
               .intersection(cls.dropna().index)
               .intersection(eoq.dropna().index)
               .intersection(flat.dropna().index)
               .intersection(sp.index))
        if len(idx) == 0:
            continue
        rows.append({"case": case, "items": len(idx),
                     "one_setting": float(flat[idx].mean()),
                     "demand_class": float(cls[idx].mean()),
                     "economic_order_quantity": float(eoq[idx].mean()),
                     "hindsight": float(hindsight[idx].mean()),
                     "SP": float(sp[idx].mean()),
                     "best_single_period": int(one.idxmin())})
    out = pd.DataFrame(rows)
    for c in ("one_setting", "demand_class", "economic_order_quantity", "hindsight"):
        out[c + "_vs_SP_pct"] = 100.0 * (out[c] - out.SP) / out.SP
    span = out.one_setting - out.SP
    for c in ("demand_class", "economic_order_quantity", "hindsight"):
        out[c + "_closed_pct"] = 100.0 * (out.one_setting - out[c]) / span
    out.attrs["by_class"] = by_class
    return out


def best_period_by_class(grid):
    """Median best coverage period per demand class, with its spread."""
    g = grid.copy()
    g.loc[g.fill < FLOOR, "cost"] = np.nan
    piv = g.pivot_table(index=["quadrant", "code"], columns="L", values="cost")
    best = piv.idxmin(axis=1).dropna()
    q = best.index.get_level_values("quadrant")
    return pd.DataFrame({"median": best.groupby(q).median(),
                         "p10": best.groupby(q).quantile(.1),
                         "p90": best.groupby(q).quantile(.9),
                         "items": best.groupby(q).size()})
