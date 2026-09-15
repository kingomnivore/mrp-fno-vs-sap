"""Plan stability, measured under two tolerances and published only if both agree.

The companion Infor LN article measured plan churn throughout and published
nothing from it. The reason is recorded here so it is not repeated: counting a
planned order as changed by a fixed number of units, and by a fixed percentage
of its previous quantity, produced conclusions pointing in opposite directions.
That is a property of the measure, not of the products, and picking whichever
tolerance gave the tidier answer would have been the worst thing in the article.

So the rule for this article is fixed before the run, not after:

    A stability result is publishable only if it holds under BOTH an absolute
    and a relative tolerance. If the two disagree in sign, or one is
    significant and the other is not, nothing is published and the article says
    in one sentence that the measure did not survive.

This matters here because SAP's planning time fence and firming types exist
precisely to dampen replanning, and F&O's freeze time fence is the nearest
equivalent. It is tempting material. The temptation is why the rule is written
down first.

Two measures come out of rolling.simulate:

    PCR         share of days in the actionable window whose planned quantity
                moved at all
    order_churn share of planned ORDERS that moved, which is what an action
                message actually is
"""
import numpy as np
import pandas as pd

import config as C
import data as D
import rolling as R
import sap as S
import sapcases

# Pre-registered. An order has moved if its quantity changes by more than
# ABS_TOL units, or by more than REL_TOL of its previous quantity.
ABS_TOL = R.TOL      # 0.5 units, the existing absolute rule
REL_TOL = 0.10       # 10 per cent of the previous planned quantity


def run(daily, profile, prices, case="Seasonal wholesaler", log=print):
    """Every rule, replayed twice: once per tolerance, everything else equal.

    The same seed is used for both, so the demand signal and the forecast error
    are identical and the only thing that differs is how a change is counted.
    """
    spec = C.CASES[case]
    pool = D.sample(profile, daily)
    pool = pool[pool["StockCode"].isin(prices.index)]
    rows = []
    for tol_name, rel in (("absolute", None), ("relative", REL_TOL)):
        sc = {"lead_time": spec["lead_time"], "replan_every": C.REPLAN_EVERY,
              "visibility": C.VISIBILITY_DAYS, "noise": spec["noise"],
              "opening_stock_days": C.OPENING_STOCK_DAYS,
              "coverage_fence": None, "rel_tol": rel}
        for _, it in pool.iterrows():
            code = it["StockCode"]
            a = D.series(daily, code)
            md = float(a.mean())
            if md <= 0:
                continue
            price = float(prices[code])
            for base in sapcases.configs(price):
                cfg = sapcases._bind(base, spec["lead_time"], md)
                m = R.simulate(a, cfg, sc,
                               seed=R.stable_seed(code, cfg["name"], case))
                rows.append({"tolerance": tol_name, "config": cfg["name"],
                             "who": cfg["who"], "code": code,
                             "quadrant": it["quadrant"],
                             "PCR": m["PCR"], "order_churn": m["order_churn"]})
        log(f"  {tol_name} tolerance done")
    return pd.DataFrame(rows)


def verdict(raw, metric="order_churn"):
    """Rank rules under each tolerance and report whether the two agree.

    Agreement is tested on the ordering, not on the level, because the level is
    not comparable between tolerances by construction: a relative tolerance
    ignores small absolute moves on large quantities and vice versa. What has
    to survive is which rules are more stable than which.
    """
    agg = raw.groupby(["tolerance", "config", "who"], as_index=False)[metric].mean()
    piv = agg.pivot_table(index=["config", "who"], columns="tolerance",
                          values=metric)
    piv["rank_abs"] = piv["absolute"].rank()
    piv["rank_rel"] = piv["relative"].rank()
    piv["rank_moved"] = (piv["rank_abs"] - piv["rank_rel"]).abs()
    rho = piv["absolute"].corr(piv["relative"], method="spearman")
    same_winner = piv["absolute"].idxmin() == piv["relative"].idxmin()
    return piv.sort_values("rank_abs"), rho, same_winner
