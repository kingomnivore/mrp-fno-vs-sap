"""Entry point. Reproduces every number in the article.

    python run.py

Writes outputs.txt, seven figures and the result CSVs into the working
directory. The first run derives and caches a daily demand matrix and a median
unit price per item from the source workbook, so it takes appreciably longer
than later ones.

The gates run first and abort on failure. SAP's published worked examples come
before the structural identities, because an engine that is self-consistent but
does not implement the vendor's rule passes the identities and fails the
examples.
"""
import time

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

import config as C
import data as D
import external
import mapping
import placement
import report_sap as RS
import sap_examples
import sapcases
import selftest
import static
import stability
import style
import tuning
from figures import sap_figs

FLOOR = RS.FLOOR


def main():
    t0 = time.time()
    out = RS.Report()
    quiet = lambda s: None

    print("Loading demand and prices...")
    daily = D.daily_demand()
    profile = D.classify(daily, D.active_items(daily), C.HEADLINE_PERIOD)
    prices = D.unit_prices()

    RS.parameters(out)

    # ---------------------------------------------------------------- gates
    print("\nGates...")
    examples = sap_examples.run_all(log=quiet)
    structural = selftest.run_all(daily, profile, log=quiet)
    RS.gate(out, examples, structural)
    RS.external_check(out, external.run())
    print("  vendor examples and structural identities pass")

    codes = [c for c in D.active_items(daily) if c in prices.index]
    RS.prices(out, prices.reindex(codes), daily[codes].mean(axis=0))

    # ------------------------------------------------- perfect foresight
    print("\nStatic baseline...")
    st = static.run(daily, profile, prices, log=quiet)
    st.to_csv("results_h1_static.csv", index=False)
    RS.static_h1(out, st)
    st_best = static.best_period_by_cost(st)
    st_best.to_csv("results_best_period_by_cost.csv", index=False)

    # ------------------------------------------------- rolling regeneration
    print("Business cases...")
    raw = sapcases.run(daily, profile, prices, log=quiet)
    raw.to_csv("results_sap_cases_raw.csv", index=False)
    agg = sapcases.cost_index(sapcases.summarise(raw))
    agg.to_csv("results_sap_cases.csv", index=False)
    RS.rolling_cases(out, agg)

    print("Order cost misspecification...")
    miss = sapcases.run_misspecified(daily, profile, prices, log=quiet)
    miss.to_csv("results_sap_misspec_raw.csv", index=False)
    RS.misspecification(out, miss)

    # ------------------------------------------------ anchoring x placement
    print("Anchoring against placement...")
    pl = placement.run(daily, profile, prices, log=quiet)
    pl.to_csv("results_placement_raw.csv", index=False)
    h2 = placement.effect_of_anchoring(pl)
    h3 = placement.effect_of_placement(pl)
    rhythm = placement.anchoring_by_rhythm(pl)
    h2.to_csv("results_h2_anchoring.csv", index=False)
    h3.to_csv("results_h3_placement.csv", index=False)
    rhythm.to_csv("results_h2_by_rhythm.csv")
    RS.anchoring_placement(out, h2, h3, rhythm)

    # ------------------------------------------------------- the far horizon
    print("Far horizon...")
    h4 = far_horizon(raw)
    RS.far_horizon(out, h4)

    # ------------------------------------------------------------- tuning
    print("Coverage period tuning...")
    grid = tuning.run(daily, profile, prices, log=quiet)
    grid.to_csv("results_tuning_grid.csv", index=False)
    sp_cost = (raw[(raw.config == "SP Part Period Balancing") & (raw.fill >= FLOOR)]
               .set_index(["case", "code"])["cost"])
    rules = tuning.assignment_rules(grid, daily, prices, sp_cost)
    rules.to_csv("results_assignment_rules.csv", index=False)
    by_class = tuning.best_period_by_class(grid)
    RS.tuning(out, rules, by_class)

    # ------------------------------------------------------------- stability
    print("Plan stability, both tolerances...")
    stab = stability.run(daily, profile, prices, log=quiet)
    stab.to_csv("results_stability_raw.csv", index=False)
    verdicts = []
    for metric in ("order_churn", "PCR"):
        piv, rho, _ = stability.verdict(stab, metric)
        piv.to_csv(f"results_stability_{metric}.csv")
        verdicts.append((metric, rho,
                         piv["absolute"].idxmin()[0], piv["relative"].idxmin()[0],
                         piv.rank_moved.max()))
    RS.stability(out, verdicts)

    # --------------------------------------------------------- the mapping
    print("Migration mapping...")
    mp = mapping.run(daily, profile, prices, log=quiet)
    mp.to_csv("results_mapping_raw.csv", index=False)
    tab, best_cat = mapping.table(mp)
    tab.to_csv("results_mapping.csv", index=False)
    RS.mapping_table(out, tab, best_cat)

    out.save(C.OUTPUTS)

    # ------------------------------------------------------------- figures
    print("\nFigures...")
    style.apply()
    written = [
        sap_figs.cost_by_case(agg),
        sap_figs.misspecification(miss),
        sap_figs.two_way(pl),
        sap_figs.rhythm(rhythm),
        sap_figs.far_horizon(h4),
        sap_figs.tuning_curve(grid.dropna(subset=["cost"]), by_class),
        sap_figs.mapping(tab),
    ]
    print("\nwrote %s and %s" % (C.OUTPUTS, ", ".join(written)))
    print("total %.0fs" % (time.time() - t0))


def far_horizon(raw):
    """Effect of the long-term lot size, against lead time.

    Compared with a plain 7 day calendar bucket, which is the same near-term
    rule with no long-term switch, so any difference is the far-horizon rule.
    """
    base = raw[raw.config == "Calendar bucket 7d"].set_index(["case", "code"])
    out = {}
    for far in ("Short 7d / long 28d", "Short 7d / long 56d"):
        s = raw[raw.config == far].set_index(["case", "code"])
        j = base.join(s, rsuffix="_sl", how="inner")
        rows = []
        for case, g in j.groupby(level="case"):
            same = (np.isclose(g.cost, g.cost_sl, rtol=1e-9)
                    & np.isclose(g.fill, g.fill_sl, rtol=1e-9))
            rows.append(dict(case=case, lead=C.CASES[case]["lead_time"],
                             identical_pct=100 * same.mean(),
                             cost_pct=100 * (g.cost_sl.median() - g.cost.median())
                             / g.cost.median(),
                             fill_delta=100 * (g.fill_sl.mean() - g.fill.mean())))
        out[far] = pd.DataFrame(rows).sort_values("lead")
    return out


if __name__ == "__main__":
    main()
