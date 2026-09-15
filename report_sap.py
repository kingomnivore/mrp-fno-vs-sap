"""Builds outputs.txt. Every number the article quotes appears here.

The article's numbers come from this file and from nowhere else. Loose
exploratory scripts draw different samples and return different values for the
same quantity, which is how a stale figure reaches print.
"""
import io

import numpy as np
import pandas as pd

import config as C
import sap as S

FLOOR = 0.95


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        self.lines.append(str(text))

    def rule(self, title):
        self("")
        self("=" * 74)
        self(title)
        self("=" * 74)

    def save(self, path):
        io.open(path, "w", encoding="utf-8").write("\n".join(self.lines) + "\n")


def parameters(out):
    out.rule("0. PARAMETERS THAT ARE JUDGEMENT, NOT DERIVATION")
    out(f"  storage cost percentage per year   {C.STORAGE_PCT:.0%}")
    out(f"  order cost, headline               {C.ORDER_COST:g}")
    out(f"  order cost sweep                   {C.ORDER_COST_SWEEP}")
    out(f"  misspecification factors           {C.MISSPECIFY}")
    out(f"  storage days per year              {S.STORAGE_DAYS_PER_YEAR:g}")
    out(f"  service floor for any cost ranking {FLOOR:.0%}")
    out(f"  Groff reading used                 {S.GROFF_DEFAULT}")
    out(f"  replan cadence, days               {C.REPLAN_EVERY}")
    out(f"  firm demand visibility, days       {C.VISIBILITY_DAYS}")
    out(f"  opening stock, days of mean demand {C.OPENING_STOCK_DAYS}")


def gate(out, examples, structural):
    out.rule("1. GATES")
    out("  SAP's published worked example, shared by all four optimum procedures:")
    out("    price 20, lot-size-independent costs 100, storage 10%,")
    out("    1000 units on four dates one week apart")
    out("")
    for name, passed, detail in examples:
        out(f"    [{'PASS' if passed else 'FAIL'}] {name}")
        out(f"           {detail}")
    out("")
    out("  Structural identities:")
    for name, passed, detail in structural:
        out(f"    [{'PASS' if passed else 'FAIL'}] {name}")
        out(f"           {detail}")


def external_check(out, ext):
    out.rule("2. EXTERNAL CHECK")
    out("  This firm's monthly revenue against ONS series J596,")
    out("  Retail Sales Index for non-store retailing, value not seasonally adjusted.")
    out(f"    overlapping months        {ext['months']}  ({ext['from']} to {ext['to']})")
    out(f"    correlation, levels       {ext['r_levels']:.3f}")
    out(f"    correlation, logs         {ext['r_logs']:.3f}")
    out(f"    peak month, this firm     {ext['peak_dataset']}")
    out(f"    peak month, ONS           {ext['peak_ons']}")
    out(f"    November against own mean {ext['nov_ratio_dataset']:.2f} vs ONS {ext['nov_ratio_ons']:.2f}")


def prices(out, p, rate):
    out.rule("3. PRICES AND WHAT THEY IMPLY")
    out(f"  items carrying a median unit price   {p.size:,}")
    out(f"  median unit price                    {p.median():.2f}")
    out(f"  5th to 95th percentile               {p.quantile(.05):.2f} to {p.quantile(.95):.2f}")
    out("")
    out("  EOQ-implied cycle length in days, by assumed order cost:")
    out(f"    {'order cost':>12} {'p10':>8} {'median':>8} {'p90':>8} {'% over 56d':>12}")
    h = p * C.STORAGE_PCT / S.STORAGE_DAYS_PER_YEAR
    for K in (0.25, 1.0, 5.0, 15.0, 50.0):
        T = np.sqrt(2.0 * K / (h * rate))
        out(f"    {K:>12g} {T.quantile(.1):>8.1f} {T.median():>8.1f} "
            f"{T.quantile(.9):>8.1f} {100 * (T > 56).mean():>11.1f}%")


def static_h1(out, df):
    out.rule("4. H1 STATIC. PERFECT FORESIGHT, SINGLE PASS")
    out("  Cost as a multiple of the Wagner and Whitin optimum. Median over items.")
    out("")
    g = df.groupby("K")[["SP", "WI", "DY", "GR"]].median()
    out(f"    {'order cost':>12} {'SP':>8} {'WI':>8} {'DY':>8} {'GR':>8}")
    for K, r in g.iterrows():
        out(f"    {K:>12g} {r.SP:>8.3f} {r.WI:>8.3f} {r.DY:>8.3f} {r.GR:>8.3f}")
    out("")
    Lc = [c for c in df.columns if c.startswith("L")]
    out("  Best single coverage period against per-item tuning:")
    out(f"    {'order cost':>12} {'one setting':>12} {'at':>6} {'tuned':>8} {'SP':>8} {'closed':>8}")
    for K, s in df.groupby("K"):
        one = s[Lc].median()
        tuned = s[Lc].min(axis=1).median()
        sp = s["SP"].median()
        closed = 100 * (one.min() - tuned) / (one.min() - sp)
        out(f"    {K:>12g} {one.min():>12.3f} {one.idxmin():>6} {tuned:>8.3f} "
            f"{sp:>8.3f} {closed:>7.1f}%")


def rolling_cases(out, agg):
    out.rule("5. H1 ROLLING. WEEKLY REGENERATION WITH FORECAST ERROR")
    out(f"  Ranked among rules clearing a {FLOOR:.0%} fill floor.")
    out("")
    for case in agg["case"].unique():
        s = agg[agg.case == case]
        ok = s[s.fill >= FLOOR].copy()
        dropped = sorted(set(s.config) - set(ok.config))
        ok["idx"] = ok.cost / ok.cost.min()
        b = ok.loc[ok.idx.idxmin()]
        f = ok[ok.who != "SAP only"]
        bf = f.loc[f.idx.idxmin()]
        out(f"  {case}")
        out(f"    best overall        {b.config}  ({b.who})")
        out(f"    best F&O available  {bf.config}, costing {100 * (bf.idx - 1):+.1f}%")
        out(f"    dropped below floor {', '.join(dropped) if dropped else 'none'}")
        out("")
    out("  Full table, cost indexed to the cheapest rule in each case:")
    out("")
    for case in agg["case"].unique():
        out(f"    {case}")
        s = agg[agg.case == case].sort_values("cost_index")
        out(f"      {'rule':<34}{'who':<10}{'fill':>8}{'cover':>9}{'orders/y':>10}{'cost idx':>10}")
        for _, r in s.iterrows():
            out(f"      {r.config:<34}{r.who:<10}{r.fill:>8.4f}{r.cover_days:>9.1f}"
                f"{r.orders_per_year:>10.1f}{r.cost_index:>10.3f}")
        out("")


def misspecification(out, m):
    out.rule("6. H1 ROBUSTNESS. WHEN THE ASSUMED ORDER COST IS WRONG")
    out("  The rule is run believing the order cost is factor x true.")
    out("  The outcome is then priced at the TRUE cost.")
    out("  F&O's Per period reads no cost and so cannot be misinformed about one.")
    out("")
    agg = m.groupby(["factor", "config", "who"], as_index=False).agg(
        fill=("fill", "mean"), cost=("cost", "mean"))
    piv = agg.pivot_table(index=["config", "who"], columns="factor", values="cost")
    piv["swing_pct"] = 100 * (piv.max(axis=1) - piv.min(axis=1)) / piv.min(axis=1)
    out(f"    {'rule':<34}{'who':<10}" + "".join(f"{f'K x{c:g}':>10}" for c in piv.columns[:-1]) + f"{'swing':>9}")
    for (cfg, who), r in piv.sort_values("swing_pct", ascending=False).iterrows():
        out(f"    {cfg:<34}{who:<10}" + "".join(f"{r[c]:>10.2f}" for c in piv.columns[:-1])
            + f"{r.swing_pct:>8.1f}%")
    out("")
    out("  Best SAP optimum against best F&O coverage period, at each level:")
    for f in sorted(m.factor.unique()):
        s = agg[(agg.factor == f) & (agg.fill >= FLOOR)]
        sp, fo = s[s.who == "SAP only"], s[s.who != "SAP only"]
        if sp.empty or fo.empty:
            continue
        b, g = sp.loc[sp.cost.idxmin()], fo.loc[fo.cost.idxmin()]
        out(f"    assumed K = {f:g}x true   best SAP {b.config:<32}{b.cost:>8.2f}   "
            f"best F&O {g.config:<18}{g.cost:>8.2f}   SAP ahead {100 * (g.cost - b.cost) / b.cost:+.1f}%")


def anchoring_placement(out, h2, h3, rhythm):
    out.rule("7. H2 AND H3. ANCHORING AGAINST RECEIPT PLACEMENT")
    out("  Two-way design. H2 reads anchored against floating at the same")
    out("  placement; H3 reads the three placements at the same anchoring.")
    out("")
    out("  H2. Calendar-anchored against floating, receipt at first requirement.")
    out(f"    {'bucket':>8}{'cost %':>10}{'fill %':>10}{'cover %':>10}{'orders %':>10}")
    for _, r in h2.iterrows():
        out(f"    {int(r.L):>7}d{r.cost_pct:>10.2f}{r.fill_pct:>10.2f}"
            f"{r.cover_days_pct:>10.2f}{r.orders_per_year_pct:>10.2f}")
    out("")
    out("  H2 against weekday rhythm. Anchored minus floating cost, per cent.")
    out("  Rows are the share of an item's demand on its busiest weekday.")
    out("  Flat across the six days this firm trades would be 0.167.")
    out("    " + rhythm.to_string().replace("\n", "\n    "))
    out("")
    out("  H3. Three receipt positions, bucket held anchored.")
    out("  Read against `first`, the position SAP ships and the only one F&O has.")
    out(f"    {'bucket':>8} {'placement':<10}{'cost %':>10}{'fill %':>10}{'cover %':>10}{'fill level':>12}")
    for _, r in h3.iterrows():
        lvl = r.get(f"fill_{r.placement}", float('nan'))
        out(f"    {int(r.L):>7}d {r.placement:<10}{r.cost_pct:>10.2f}{r.fill_pct:>10.2f}"
            f"{r.cover_days_pct:>10.2f}{lvl:>12.4f}")


def far_horizon(out, table):
    out.rule("8. H4. SHORT-TERM AND LONG-TERM LOT SIZE")
    out("  Compared against a plain 7 day calendar bucket, which is the same")
    out("  near-term rule with no long-term switch. Any difference comes from")
    out("  the far-horizon rule alone.")
    out("")
    for far, t in table.items():
        out(f"  {far}")
        out(f"    {'case':<26}{'lead':>6}{'identical':>11}{'cost %':>9}{'fill pts':>10}")
        for _, r in t.iterrows():
            out(f"    {r.case:<26}{int(r.lead):>5}d{r.identical_pct:>10.1f}%"
                f"{r.cost_pct:>9.2f}{r.fill_delta:>10.2f}")
        out("")


def tuning(out, d, by_class):
    """Coverage period tuning, and which assignment rule survives.

    The hindsight column picks each item's best period knowing the outcome. It
    bounds the other three and is not a method. Anything quoting it has to say
    so.
    """
    out.rule("9. WHAT TUNING THE COVERAGE PERIOD IS WORTH")
    out("  Fine grid, 3 to 112 days, rolling regeneration, service floor applied.")
    out("  Cost against Part Period Balancing in the same case, per cent.")
    out("  Negative means the F&O rule is cheaper.")
    out("")
    out(f"    {'case':<26}{'one setting':>13}{'demand class':>14}"
        f"{'EOQ rule':>10}{'hindsight':>11}{'best period':>13}")
    for _, r in d.iterrows():
        out(f"    {r['case']:<26}{r['one_setting_vs_SP_pct']:>12.1f}%"
            f"{r['demand_class_vs_SP_pct']:>13.1f}%"
            f"{r['economic_order_quantity_vs_SP_pct']:>9.1f}%"
            f"{r['hindsight_vs_SP_pct']:>10.1f}%"
            f"{int(r['best_single_period']):>11}d")
    out("")
    out("  Share of the one-setting gap each rule closes:")
    out(f"    {'case':<26}{'demand class':>14}{'EOQ rule':>11}{'hindsight':>12}")
    for _, r in d.iterrows():
        out(f"    {r['case']:<26}{r['demand_class_closed_pct']:>13.1f}%"
            f"{r['economic_order_quantity_closed_pct']:>10.1f}%"
            f"{r['hindsight_closed_pct']:>11.1f}%")
    out("")
    out("  Best coverage period per item, by demand class, all cases pooled:")
    out(f"    {'class':<16}{'median':>9}{'10th':>8}{'90th':>8}{'items':>8}")
    for k, r in by_class.iterrows():
        out(f"    {k:<16}{r['median']:>8.0f}d{r['p10']:>7.0f}d{r['p90']:>7.0f}d{int(r['items']):>8}")


def mapping_table(out, t, best_cat):
    out.rule("10. THE MAPPING. AN SAP LOT-SIZE KEY ARRIVING IN F&O")
    out(f"  Seasonal wholesaler, service floor {FLOOR:.0%}.")
    out(f"  'one setting' is the best single coverage period for the catalogue, {best_cat}.")
    out("")
    out(f"    {'key':<5}{'procedure':<34}{'family':<10}{'status':<15}{'fill':>7}{'tuned %':>10}{'one set %':>11}")
    for _, r in t.iterrows():
        out(f"    {r['SAP key']:<5}{r['SAP procedure']:<34}{r['family']:<10}"
            f"{r['status']:<15}{r['fill_sap']:>7.3f}{r['tuned vs SAP %']:>10.2f}"
            f"{r['one setting vs SAP %']:>11.2f}")


def stability(out, verdicts):
    out.rule("11. PLAN STABILITY. PRE-REGISTERED, AND NOT PUBLISHED")
    out("  The rule was fixed before the run: a stability result is publishable")
    out("  only if it holds under BOTH an absolute and a relative tolerance.")
    out("  Absolute tolerance 0.5 units. Relative tolerance 10 per cent.")
    out("")
    for metric, rho, wa, wr, move in verdicts:
        out(f"    {metric}")
        out(f"      spearman between the two tolerances   {rho:.3f}")
        out(f"      most stable rule, absolute tolerance  {wa}")
        out(f"      most stable rule, relative tolerance  {wr}")
        out(f"      largest rank move                     {move:.0f} places")
        out("")
    out("  The order-level measure disagrees on which rule is most stable and")
    out("  moves ranks by five places. No stability number is published.")
