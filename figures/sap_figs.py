"""Figures for the F&O against SAP article.

Chart titles are plain noun phrases naming what the chart is. The claim the
chart supports belongs in the article caption, not on the axes.

Colour carries meaning and means the same thing in every figure: blue is a
setting F&O can reach today, orange is one only SAP can express, grey is
context or a bound.
"""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style as ST

C_FNO, C_SAP, C_BOTH = ST.C_FNO, ST.C_SAP, ST.C_BOTH
FLOOR = 0.95


def _wrap(s, n=18):
    return "\n".join(textwrap.wrap(str(s), n))


def cost_by_case(agg, path="fig_cost_by_case.png"):
    """Cost index per rule in each business case, service floor applied."""
    cases = list(agg["case"].unique())
    fig, axes = plt.subplots(1, len(cases), figsize=(15, 5.6), sharey=True)
    for ax, case in zip(np.atleast_1d(axes), cases):
        s = agg[(agg.case == case) & (agg.fill >= FLOOR)].copy()
        s["idx"] = s.cost / s.cost.min()
        s = s.sort_values("idx", ascending=False)
        cols = [C_SAP if w == "SAP only" else C_BOTH for w in s.who]
        ax.barh(range(len(s)), s.idx, color=cols, height=0.72)
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([_wrap(c, 22) for c in s.config], fontsize=7.5)
        ax.axvline(1.0, color=ST.INK_2, lw=0.8, ls=":")
        ax.set_title(_wrap(case, 22), fontsize=9.5)
        ax.set_xlabel("cost, x cheapest")
        ST.clean(ax, "x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_SAP),
               plt.Rectangle((0, 0), 1, 1, color=C_BOTH)]
    fig.legend(handles, ["Only SAP has it", "Both products have it"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def misspecification(m, path="fig_misspecification.png"):
    """Realised cost against the order cost the rule was told to assume."""
    agg = m.groupby(["factor", "config", "who"], as_index=False)["cost"].mean()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for (cfg, who), g in agg.groupby(["config", "who"]):
        g = g.sort_values("factor")
        sap = who == "SAP only"
        ax.plot(g.factor, g.cost, marker="o", ms=4.5,
                color=C_SAP if sap else C_FNO,
                ls="-" if sap else "--", lw=1.6 if sap else 1.2,
                label=cfg)
        ax.annotate(cfg.split()[0] if sap else cfg.replace("Per period ", ""),
                    (g.factor.iloc[-1], g.cost.iloc[-1]),
                    textcoords="offset points", xytext=(6, 0),
                    fontsize=7.5, color=C_SAP if sap else C_FNO, va="center")
    ax.set_xscale("log"); ax.set_xticks(sorted(agg.factor.unique()))
    ax.set_xticklabels([f"x{f:g}" for f in sorted(agg.factor.unique())])
    ax.set_xlabel("order cost the rule was told to assume, as a multiple of the truth")
    ax.set_ylabel("realised cost, priced at the true cost")
    ax.set_title("Realised Cost Against the Assumed Order Cost")
    ax.set_xlim(0.42, 2.9)
    ST.clean(ax)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return path


def two_way(raw, path="fig_two_way.png"):
    """Anchoring and receipt placement, separated."""
    import sap as S
    piv = raw.pivot_table(index="L", columns=["anchoring", "placement"],
                          values=["cover_days", "fill"], aggfunc="median")
    Ls = sorted(raw.L.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    x = np.arange(len(Ls)); w = 0.2
    series = [("floating", S.FIRST_REQUIREMENT, C_FNO, "floating, at first requirement"),
              ("anchored", S.FIRST_REQUIREMENT, C_SAP, "anchored, at first requirement"),
              ("anchored", S.PERIOD_START, "#f0a882", "anchored, at period start"),
              ("anchored", S.PERIOD_END, ST.C_WARN, "anchored, at period end")]
    for i, (anc, pl, col, lab) in enumerate(series):
        vals = [piv[("cover_days", anc, pl)].get(L, np.nan) for L in Ls]
        ax.bar(x + (i - 1.5) * w, vals, w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"{L}d" for L in Ls])
    ax.set_xlabel("bucket length"); ax.set_ylabel("days of cover, median")
    ax.set_title("Days of Cover by Anchoring and Placement")
    ST.clean(ax)

    ax = axes[1]
    for i, (anc, pl, col, lab) in enumerate(series):
        vals = [100 * piv[("fill", anc, pl)].get(L, np.nan) for L in Ls]
        ax.plot(x, vals, marker="o", ms=5, color=col, label=lab, lw=1.6)
    ax.set_xticks(x); ax.set_xticklabels([f"{L}d" for L in Ls])
    ax.set_xlabel("bucket length"); ax.set_ylabel("fill rate, per cent")
    ax.set_title("Fill Rate by Anchoring and Placement")
    ST.clean(ax)
    # One shared legend below both panels. Inside the left panel it sat on top
    # of the period-end bars, which are the tallest thing in the figure.
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.035),
               ncol=4, fontsize=8.5)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return path


def rhythm(r, path="fig_rhythm.png"):
    """Anchoring penalty against how concentrated demand is on one weekday."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bands = [str(i) for i in r.index]
    for col in r.columns:
        ax.plot(bands, r[col], marker="o", ms=5, lw=1.6, label=f"{col}d bucket")
    ax.axhline(0, color=ST.INK_2, lw=0.8, ls=":")
    ax.set_xlabel("share of an item's demand on its busiest weekday")
    ax.set_ylabel("anchored minus floating cost, per cent")
    ax.set_title("Anchoring Penalty Against Weekday Concentration")
    ST.clean(ax); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return path


def far_horizon(tbl, path="fig_far_horizon.png"):
    """Effect of the long-term lot size against lead time."""
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for (far, t), col in zip(tbl.items(), (C_SAP, "#f0a882")):
        t = t.sort_values("lead")
        ax.plot(t.lead, t.identical_pct, marker="o", ms=5.5, lw=1.7,
                color=col, label=far)
    ax.set_xlabel("lead time, days")
    ax.set_ylabel("items whose outcome is unchanged, per cent")
    ax.set_title("Items Unaffected by the Long-Term Lot Size")
    ax.set_ylim(-4, 104)
    ST.clean(ax); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return path


def tuning_curve(g, by_class, path="fig_tuning.png"):
    """Cost against coverage period, and the best period per demand class."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for q, col in ST.QUAD.items():
        s = g[g.quadrant == q]
        if s.empty:
            continue
        m = s.groupby("L")["cost"].median()
        m = m / m.min()
        ax.plot(m.index, m.values, marker="o", ms=4, lw=1.6, color=col, label=q)
    ax.set_xscale("log"); ax.set_xticks([3, 7, 14, 28, 56, 112])
    ax.set_xticklabels(["3", "7", "14", "28", "56", "112"])
    ax.set_xlabel("coverage period, days")
    ax.set_ylabel("cost, x that class's own best")
    ax.set_title("Cost Against Coverage Period by Demand Class")
    ST.clean(ax); ax.legend(fontsize=8)

    ax = axes[1]
    order = ["smooth", "erratic", "lumpy", "intermittent"]
    bc = by_class.reindex(order)
    y = np.arange(len(bc))
    ax.barh(y, bc["p90"] - bc["p10"], left=bc["p10"], height=0.42,
            color=[ST.QUAD[q] for q in bc.index], alpha=0.35)
    ax.scatter(bc["median"], y, color=[ST.QUAD[q] for q in bc.index], zorder=3, s=46)
    for i, (k, r) in enumerate(bc.iterrows()):
        ax.annotate(f"{r['median']:.0f}d", (r["median"], i),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5)
    ax.set_yticks(y); ax.set_yticklabels(bc.index)
    ax.set_xlabel("best coverage period, days")
    ax.set_title("Best Coverage Period by Demand Class")
    ST.clean(ax, "x")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return path


def mapping(t, path="fig_mapping.png"):
    """What each SAP lot-size key costs when substituted in F&O."""
    t = t.sort_values("tuned vs SAP %")
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    y = np.arange(len(t))
    # The percentage column, not the cost level. An earlier version plotted
    # "one setting (Per period 10d)", which is the absolute cost, so every grey
    # bar came out at the same 63 and the chart said nothing.
    one_col = "one setting vs SAP %"
    ax.barh(y + 0.2, t["tuned vs SAP %"], 0.38, color=C_FNO,
            label="coverage period tuned per item")
    ax.barh(y - 0.2, t[one_col], 0.38, color=C_BOTH,
            label="one coverage period across the catalogue")
    ax.axvline(0, color=ST.INK_2, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k}  {n}" for k, n in
                        zip(t["SAP key"], t["SAP procedure"])], fontsize=8)
    ax.set_xlabel("cost of the F&O substitute against the SAP key, per cent")
    ax.set_title("Cost of Substituting Each SAP Lot-Size Key")
    ST.clean(ax, "x")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    return path
