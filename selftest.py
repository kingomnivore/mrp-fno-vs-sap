"""Four structural checks. Identities, not plausibility tests.

Three of these failed the first time they were written, and each failure was a
real defect that would otherwise have reached the article:

  - the Wagner and Whitin dynamic programme tiled calendar periods where it
    should have tiled demand occurrences, and returned values above the true
    optimum, which silently weakened the bound it exists to provide;
  - the replay committed orders on their receipt date rather than their release
    date, so nothing was ever ordered in time and fill collapsed to 1.9%;
  - runs were seeded from Python's hash(), which is salted per process, so the
    published figures would not have reproduced on anyone else's machine.

Run this before reading any result.
"""
import numpy as np

import config as C
import engine as E
import rolling as R
import sap as S


def _demand(rng, n=120, p=0.45, hi=60):
    return (rng.random(n) < p) * rng.integers(1, hi, n).astype(float)


def check_wagner_whitin(trials=200, seed=1):
    """No feasible lot-sizing rule may cost less than the exact optimum.

    Feasible means the schedule never backorders. SAP's period-end availability
    date does backorder by construction, so it is excluded here and asserted
    separately in check_period_end_backorders. A rule that serves demand late
    can always look cheaper than the optimum, because the holding it avoids is
    holding it was supposed to pay.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        d = _demand(rng)
        if d.sum() <= 0:
            continue
        for setup, hold in ((50.0, 1.0), (500.0, 1.0), (50.0, 0.1)):
            opt = E.wagner_whitin(d, setup, hold)
            # price and percentage chosen so SAP's daily holding equals `hold`
            price, pct = hold * S.STORAGE_DAYS_PER_YEAR, 1.0
            schedules = [E.lot_for_lot(d),
                         E.period_floating(d, 7),
                         E.period_anchored(d, 7, 0),
                         E.period_anchored_at_need(d, 7, 0),
                         E.period_varying(d, 7, 28, 56),
                         S.period_calendar(d, 7, S.FIRST_REQUIREMENT, 0),
                         S.period_calendar(d, 7, S.PERIOD_START, 0)]
            schedules += [fn(d, price, pct, setup) for fn in S.OPTIMUM.values()]
            for orders in schedules:
                cost = E.cost_of(orders, d, setup, hold)
                worst = min(worst, cost - opt)
    return worst >= -1e-6, f"smallest margin above the optimum: {worst:+.6f}"


def check_unit_bucket(trials=200, seed=2):
    """A one day bucket must reproduce one order per requirement exactly."""
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(trials):
        d = _demand(rng)
        rules = [E.period_floating,
                 lambda x, L: E.period_anchored(x, L, 0),
                 lambda x, L: E.period_anchored_at_need(x, L, 0)]
        rules += [(lambda pl: lambda x, L: S.period_calendar(x, L, pl, 0))(pl)
                  for pl in S.PLACEMENTS]
        for fn in rules:
            if not np.array_equal(fn(d, 1), E.lot_for_lot(d)):
                bad += 1
    return bad == 0, f"{bad} mismatches across {trials} series and six rules"


def check_zero_noise(daily, profile, items=20):
    """With a perfect forecast, a calendar-anchored plan must never move.

    A floating bucket may still move, because its boundaries are defined by the
    demand rather than by the calendar. That is the mechanism, not a defect,
    which is why only the anchored rules are asserted here.
    """
    import data as D
    sample = D.sample(profile, daily, per_quadrant=max(1, items // 4))
    scenario = {"lead_time": 0, "replan_every": C.REPLAN_EVERY,
                "visibility": C.VISIBILITY_DAYS, "noise": 0.0,
                "opening_stock_days": C.OPENING_STOCK_DAYS}
    worst = 0.0
    for _, it in sample.iterrows():
        a = D.series(daily, it["StockCode"])
        for cfg in ({"name": "anch14", "kind": "anch_need", "L": 14},
                    {"name": "l4l", "kind": "l4l", "L": 1}):
            m = R.simulate(a, cfg, scenario, seed=R.stable_seed(it["StockCode"], cfg["name"]))
            worst = max(worst, m["PCR"])
    return worst < 1e-12, f"largest plan churn on a perfect forecast: {worst:.2e}"


def check_reproducible():
    """The seed must not depend on the interpreter session."""
    a = R.stable_seed("85123A", "Period 14d", "Long-lead importer")
    b = R.stable_seed("85123A", "Period 14d", "Long-lead importer")
    known = 2560379992   # CRC32 of the joined key; fixed across sessions
    return (a == b == known), f"stable_seed returned {a}, expected {known}"


def check_period_end_backorders(trials=200, seed=3):
    """SAP's three receipt placements must behave as documented.

    "The system sets the availability date for period lot-sizing procedures to
    the first requirements date of the period. However, you can also define
    that the availability date is at the beginning or end of the period."

    Three assertions, all identities:

      - all three place the same total quantity, so the placement moves timing
        and nothing else;
      - first-requirement and period-start never drive projected on-hand
        negative, which is why the Wagner and Whitin bound applies to them;
      - period-end always does, on some series, because the receipt lands after
        demand earlier in the same period has been required.

    The third is the mechanism the article prices, so it is asserted rather
    than assumed. If it ever stopped holding, period_calendar would have
    silently stopped implementing the setting.
    """
    rng = np.random.default_rng(seed)
    qty_bad = 0
    worst = {pl: 0.0 for pl in S.PLACEMENTS}
    for _ in range(trials):
        d = _demand(rng)
        if d.sum() <= 0:
            continue
        for L in (7, 14, 28):
            outs = {pl: S.period_calendar(d, L, pl, 0) for pl in S.PLACEMENTS}
            totals = {round(float(o.sum()), 6) for o in outs.values()}
            if len(totals) != 1:
                qty_bad += 1
            for pl, o in outs.items():
                inv = np.cumsum(o - d)
                worst[pl] = min(worst[pl], float(inv.min()))
    ok = (qty_bad == 0
          and worst[S.FIRST_REQUIREMENT] >= -1e-9
          and worst[S.PERIOD_START] >= -1e-9
          and worst[S.PERIOD_END] < 0)
    return ok, (f"{qty_bad} quantity mismatches; worst projected on-hand "
                f"first {worst[S.FIRST_REQUIREMENT]:+.0f}, "
                f"start {worst[S.PERIOD_START]:+.0f}, "
                f"end {worst[S.PERIOD_END]:+.0f}")


def check_long_term_valid_from():
    """SAP's worked example for when the long-term lot size takes over.

    "The calculation of the date from which the long-term lot size is valid is
    15 March. Four months have been entered for the periods for this
    calculation. The system calculates four months into the future (15 July)
    and then rounds up to the beginning of the next complete period. The
    long-term lot size is thus valid from 1 August."

    On a day grid of L-day periods, counting `periods` forward from a period
    boundary and rounding up must land on a period boundary, and must never
    land short of the raw count.
    """
    bad = []
    for L in (7, 14, 28, 30):
        for n in (1, 2, 4, 6):
            for start in (0, L, 3 * L):
                v = S.long_term_valid_from(n, L, start)
                if v % L != 0 or v < start + n * L:
                    bad.append((L, n, start, v))
    return not bad, ("every switch date lands on a period boundary at or after "
                     "the raw count" if not bad else f"{len(bad)} bad: {bad[:3]}")


def run_all(daily, profile, log=print):
    log("Structural checks")
    results = [
        ("no rule beats the Wagner and Whitin optimum", *check_wagner_whitin()),
        ("a one day bucket equals one order per requirement", *check_unit_bucket()),
        ("a calendar-anchored plan does not move on a perfect forecast",
         *check_zero_noise(daily, profile)),
        ("results reproduce in a fresh process", *check_reproducible()),
        ("SAP's three receipt placements behave as documented",
         *check_period_end_backorders()),
        ("the long-term lot size switches on a period boundary",
         *check_long_term_valid_from()),
    ]
    ok = True
    for name, passed, detail in results:
        ok &= passed
        log(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        log(f"         {detail}")
    if not ok:
        raise SystemExit("A structural check failed. Nothing downstream is safe.")
    return results
