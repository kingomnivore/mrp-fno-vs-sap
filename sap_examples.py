"""The gate: SAP's own published worked examples, reproduced exactly.

Every optimum lot-sizing procedure in sap.py is named after a published
heuristic whose textbook form differs from SAP's. Reimplementing from the
literature and assuming SAP does the same thing is the mistake this file
exists to catch. An identity check proves the code is self-consistent; a
vendor's published arithmetic proves the code implements the vendor's rule.

All four pages carry the same worked example, which is what makes it a gate
rather than four separate tests. Same price, same costs, same requirements, and
three different answers across the four procedures:

    Price                          20
    Lot-size-independent costs    100
    Storage cost percentage        10%
    Requirements                 1000 on 06.07, 13.07, 20.07 and 27.07

    SP  Part Period Balancing          2000
    WI  Sliding Economic Lot Size      2000
    DY  Dynamic Planning Calculation   3000
    GR  Groff Reorder Procedure        1000

Groff additionally prints its two comparison figures, 1.79 against 19.18, and
those are asserted too, because the lot size alone would pass under several
wrong readings of the rule.

Run this before anything else. Nothing downstream is safe until it is green.
"""
import numpy as np

import sap as S

# The example, as published. Requirements one week apart.
PRICE = 20.0
STORAGE_PCT = 0.10
K = 100.0
SPACING = 7
QTY = 1000.0
HORIZON = 28

EXPECTED = {"SP": 2000.0, "WI": 2000.0, "DY": 3000.0, "GR": 1000.0}

# The two figures printed in the Groff table, for the step from 1000 to 2000.
GROFF_SAVING = 1.79
GROFF_ADDED_STORAGE = 19.18


def example_series():
    """1000 units on each of four dates, one week apart."""
    net = np.zeros(HORIZON)
    net[::SPACING][:4] = QTY
    return net


def check_first_lot():
    """Each procedure must return SAP's published first lot size."""
    net = example_series()
    bad = []
    for code, fn in S.OPTIMUM.items():
        out = fn(net, PRICE, STORAGE_PCT, K)
        got = float(out[0])
        want = EXPECTED[code]
        if abs(got - want) > 1e-6:
            bad.append(f"{code} returned {got:.0f}, SAP publishes {want:.0f}")
    detail = "; ".join(bad) if bad else (
        "SP 2000, WI 2000, DY 3000, GR 1000, all as published")
    return not bad, detail


def check_groff_variants():
    """Every reading of Groff's rule must reproduce SAP's published answer.

    SAP's example has a single step, so it cannot separate the two readings.
    That is the point of this check: it records that the example passes under
    both, so nobody later mistakes the gate being green for the rule being
    settled. sap.groff explains which is the default and why.
    """
    net = example_series()
    bad = []
    for v in (S.GROFF_STOP_RULE, S.GROFF_DAY_OFFSET):
        got = float(S.groff(net, PRICE, STORAGE_PCT, K, variant=v)[0])
        if abs(got - EXPECTED["GR"]) > 1e-6:
            bad.append(f"{v} returned {got:.0f}")
    return not bad, ("both readings return SAP's 1000, so the published "
                     "example does not choose between them"
                     if not bad else "; ".join(bad))


def check_groff_figures():
    """Groff's own two comparison figures, to the two decimals SAP prints.

    SAP's table gives, for the step from a 1000 lot to a 2000 lot, savings in
    lot-size-independent costs of 1.79 against additional storage costs of
    19.18. Reproducing the lot size without these would pass under readings of
    the rule that happen to stop in the same place for the wrong reason.
    """
    h = S.holding_per_unit_day(PRICE, STORAGE_PCT)
    saving, added = S.groff_figures(K, h, QTY, SPACING)
    ok = (abs(saving - GROFF_SAVING) < 5e-3
          and abs(added - GROFF_ADDED_STORAGE) < 5e-3)
    return ok, (f"saving {saving:.2f} against SAP's {GROFF_SAVING:.2f}, "
                f"added storage {added:.2f} against SAP's {GROFF_ADDED_STORAGE:.2f}")


def check_procedures_differ():
    """The four must not collapse onto each other.

    Three distinct answers on one example. If a refactor ever makes all four
    agree, the gate above would still pass for the two that happen to be 2000,
    so the spread is asserted separately.
    """
    net = example_series()
    got = {c: float(fn(net, PRICE, STORAGE_PCT, K)[0])
           for c, fn in S.OPTIMUM.items()}
    n = len(set(got.values()))
    return n == 3, f"{n} distinct lot sizes across the four procedures: {got}"


def check_storage_divisor():
    """365, not 360, is what reproduces SAP's figures.

    SAP does not state how the annual storage percentage is spread across days.
    The divisor is a recorded assumption, so it is asserted rather than left to
    drift: at 360 the Groff figures come out at 19.44, not 19.18.
    """
    at_360 = PRICE * STORAGE_PCT / 360.0 * QTY * SPACING / 2.0
    ok = abs(at_360 - GROFF_ADDED_STORAGE) > 5e-3
    return ok, (f"a 360 day divisor gives {at_360:.2f} where SAP prints "
                f"{GROFF_ADDED_STORAGE:.2f}, so 365 is the right reading")


def run_all(log=print):
    log("SAP worked examples")
    results = [
        ("published lot sizes reproduce", *check_first_lot()),
        ("Groff's two comparison figures reproduce", *check_groff_figures()),
        ("both readings of Groff pass the published example",
         *check_groff_variants()),
        ("the four procedures give three distinct answers", *check_procedures_differ()),
        ("the storage divisor is pinned to 365", *check_storage_divisor()),
    ]
    ok = True
    for name, passed, detail in results:
        ok &= passed
        log(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        log(f"         {detail}")
    if not ok:
        raise SystemExit(
            "The SAP procedures do not reproduce SAP's published arithmetic. "
            "Nothing built on them is safe.")
    return results


if __name__ == "__main__":
    run_all()
