"""Lot-sizing rules for Dynamics 365 F&O, and the exact optimum as a bound.

SAP S/4HANA's procedures live in sap.py. This module holds F&O's coverage
codes, the Wagner and Whitin dynamic programme used to bound every rule, and
the cost function both products are scored on.

No F&O or SAP system is run. These are reimplementations of documented
behaviour. Each rule quotes the sentence it implements.

Some functions here were written for the companion Infor LN article and are
kept because this article still needs them: period_anchored and
period_anchored_at_need are the two receipt placements SAP offers and F&O does
not, and fixed_order_quantity is SAP's FX lot-size key as well as LN's Fixed
Order Quantity. economic_order_quantity is retained and unused.
"""
import numpy as np


# ---------------------------------------------------------------- lot sizing

def lot_for_lot(net):
    """One order per requirement.

    F&O *Requirement*: "the system creates a planned purchase, transfer, or
    production order per requirement for the product".
    LN order planning records planning data "on a second-by-second basis".
    """
    return net.copy()


def period_floating(net, L):
    """Bucket of L days that starts at the first demand, then re-anchors.

    F&O *Period*: "The period starts with the first demand of the item and
    covers the defined length in time. The next period starts with the next
    requirements of the item."
    """
    out = np.zeros_like(net)
    n = net.size
    i = 0
    while i < n:
        if net[i] <= 0:
            i += 1
            continue
        j = min(i + L, n)
        out[i] = net[i:j].sum()
        i = j
    return out


def period_anchored(net, L, anchor=0):
    """Bucket of L days aligned to a fixed calendar grid.

    LN plan period start may be "a specific day of the week" or "a specific day
    of the month" rather than Rolling. Receipt sits at the start of the plan
    period and covers that period's requirement.
    """
    out = np.zeros_like(net)
    n = net.size
    for s in range(-(anchor % L), n, L):
        a, b = max(s, 0), min(s + L, n)
        if a >= n or b <= 0:
            continue
        q = net[a:b].sum()
        if q > 0:
            out[a] = q
    return out


def period_anchored_at_need(net, L, anchor=0):
    """Fixed calendar buckets, but the receipt sits at the first demand inside
    each bucket rather than on the bucket boundary.

    This is the fair comparison against period_floating. period_anchored places
    the receipt on the boundary, which orders on average L/2 days early and
    therefore holds more stock by construction. That confounds two separate
    things: where the bucket BOUNDARY sits (the real difference between a
    calendar-anchored plan period and a floating one) and where the RECEIPT
    sits inside the bucket (an implementation choice). This function isolates
    the boundary.
    """
    out = np.zeros_like(net)
    n = net.size
    for s in range(-(anchor % L), n, L):
        a, b = max(s, 0), min(s + L, n)
        if a >= n or b <= 0:
            continue
        seg = net[a:b]
        q = seg.sum()
        if q > 0:
            first = a + int(np.flatnonzero(seg > 0)[0])
            out[first] = q
    return out


def period_varying(net, L_near, L_far, switch):
    """Short buckets near, long buckets far, both calendar-anchored.

    LN: "Plan periods can be of varying length, and are defined in calendar
    days, weeks, or months." Infor advises that larger plan periods "preferably
    should be an integer multiple of the smaller plan periods", which is the
    assertion below.
    """
    assert L_far % L_near == 0, "LN advises larger plan periods be an integer multiple of smaller"
    out = np.zeros_like(net)
    n = net.size
    sw = int(min(switch, n))
    if sw > 0:
        out[:sw] = period_anchored(net[:sw], L_near, anchor=0)
    if sw < n:
        out[sw:] = period_anchored(net[sw:], L_far, anchor=0)
    return out


def fixed_order_quantity(net, Q):
    """Every order is exactly Q. SAP lot-size key FX, Fixed lot size.

    SAP: "the system always plans the procurement quantity as the fixed lot
    size", creating multiple orders where one is not enough. Infor LN documents
    the same mechanism as Fixed Order Quantity.

    Dynamics 365 F&O has no equivalent coverage code. Its Default order
    settings carry Multiple, Min. order quantity and Max. order quantity, which
    modify a quantity the lot-sizing rule has already chosen; none of them is a
    lot-sizing rule that fixes the quantity itself.
    """
    out = np.zeros_like(net)
    if Q <= 0:
        return out
    inv = 0.0
    for d in range(net.size):
        need = net[d] - inv
        while need > 1e-9:
            out[d] += Q
            inv += Q
            need -= Q
        inv -= net[d]
        if inv < 0:
            inv = 0.0
    return out


def economic_order_quantity(net, setup_cost, hold_cost):
    """EOQ as a FLOOR on the order quantity, which is what LN documents.

    Infor LN: "If the order method is Economic Order Quantity (EOQ),
    Enterprise Planning sets the order quantities to at least the economic
    order quantity." So EOQ raises a lot-for-lot quantity up to the EOQ where
    the requirement is smaller, and leaves it alone where the requirement is
    larger. An earlier version of this function treated EOQ as a fixed lot,
    which is the Fixed Order Quantity method, not this one.

    The setup-to-holding ratio is a property of the business and is not in the
    data, so it is swept rather than chosen. F&O has no EOQ coverage code: its
    replenishment methods are Period, Requirement, Min./Max., Priority,
    Decoupling point and Manual.
    """
    d = float(net.sum()) / max(net.size, 1)
    if d <= 0 or hold_cost <= 0:
        return lot_for_lot(net)
    Q = float(np.sqrt(2.0 * setup_cost * d / hold_cost))
    out = np.zeros_like(net)
    inv = 0.0
    for i in range(net.size):
        need = net[i] - inv
        if need > 1e-9:
            q = max(need, Q)          # "at least the economic order quantity"
            out[i] = q
            inv += q
        inv -= net[i]
        if inv < 0:
            inv = 0.0
    return out


def min_max(net, onhand_start, mn, mx):
    """Replenish to max when projected on-hand falls below min.

    F&O *Min/Max*: "replenishes inventory up to a certain level when the
    predicted on-hand quantity is below a threshold. The replenishment quantity
    is the difference between the maximum level and the predicted on-hand
    level."
    """
    out = np.zeros_like(net)
    inv = float(onhand_start)
    for d in range(net.size):
        inv -= net[d]
        if inv < mn:
            q = mx - inv
            out[d] += q
            inv += q
    return out


# ---------------------------------------------------------------- benchmark

def wagner_whitin(dem, setup_cost, hold_cost):
    """Exact minimum-cost lot sizing (Wagner and Whitin 1958 dynamic programme).

    Used only as a provable lower bound to validate the engine: no rule above
    may cost less than this on the same series.

    The recursion runs over demand *occurrences*, not calendar periods. Under
    the zero-inventory property an optimal order is placed on a day that
    carries demand, so tiling the calendar and forcing each order to a block's
    first day is wrong: a block opening on a zero-demand day pays holding from
    too early and the DP returns a value above the true optimum.
    """
    t = np.flatnonzero(dem > 0)
    if t.size == 0:
        return 0.0
    d = dem[t]
    K = t.size
    f = np.full(K + 1, np.inf)
    f[0] = 0.0
    for k in range(1, K + 1):
        for i in range(1, k + 1):
            hold = float(np.dot(hold_cost * (t[i - 1:k] - t[i - 1]), d[i - 1:k]))
            c = f[i - 1] + setup_cost + hold
            if c < f[k]:
                f[k] = c
    return float(f[K])


def cost_of(orders, dem, setup_cost, hold_cost):
    """Ordering plus holding cost of a receipt schedule against a demand series."""
    inv = 0.0
    hold = 0.0
    for d in range(dem.size):
        inv += orders[d]
        inv -= dem[d]
        hold += hold_cost * max(inv, 0.0)
    return setup_cost * int((orders > 1e-9).sum()) + hold
