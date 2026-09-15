"""SAP S/4HANA lot-sizing procedures, reimplemented from SAP Help Portal.

No SAP system is run. These are reimplementations of documented behaviour, in
the same way engine.py reimplements Dynamics 365 F&O. Each function quotes the
sentence it implements, from SAP S/4HANA 2025 FPS01.

SAP groups lot-sizing into three families:

    "Three groups of lot-sizing procedures are available: Static lot-sizing
    procedures, Period lot-sizing procedures, Optimum lot-sizing procedures."

F&O has rules in the first two families and nothing in the third. The optimum
family is what this module is mainly for.

Every procedure here is gated against SAP's own published worked example. See
sap_examples.py. Nothing in this module should be trusted until that gate runs
green, because several of these rules are named after published heuristics
whose textbook forms differ from SAP's.
"""
import numpy as np


# --------------------------------------------------------------- cost basis

# SAP's optimum procedures price storage from three material master fields:
# price, the storage costs indicator (a percentage) and the lot-size-independent
# costs. The examples on the help pages are reproduced exactly by treating the
# storage percentage as an annual rate spread over 365 days. SAP does not state
# the divisor; 365 is what returns SAP's published figures, and 360 does not.
STORAGE_DAYS_PER_YEAR = 365.0


def holding_per_unit_day(price, storage_pct):
    """Storage cost of one unit for one day.

    storage_pct is SAP's storage costs indicator, a fraction (0.10 for 10%).
    """
    return price * storage_pct / STORAGE_DAYS_PER_YEAR


def _storage(h, offs, qs):
    """Total storage cost of a lot: each quantity held from the receipt date."""
    return h * float(np.dot(np.asarray(offs, float), np.asarray(qs, float)))


# ------------------------------------------------------- the grouping driver

def _group(net, accept):
    """Walk the net requirements, forming lots by a stopping rule.

    SAP: "The starting point for lot sizing is the first material shortage date
    that is determined during the net requirements calculation. The shortage
    quantity determined here represents the minimum order quantity. The system
    then adds successive shortage quantities to this lot size until, by means
    of the particular cost criterion, optimum costs have been established."

    So every optimum procedure is the same greedy sequential grouping, and
    "The only differences between the various optimum lot-sizing procedures are
    the cost criteria." That is the whole of `accept`.

    accept(offs, qs, t, d) -> bool, where offs and qs are the day offsets and
    quantities already in the lot, and (t, d) is the candidate being considered.
    The receipt always sits on the first shortage date of the lot.
    """
    out = np.zeros_like(net)
    idx = np.flatnonzero(net > 0)
    i = 0
    while i < idx.size:
        r = int(idx[i])
        offs, qs = [0], [float(net[r])]
        j = i + 1
        while j < idx.size:
            t, d = int(idx[j] - r), float(net[idx[j]])
            if not accept(offs, qs, t, d):
                break
            offs.append(t)
            qs.append(d)
            j += 1
        out[r] = sum(qs)
        i = j
    return out


# --------------------------------------------- optimum lot-sizing procedures
# SAP: "In static and period lot-sizing procedures, the costs resulting from
# stockkeeping, from the setup procedures or from purchasing are not taken into
# consideration. The aim of optimum lot-sizing procedures, on the other hand,
# is to group shortages together in such a way that costs are minimized."
#
# Dynamics 365 F&O has no coverage code in this family. Its replenishment
# methods are Per requirement, Per period, Min/Max, Priority, Decoupling point
# and Manual, none of which reads a cost.
#
# The S/4HANA help pages title these differently from the overview page that
# lists them. Both names are given.

def part_period_balancing(net, price, storage_pct, K):
    """SP. Part Period Balancing.

    "In part period balancing, starting from the shortage date, the system
    groups consecutive requirement quantities into one lot, until the total of
    the storage costs is the same as the lot-size-independent costs."

    The example stops *before* exceeding: "The most favorable lot size is 2000
    pieces, since with an additional requirements grouping, the total storage
    costs would be greater than the lot-size-independent costs."
    """
    h = holding_per_unit_day(price, storage_pct)
    return _group(net, lambda offs, qs, t, d: _storage(h, offs, qs) + h * d * t <= K)


def sliding_economic_lot_size(net, price, storage_pct, K):
    """WI. Sliding Economic Lot Size, listed as the Least Unit Cost Procedure.

    "With the sliding economic lot size, starting from the shortage date, the
    system groups consecutive requirement quantities into one lot size until
    the total costs per piece form a minimum. The total costs are the sum of
    the lot-size-independent costs and the total storage costs."

    Cost per piece falls and then rises, so the minimum is taken as the last
    quantity before it rises. That is a greedy descent, and it can stop at a
    local minimum where an exhaustive search would not. SAP's wording, "until
    the total costs per piece form a minimum", is read as the greedy form
    because the rest of the family is described as sequential grouping.
    """
    h = holding_per_unit_day(price, storage_pct)

    def accept(offs, qs, t, d):
        now = (K + _storage(h, offs, qs)) / sum(qs)
        then = (K + _storage(h, offs + [t], qs + [d])) / (sum(qs) + d)
        return then < now

    return _group(net, accept)


def dynamic_planning_calculation(net, price, storage_pct, K):
    """DY. Dynamic Planning Calculation, listed as Dynamic Lot Size Creation.

    "In the dynamic planning calculation, starting from the shortage date, the
    system groups requirement quantities into one lot until the additional
    storage costs incurred are greater than the lot-size-independent costs."

    "Additional" is the storage cost of the candidate quantity alone, not the
    running total. That is what separates DY from SP, and it is why DY returns
    3000 on the example where SP returns 2000.
    """
    h = holding_per_unit_day(price, storage_pct)
    return _group(net, lambda offs, qs, t, d: h * d * t <= K)


GROFF_STOP_RULE = "stop_rule"
GROFF_DAY_OFFSET = "day_offset"
GROFF_DEFAULT = GROFF_STOP_RULE


def groff(net, price, storage_pct, K, variant=GROFF_DEFAULT):
    """GR. Groff Reorder Procedure.

    "The Groff reorder procedure is based on the fact that additional storage
    costs are equal to the saving in lot size independent costs according to
    the classical lot sizing formula for the minimum costs."

    "Starting from a certain period, the system keeps grouping requirements
    into a lot until the increase in the average storage costs per period is
    larger than the decrease in the lot size independent costs per period."

    THE DOCUMENTATION DOES NOT SETTLE THIS RULE, and the two readings that fit
    it disagree by a factor of two to three on real demand. Both are here
    because picking one silently is how a wrong number reaches print.

    SAP publishes one worked example: K=100, price 20, storage 10 per cent,
    1000 units a week apart, savings 1.79 against additional storage 19.18,
    answer 1000. In that example the lot covers one requirement and the
    candidate sits seven days out, so the number of periods covered and the
    candidate's day offset are the same number. Every reading below returns
    1.79, 19.18 and 1000. A single example with one step cannot separate them.

      GROFF_STOP_RULE, the default. Groff's published rule, add while
          d <= 2K / (h * t * (t + 1)).
          This is the form in Groff (1979) and in Zoller and Robrade (1988),
          who compare it against the other heuristics and recommend it to
          software vendors by name.

      GROFF_DAY_OFFSET. Reads both sides literally off the printed table,
          comparing K/(t(t+1)) against h*d*t/2, which carries an extra factor
          of t and therefore stops much earlier.

    The default is the published stop rule, on three grounds. SAP names the
    procedure after Groff. The day-offset reading leaves GR dominated by SP at
    every order cost tested, which is not a procedure a vendor ships as a
    separate option. And the published rule is what the literature that SAP is
    referring to actually says.

    It stays an assumption. On this catalogue GR costs 1.12 to 1.53 times the
    Wagner and Whitin optimum under the stop rule and 1.42 to 3.83 times under
    the day-offset reading, so no claim about GR is safe without naming which
    was used. Limits says so.
    """
    h = holding_per_unit_day(price, storage_pct)

    if variant == GROFF_STOP_RULE:
        def accept(offs, qs, t, d):
            return h * d * t * (t + 1.0) / 2.0 <= K
    elif variant == GROFF_DAY_OFFSET:
        def accept(offs, qs, t, d):
            return K / (t * (t + 1.0)) > h * d * t / 2.0
    else:
        raise ValueError(variant)

    return _group(net, accept)


def groff_figures(K, h, d, t):
    """The two figures SAP prints in the Groff table, for one step.

    Both readings agree on these, which is why they cannot be told apart on
    the published example. Kept separate so the gate can assert them without
    going through whichever variant is currently the default.
    """
    saving = K / (t * (t + 1.0))
    added = h * d * t / 2.0
    return saving, added


OPTIMUM = {
    "SP": part_period_balancing,
    "WI": sliding_economic_lot_size,
    "DY": dynamic_planning_calculation,
    "GR": groff,
}

OPTIMUM_NAMES = {
    "SP": "Part Period Balancing",
    "WI": "Sliding Economic Lot Size",
    "DY": "Dynamic Planning Calculation",
    "GR": "Groff Reorder Procedure",
}


# ------------------------------------------------ period lot-sizing, anchored
# SAP: "In period lot-sizing procedures, the system groups several requirements
# within a time interval together to form a lot." The intervals are days,
# weeks, months, posting periods, or freely definable periods from a planning
# calendar. They are calendar intervals.
#
# F&O's Per period is not: "The period starts with the first demand of the item
# and covers the defined length in time. The next period starts with the next
# requirements of the item." It floats with demand. engine.period_floating is
# that rule; the functions below are SAP's.

FIRST_REQUIREMENT = "first"
PERIOD_START = "start"
PERIOD_END = "end"
PLACEMENTS = (FIRST_REQUIREMENT, PERIOD_START, PERIOD_END)


def period_calendar(net, L, placement=FIRST_REQUIREMENT, anchor=0):
    """Calendar buckets of L days, with the receipt at one of three positions.

    "The system sets the availability date for period lot-sizing procedures to
    the first requirements date of the period. However, you can also define
    that the availability date is at the beginning or end of the period."

    Set by the Customizing scheduling indicator. F&O has no equivalent: its
    receipt is always at the first demand, and the bucket is never anchored.

    PERIOD_END places the receipt after demand earlier in the bucket has
    already been required. That is what the setting does, and the replay is
    left to show what it costs rather than the rule refusing to produce it.
    """
    assert placement in PLACEMENTS, placement
    out = np.zeros_like(net)
    n = net.size
    for s in range(-(anchor % L), n, L):
        a, b = max(s, 0), min(s + L, n)
        if a >= n or b <= 0:
            continue
        seg = net[a:b]
        q = float(seg.sum())
        if q <= 0:
            continue
        if placement == PERIOD_START:
            day = a
        elif placement == PERIOD_END:
            day = b - 1
        else:
            day = a + int(np.flatnonzero(seg > 0)[0])
        out[day] += q
    return out


# --------------------------------------------- short-term and long-term lot size

def long_term_valid_from(periods, L, start=0):
    """The day the long-term lot size takes over.

    "The start of the effectivity period of the long-term lot size is
    determined using the number of periods specified in Customizing. The system
    always rounds up to the end of a period that has already started and then
    sets the valid from date to the beginning of the next complete period."

    SAP's worked example: the calculation date is 15 March, four months are
    configured, the system counts to 15 July and rounds up, so the long-term
    lot size is valid from 1 August.

    Here periods are L days long on the same calendar grid the anchored buckets
    use, so counting `periods` forward from `start` and rounding up to the next
    whole period is the same arithmetic on a day index.
    """
    raw = start + periods * L
    return int(np.ceil(raw / float(L)) * L)


def short_long(net, near, far, switch):
    """A different lot-sizing procedure near-term and far-term.

    "If you define a short-term and a long-term lot size for a material, you
    can split up the time axis for the material requirements planning into a
    short-term and a long-term area and thus carry out the procurement quantity
    calculation using two different lot-sizing procedures."

    "During the planning run the system calculates the time axis for the
    validity of both lot-sizing procedures. The procurement quantities in the
    first, short-term area are calculated using the short-term lot size. From
    the valid from date of the long-term lot size the system switches to the
    long-term lot size."

    "If no long-term lot size has been specified for a lot-sizing procedure,
    the system plans the complete planning period using the short-term lot
    size." That is the `far is None` branch.

    F&O carries one coverage code per item across the whole horizon.

    near and far are callables taking a net-requirement array and returning a
    receipt array.
    """
    if far is None:
        return near(net)
    n = net.size
    sw = int(min(max(switch, 0), n))
    out = np.zeros_like(net)
    if sw > 0:
        out[:sw] = near(net[:sw])
    if sw < n:
        out[sw:] = far(net[sw:])
    return out
