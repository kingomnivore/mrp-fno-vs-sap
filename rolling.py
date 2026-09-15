"""Rolling regeneration: replan every period on a moving demand signal.

Timing convention, which the first version of this file got wrong:

  receipt date  = the day stock is needed and arrives
  release date  = receipt date - lead time
  an order is committed when its RELEASE date falls in the current period

Committing by receipt date instead of release date means nothing is ever
ordered in time, which shows up as a fill rate near zero. Self-tests [7] and
[8] exist to catch exactly that class of error.
"""
import zlib

import numpy as np
import engine as E
import sap as S


def stable_seed(*parts):
    """A seed that is identical across interpreter sessions.

    Python salts hash() for strings unless PYTHONHASHSEED is fixed, so seeding
    with hash() makes a run irreproducible on the next process. That defeats
    the point of publishing the code.
    """
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0xFFFFFFFF


TOL = 0.5          # a planned quantity has "changed" if it moves by more than this
STAB_WINDOW = 90   # days beyond today over which plan stability is measured


def build_signal(actual, t0, visibility, err, coverage_fence):
    """The demand signal a planning run sees on day t0.

    Inside the firm visibility window the signal is the actual demand. Beyond
    it, it carries the forecast error `err`, which the caller evolves between
    runs. How that error evolves is the single assumption every churn result
    in this analysis rests on, which is why it is a parameter and not a
    constant. See `evolve_error`.
    """
    T = actual.size
    sig = np.zeros(T)
    sig[t0:] = actual[t0:]
    if err is not None:
        far = np.arange(T) > t0 + visibility
        sig = np.where(far, np.maximum(sig * (1.0 + err), 0.0), sig)
    if coverage_fence is not None:
        # F&O: "The system doesn't generate requirement transactions for any
        # supply and demand that falls outside the coverage time fence."
        sig = np.where(np.arange(T) > t0 + coverage_fence, 0.0, sig)
    sig[:t0] = 0.0
    return sig


def evolve_error(err, sigma, rho, rng, size):
    """Advance the forecast error one planning cycle.

    rho = 0 redraws the error independently at every run, so a day's forecast
    is unrelated to what it was last week. rho -> 1 makes the error persist,
    so a forecast drifts rather than jumping. Real forecasts sit between the
    two, and the choice changes measured churn a great deal, so both are run.
    """
    if sigma <= 0:
        return None
    fresh = rng.normal(0.0, sigma, size)
    if err is None or rho <= 0:
        return fresh
    return rho * err + np.sqrt(max(1.0 - rho * rho, 0.0)) * fresh


def net_requirements(sig, committed, inv_now, t0):
    """Walk projected on-hand forward, netting committed arrivals at their dates."""
    net = np.zeros_like(sig)
    inv = float(inv_now)
    for d in range(t0, sig.size):
        inv += committed[d]
        inv -= sig[d]
        if inv < 0.0:
            net[d] = -inv
            inv = 0.0
    return net


def lot_size(net, cfg, avail, t0):
    k = cfg["kind"]
    if k == "l4l":
        return E.lot_for_lot(net)
    if k == "float":
        return E.period_floating(net, cfg["L"])
    if k == "anch":
        return E.period_anchored(net, cfg["L"], anchor=cfg.get("anchor", 0))
    if k == "anch_need":
        return E.period_anchored_at_need(net, cfg["L"], anchor=cfg.get("anchor", 0))
    if k == "vary":
        out = np.zeros_like(net)
        sw = int(min(t0 + cfg["switch"], net.size))
        out[:sw] = E.period_anchored(net[:sw], cfg["L_near"], 0)
        out[sw:] = E.period_anchored(net[sw:], cfg["L_far"], 0)
        return out
    if k == "minmax":
        return E.min_max(net, avail, cfg["mn"], cfg["mx"])
    if k == "foq":
        return E.fixed_order_quantity(net, cfg["Q"])
    if k == "eoq":
        return E.economic_order_quantity(net, cfg["setup"], cfg["hold"])

    # ------------------------------------------------------------- SAP
    # The calendar grid below is indexed on the absolute day, not on t0, so a
    # bucket boundary stays where it is from one planning run to the next.
    # That is what makes a calendar-anchored bucket a calendar bucket; keying
    # it to the run date would turn it back into a floating one.
    if k == "sap_opt":
        return S.OPTIMUM[cfg["code"]](net, cfg["price"], cfg["storage_pct"],
                                      cfg["K"])
    if k == "sap_cal":
        return S.period_calendar(net, cfg["L"], cfg.get("placement",
                                 S.FIRST_REQUIREMENT), cfg.get("anchor", 0))
    if k == "sap_sl":
        near = _sap_sub(cfg["near"])
        far = _sap_sub(cfg["far"])
        switch = S.long_term_valid_from(cfg["periods"], cfg["near"]["L"], t0)
        return S.short_long(net, near, far, switch)
    raise ValueError(k)


def _sap_sub(sub):
    """Bind one half of a short-term/long-term pair to a callable.

    SAP: "If no long-term lot size has been specified for a lot-sizing
    procedure, the system plans the complete planning period using the
    short-term lot size." A None here is that case.
    """
    if sub is None:
        return None
    if sub["kind"] == "sap_cal":
        return lambda n: S.period_calendar(
            n, sub["L"], sub.get("placement", S.FIRST_REQUIREMENT),
            sub.get("anchor", 0))
    if sub["kind"] == "l4l":
        return E.lot_for_lot
    if sub["kind"] == "float":
        return lambda n: E.period_floating(n, sub["L"])
    raise ValueError(sub["kind"])


def simulate(actual, cfg, scenario, seed):
    """Replay one item under one configuration. Returns metrics."""
    T = actual.size
    LT = scenario["lead_time"]
    step = scenario["replan_every"]
    vis = scenario["visibility"]
    sigma = scenario["noise"]
    rho = scenario.get("noise_persistence", 0.0)
    fence = scenario.get("coverage_fence")
    freeze = cfg.get("freeze", 0)

    # Discard a warm-up before measuring. The first run has to cover every
    # requirement inside the lead time at once, which puts a lump of stock on
    # the books that has nothing to do with the steady-state behaviour of the
    # rule. Without this, a 60 day lead time reported roughly twice the cover
    # it actually settles at.
    warm = int(scenario.get("warmup_days", 2 * LT + 60))
    rng = np.random.default_rng(seed)
    mean_daily = actual.mean()
    inv = float(scenario.get("opening_stock_days", 0)) * mean_daily
    committed = np.zeros(T)     # quantity ARRIVING on each day, already firm
    unmet = 0.0
    inv_trace = []
    prev_plan = None
    churn_num = churn_den = 0
    ochurn_num = ochurn_den = 0
    n_orders = 0
    served = 0.0
    n_days = 0
    err = None

    for t0 in range(0, T, step):
        # --- plan on the state as at the start of the period
        err = evolve_error(err, sigma, rho, rng, T)
        sig = build_signal(actual, t0, vis, err, fence)
        net = net_requirements(sig, committed, inv, t0)
        receipts = lot_size(net, cfg, inv + committed[t0:].sum(), t0)

        # a receipt cannot be earlier than today + lead time; pull-ins pile up
        # on the earliest feasible day, which is how a delayed requirement behaves
        earliest = min(t0 + LT, T - 1)
        if earliest > 0:
            late = receipts[:earliest].sum()
            receipts[:earliest] = 0.0
            receipts[earliest] += late

        # --- plan stability against the previous run
        #
        # Measured over the ACTIONABLE horizon only, which starts after the
        # lead time. A requirement that falls inside the lead time cannot be
        # rescheduled by the planner; it is a delay, which both products treat
        # as a separate signal. Including it would score the lead-time pull-in
        # as churn: with a perfect forecast that alone produced 1.5% PCR at a
        # five-day lead time and exactly 0.0% at a zero-day lead time.
        if prev_plan is not None:
            a, b = t0 + LT + 1, min(t0 + LT + 1 + STAB_WINDOW, T)
            if b > a:
                if freeze > 0 and a < b:
                    fz = int(min(t0 + freeze, T))
                    if fz > a:
                        receipts[a:fz] = prev_plan[a:fz]
                # PCR honours rel_tol for the same reason order churn
                # does. An earlier version compared against TOL
                # unconditionally, so running the two tolerances
                # returned identical PCR and the dual-tolerance test
                # was silently comparing the absolute measure with
                # itself. That reads as perfect agreement.
                _rel = scenario.get("rel_tol")
                if _rel:
                    _po, _cu = prev_plan[a:b], receipts[a:b]
                    _base = np.maximum(np.maximum(_po, _cu), 1.0)
                    changed = (np.abs(_cu - _po) / _base) > _rel
                else:
                    changed = np.abs(receipts[a:b] - prev_plan[a:b]) > TOL
                churn_num += int(changed.sum())
                churn_den += int(b - a)
                # Order-level churn: what a planner actually opens. Counting
                # changed DAYS biases towards items with dense demand, because
                # they have more non-empty cells to differ. Counting changed
                # ORDERS does not, and it is what an action message is.
                po, cu = prev_plan[a:b], receipts[a:b]
                union = int(((po > TOL) | (cu > TOL)).sum())
                if union:
                    # An absolute tolerance flags any change on a large-quantity
                    # item, so it can distort a result about size variability.
                    # rel_tol compares the change against the previous quantity.
                    rel = scenario.get("rel_tol")
                    if rel:
                        base = np.maximum(np.maximum(po, cu), 1.0)
                        moved = (np.abs(cu - po) / base) > rel
                    else:
                        moved = np.abs(cu - po) > TOL
                    ochurn_num += int(moved.sum())
                    ochurn_den += union
        prev_plan = receipts.copy()

        # --- commit orders whose RELEASE date falls inside this period
        for rel in range(t0, min(t0 + step, T)):
            d = rel + LT
            if d < T and receipts[d] > TOL and committed[d] <= TOL:
                committed[d] += receipts[d]
                if rel >= warm:
                    n_orders += 1

        # --- execute the period: receive, then serve demand
        for d in range(t0, min(t0 + step, T)):
            inv += committed[d]
            inv -= actual[d]
            if inv < 0.0:
                if d >= warm:
                    unmet += -inv
                inv = 0.0
            if d >= warm:
                inv_trace.append(inv)
                served += actual[d]
                if d >= warm:
                    n_days += 1

    total = served
    return {
        "PCR": churn_num / churn_den if churn_den else 0.0,
        "order_churn": ochurn_num / ochurn_den if ochurn_den else 0.0,
        "fill": 1.0 - (unmet / total) if total > 0 else 1.0,
        "cover_days": (float(np.mean(inv_trace)) / mean_daily) if mean_daily > 0 else 0.0,
        "orders_per_year": n_orders * 365.0 / max(T - warm, 1),
        # Raw ingredients of the realised setup-plus-holding cost, kept
        # separate from any price so the same replay can be costed at
        # several order costs without being run again. SAP's optimum
        # procedures minimise this quantity, so the article has to report
        # it or it is not testing them on their own terms.
        "n_orders": n_orders,
        "unit_days": float(np.sum(inv_trace)),
        "measured_days": n_days,
    }
