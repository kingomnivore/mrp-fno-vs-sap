"""Fetch, filter and classify. Everything downstream reads from here.

The source workbook is committed through Git LFS, so the repository runs as
cloned. It is downloaded only if missing, then cached as a daily demand
matrix. The ONS series is committed as plain CSV.
"""
import os
import re
import zipfile
import urllib.request

import numpy as np
import pandas as pd

import config as C


# ------------------------------------------------------------------ fetch

def fetch_source():
    """Download the UCI workbook once. Returns the local path."""
    if os.path.exists(C.UCI_ZIP):
        return C.UCI_ZIP
    os.makedirs(os.path.dirname(C.UCI_ZIP), exist_ok=True)
    print("  downloading Online Retail II (44MB, only if missing)...")
    urllib.request.urlretrieve(C.UCI_URL, C.UCI_ZIP)
    return C.UCI_ZIP


def _read_workbook(path):
    z = zipfile.ZipFile(path)
    name = z.namelist()[0]
    import io as _io
    raw = _io.BytesIO(z.read(name))
    xl = pd.ExcelFile(raw, engine="openpyxl")
    frames = [pd.read_excel(raw, sheet_name=s, engine="openpyxl")
              for s in xl.sheet_names]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ------------------------------------------------------------------ load

def daily_demand():
    """Daily demand per item, as a date-by-item matrix.

    Credit invoices and negative quantities are excluded, because a return is
    not negative demand. It is supply arriving on a date nobody planned.
    """
    if os.path.exists(C.UCI_CACHE):
        d = pd.read_csv(C.UCI_CACHE, index_col=0, parse_dates=True)
        d.columns = [str(c) for c in d.columns]
        return d

    df = _read_workbook(fetch_source())
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Invoice"] = df["Invoice"].astype(str).str.strip().str.upper()
    df["Date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()

    keep = (df["StockCode"].str.match(C.ITEM_CODE_PATTERN)
            & ~df["Invoice"].str.startswith("C")
            & (df["Quantity"] > 0))
    piv = df[keep].pivot_table(index="Date", columns="StockCode",
                               values="Quantity", aggfunc="sum", fill_value=0)
    piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max(), freq="D"),
                      fill_value=0)
    piv.to_csv(C.UCI_CACHE)
    return piv


def unit_prices():
    """Median unit price per item, in sterling.

    SAP's optimum lot-sizing procedures price storage as a percentage of the
    material price, so they need a price per unit. Online Retail II carries
    one on every invoice line. The median is taken rather than the mean
    because the same stock code is sold at different prices to different
    customers, and a handful of lines carry a price of zero or a price far
    above the rest.

    Lines with a non-positive price are dropped before the median is taken.
    The same invoice and stock code filters used for demand are applied, so
    the price population matches the demand population.
    """
    if os.path.exists(C.UCI_PRICE_CACHE):
        d = pd.read_csv(C.UCI_PRICE_CACHE, index_col=0)
        d.index = [str(i) for i in d.index]
        return d["price"]

    df = _read_workbook(fetch_source())
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Invoice"] = df["Invoice"].astype(str).str.strip().str.upper()
    price_col = "Price" if "Price" in df.columns else "UnitPrice"
    keep = (df["StockCode"].str.match(C.ITEM_CODE_PATTERN)
            & ~df["Invoice"].str.startswith("C")
            & (df["Quantity"] > 0)
            & (df[price_col] > 0))
    med = df[keep].groupby("StockCode")[price_col].median()
    med.name = "price"
    med.to_csv(C.UCI_PRICE_CACHE)
    return med


def source_summary():
    """Row counts and the anomaly described in the article."""
    df = _read_workbook(fetch_source())
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Invoice"] = df["Invoice"].astype(str).str.strip().str.upper()
    item = df["StockCode"].str.match(C.ITEM_CODE_PATTERN)
    credit = df["Invoice"].str.startswith("C")
    neg = df["Quantity"] < 0
    return {
        "rows": len(df),
        "codes": int(df["StockCode"].nunique()),
        "non_item_codes": int(df.loc[~item, "StockCode"].nunique()),
        "negative_rows": int(neg.sum()),
        "credit_rows": int(credit.sum()),
        "negative_outside_credits": int((neg & ~credit).sum()),
        "date_min": str(pd.to_datetime(df["InvoiceDate"]).min().date()),
        "date_max": str(pd.to_datetime(df["InvoiceDate"]).max().date()),
        "largest_credit": int(df["Quantity"].min()),
    }


# -------------------------------------------------------------- classify

def quadrant(adi, cv2):
    if adi < C.ADI_CUT and cv2 < C.CV2_CUT:
        return "smooth"
    if adi >= C.ADI_CUT and cv2 < C.CV2_CUT:
        return "intermittent"
    if adi < C.ADI_CUT and cv2 >= C.CV2_CUT:
        return "erratic"
    return "lumpy"


def active_items(daily):
    """Items passing the activity filter."""
    a = daily.to_numpy().astype(float)
    hit = a > 0
    occ = hit.sum(axis=0)
    any_hit = hit.any(axis=0)
    first = np.where(any_hit, hit.argmax(axis=0), -1)
    last = a.shape[0] - 1 - np.where(any_hit, hit[::-1].argmax(axis=0), 0)
    span = last - first + 1
    keep = (occ >= C.MIN_DEMAND_DAYS) & (span >= C.MIN_ACTIVE_DAYS)
    return np.array(daily.columns)[keep]


def classify(daily, codes, period):
    """ADI and CV-squared for each item, measured in `period` buckets.

    SBC define the interval in review periods, not days, so the period is part
    of the measure rather than a presentation choice.
    """
    rule = C.REVIEW_PERIODS[period]
    agg = daily.loc[:, codes] if rule == "D" else daily.loc[:, codes].resample(rule).sum()
    A = agg.to_numpy().astype(float)
    rows = []
    for j, code in enumerate(codes):
        col = A[:, j]
        nz = np.flatnonzero(col > 0)
        if nz.size < 2:
            continue
        win = col[nz[0]:nz[-1] + 1]
        idx = np.flatnonzero(win > 0)
        if idx.size < 2:
            continue
        adi = float(np.mean(np.diff(idx)))
        sizes = win[win > 0]
        if sizes.size < 2 or sizes.mean() <= 0:
            continue
        cv2 = float((sizes.std(ddof=1) / sizes.mean()) ** 2)
        rows.append((code, adi, cv2, quadrant(adi, cv2)))
    return pd.DataFrame(rows, columns=["StockCode", "ADI", "CV2", "quadrant"])


def sample(profile, daily, per_quadrant=None):
    """Equal-weight sample across the four demand classes."""
    n = per_quadrant or C.PER_QUADRANT
    out = []
    for q in ("smooth", "intermittent", "erratic", "lumpy"):
        pool = profile[(profile["quadrant"] == q)
                       & (profile["StockCode"].isin(daily.columns))]
        if len(pool):
            out.append(pool.sample(min(n, len(pool)),
                                   random_state=C.SAMPLE_SEED).assign(quadrant=q))
    return pd.concat(out, ignore_index=True)


def series(daily, code):
    return daily[code].to_numpy().astype(float)
