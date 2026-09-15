"""The external check: does this firm's order book move with its own sector?

One wholesaler is not a sector, but a giftware wholesaler's revenue should
track the national index for non-store retailing. If it does not, the date
handling or the aggregation is wrong and nothing downstream is safe.

The series used is J596, value NOT seasonally adjusted. The seasonally
adjusted twin of the same series would defeat the check, because the check
turns on whether the seasonal pattern matches.
"""
import re

import numpy as np
import pandas as pd

import config as C
import data as D

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _dataset_monthly():
    df = D._read_workbook(D.fetch_source())
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    df["Invoice"] = df["Invoice"].astype(str).str.strip().str.upper()
    keep = (df["StockCode"].str.match(C.ITEM_CODE_PATTERN)
            & ~df["Invoice"].str.startswith("C")
            & (df["Quantity"] > 0))
    d = df[keep].copy()
    d["revenue"] = d["Quantity"] * d["Price"]
    d["Date"] = pd.to_datetime(d["InvoiceDate"])
    return d.groupby(d["Date"].dt.to_period("M"))["revenue"].sum()


def _ons_monthly(path=None):
    rows = []
    for line in open(path or C.ONS_CSV, encoding="utf-8-sig"):
        parts = [p.strip().strip('"') for p in line.strip().split('","')]
        parts = [p.strip('"') for p in parts]
        if len(parts) != 2:
            continue
        m = re.match(r"^(\d{4}) ([A-Z]{3})$", parts[0])
        if not m or m.group(2) not in MONTHS:
            continue
        try:
            rows.append((pd.Period(f"{m.group(1)}-{MONTHS.index(m.group(2)) + 1:02d}", "M"),
                         float(parts[1])))
        except ValueError:
            continue
    s = pd.Series(dict(rows)).sort_index()
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s


def run():
    ds = _dataset_monthly()
    # The file opens on 1 December 2009 and stops on 9 December 2011, so
    # neither December is a whole month and both are dropped.
    full = ds.iloc[1:-1]
    j = pd.DataFrame({"dataset": full}).join(
        pd.DataFrame({"ons": _ons_monthly()}), how="inner")

    r_lev = float(np.corrcoef(j["dataset"], j["ons"])[0, 1])
    r_log = float(np.corrcoef(np.log(j["dataset"]), np.log(j["ons"]))[0, 1])
    by_month_ds = j.groupby(j.index.month)["dataset"].mean()
    by_month_on = j.groupby(j.index.month)["ons"].mean()
    return {
        "months": len(j),
        "from": str(j.index.min()),
        "to": str(j.index.max()),
        "r_levels": r_lev,
        "r_logs": r_log,
        "peak_dataset": MONTHS[int(by_month_ds.idxmax()) - 1],
        "peak_ons": MONTHS[int(by_month_on.idxmax()) - 1],
        "nov_ratio_dataset": float(by_month_ds[11] / by_month_ds.mean()),
        "nov_ratio_ons": float(by_month_on[11] / by_month_on.mean()),
    }
