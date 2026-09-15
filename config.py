"""Every tunable value in one place. No logic lives here.

Values marked JUDGEMENT are choices, not derivations. Each is named in the
article's Limits section, and moving any of them moves some conclusion.
"""

# ----------------------------------------------------------------- data
UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
UCI_ZIP = "Dataset/UCI/online_retail_II.zip"
UCI_CACHE = "Dataset/UCI/daily_demand.csv"
UCI_PRICE_CACHE = "Dataset/UCI/unit_price.csv"
ONS_CSV = "Dataset/ONS/j596.csv"

# Rows whose stock code is five digits with an optional letter suffix are real
# items. The other 61 codes are postage, adjustments, bank charges and samples.
ITEM_CODE_PATTERN = r"^\d{5}[A-Za-z]*$"

# ----------------------------------------------------- classification
# Syntetos, Boylan and Croston (2005), Table 1 and Figure 3. Derived by those
# authors for selecting a forecasting method, not a lot-sizing rule; using them
# to segment coverage codes is this analysis extending their work.
ADI_CUT = 1.32
CV2_CUT = 0.49

# JUDGEMENT. An item must be active enough to replay at all.
MIN_DEMAND_DAYS = 12
MIN_ACTIVE_DAYS = 180

# JUDGEMENT. The review period the headline classification is measured in.
# Section 4 shows the classification moves a great deal with this choice.
REVIEW_PERIODS = {"daily": "D", "weekly": "W-MON", "4-weekly": "28D", "monthly": "ME"}
HEADLINE_PERIOD = "weekly"

# JUDGEMENT. Items drawn per demand class, so the classes carry equal weight
# in the results despite their unequal size in the file.
PER_QUADRANT = 50
SAMPLE_SEED = 42

# ----------------------------------------------------------- the replay
REPLAN_EVERY = 7          # JUDGEMENT. Weekly regeneration.
VISIBILITY_DAYS = 28      # JUDGEMENT. Firm demand inside this window.
OPENING_STOCK_DAYS = 14   # JUDGEMENT.
STABILITY_WINDOW = 90     # JUDGEMENT. Actionable horizon for plan churn.

# Coverage periods tested, in days.
PERIODS = (7, 14, 28, 56)
# Infor LN Fixed Order Quantity, expressed as days of mean demand so it is
# comparable across items of very different volume.
FOQ_DAYS = (7, 28)
# Infor LN Economic Order Quantity. The setup-to-holding ratio is a property
# of the business and is not in the data, so it is swept.
EOQ_RATIOS = (10, 100, 1000)

# --------------------------------------------------- SAP lot-sizing costs
# SAP's optimum procedures read three material master fields: price, the
# storage costs indicator and the lot-size-independent costs. Only price is in
# the data. The other two are swept, because H1 is precisely the question of
# how much the answer depends on knowing them.

# JUDGEMENT. Annual storage cost as a percentage of material price. 20 to 25
# per cent is the range normally carried in the inventory literature.
STORAGE_PCT = 0.20
STORAGE_PCT_SWEEP = (0.10, 0.20, 0.30)

# JUDGEMENT. Lot-size-independent cost of raising one order, in sterling.
# Nothing in the data sets this. The sweep spans two orders of magnitude, so
# the optimum procedures are tested where they should win and where they
# should not.
ORDER_COST = 1.0
ORDER_COST_SWEEP = (0.25, 1.0, 5.0, 15.0, 50.0)
# The sweep is calibrated against what it implies, not guessed. On this
# catalogue the EOQ-implied cycle at each value has a median of 18, 35, 79,
# 136 and 248 days. The low end lines the optimum procedures up against the
# 7 to 56 day coverage periods F&O is tested at; the high end is a realistic
# fully loaded cost of raising one purchase order, where 98 per cent of
# items imply a cycle longer than any coverage period tested.

# JUDGEMENT. How far the assumed order cost is allowed to be wrong before the
# optimum procedures are judged on it. A factor of two either way.
MISSPECIFY = (0.5, 2.0)

# SAP calendar bucket lengths, in days. Weekly and monthly are the buckets SAP
# names; 14 and 56 match the coverage periods already tested for F&O.
SAP_BUCKETS = (7, 14, 28, 56)

# JUDGEMENT. Where the calendar grid starts, as a day offset. The series opens
# on a Tuesday, so 0 anchors buckets to that weekday. Section on anchoring
# sweeps this, because a weekly bucket that starts on the wrong day is a
# different rule.
SAP_ANCHORS = (0, 1, 2, 3, 4, 5, 6)

# JUDGEMENT. Periods counted forward before the long-term lot size takes over.
SHORT_LONG_PERIODS = 4
SHORT_LONG_FAR = (28, 56)

# ------------------------------------------------------- business cases
# JUDGEMENT throughout. These are constructed to span situations a planner
# would recognise. No source claims they are typical.
CASES = {
    "Fast-moving distributor": dict(
        lead_time=5, noise=0.10, classes=["smooth", "erratic"]),
    "Long-lead importer": dict(
        lead_time=60, noise=0.20, classes=None),
    "Spare parts operation": dict(
        lead_time=45, noise=0.30, classes=["intermittent", "lumpy"]),
    "Seasonal wholesaler": dict(
        lead_time=20, noise=0.20, classes=None),
    "Volatile demand": dict(
        lead_time=14, noise=0.40, classes=["erratic", "lumpy"]),
}

# JUDGEMENT. A category is left unranked where the spread across configurations
# is too small to rank on without manufacturing the ranking.
MIN_SPREAD = {"Service": 0.5, "Working capital": 1.0, "Planner workload": 1.0}
# JUDGEMENT. Below this margin, on a 0-100 scale, a case is called a tie.
TIE_MARGIN = 3.0

# ---------------------------------------------------------- fence sweeps
FENCE_LEADS = (5, 15, 30, 45, 60, 75, 90)
FENCE_VALUES = (15, 30, 45, 60, 80, 100, 120)
FENCE_PERIOD = 14                       # coverage period for the coarse grid
FINE_LEAD = 60                          # lead time for the decisive sweep
FINE_FENCES = tuple(range(60, 131, 5))
FINE_PERIODS = (7, 14, 28, 56)
FENCE_ITEMS = 120

# ------------------------------------------------- forecast error sweep
# Forecast error is generated, not observed. Every plan-stability number
# depends on it, which is why it is swept rather than fixed.
SIGMAS = (0.0, 0.10, 0.20, 0.30, 0.40)
PERSISTENCE = (0.0, 0.5, 0.8)
BASE_SIGMA = 0.20

# -------------------------------------------------------------- outputs
OUTPUTS = "outputs.txt"
