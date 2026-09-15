"""Figure styling.

Transparent backgrounds, recessive axes. Colour follows meaning, not rank:
blue is a setting F&O can reach, orange is one only SAP can express, grey is
context. The same colour means the same thing in every figure.
"""
import matplotlib as mpl

C_FNO = "#2a78d6"      # reachable in F&O today
C_SAP = "#eb6834"      # expressible only in SAP S/4HANA
C_BOTH = "#6b7a86"     # available in both
C_WARN = "#c0392b"

INK = "#0b0b0b"
INK_2 = "#52514e"
RULE = "#c9c8c4"

QUAD = {"smooth": "#2a78d6", "intermittent": "#4aa3a2",
        "erratic": "#e8a33d", "lumpy": "#eb6834"}


def apply():
    mpl.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "legend.frameon": False,
    })


def clean(ax, value_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if value_axis:
        ax.grid(axis=value_axis, color=RULE, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
