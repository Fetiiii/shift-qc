"""Portable frozen figure generator; result rows are external inputs."""
from __future__ import annotations
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "figure1"
LEDGER: dict[str, dict] = {}

MNMS = "#2a78d6"
ABDO = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

_consumed: set[str] = set()
_caption_fragments: dict[str, list[str]] = {}

def val(result_id: str) -> float:
    if result_id not in LEDGER:
        raise RuntimeError(
            f"Missing frozen result row {result_id}; result figures require "
            "the separately cleared frozen ledger."
        )
    return float(LEDGER[result_id]["full_precision"])


def shown(result_id: str) -> str:
    if result_id not in LEDGER:
        raise RuntimeError(
            f"Missing frozen result row {result_id}; result figures require "
            "the separately cleared frozen ledger."
        )
    return LEDGER[result_id]["display"]


def _interval(ax, y, low, high, point, colour, marker="o"):
    ax.plot([low, high], [y, y], color=colour, linewidth=2.0,
            solid_capstyle="butt", zorder=2)
    for edge in (low, high):
        ax.plot([edge, edge], [y - 0.11, y + 0.11], color=colour, linewidth=2.0,
                zorder=2)
    ax.plot([point], [y], marker=marker, markersize=8, color=colour,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)

def _save(fig, stem: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    pdf = FIG / f"{stem}.pdf"
    fig.savefig(pdf, format="pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIG / f"{stem}.svg", format="svg", metadata={"Date": None})
    fig.savefig(FIG / f"{stem}.png", format="png", dpi=200)
    plt.close(fig)
    return pdf

def figure1() -> Path:
    """Conceptual relation between the two estimands. Carries no scientific value."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 54)
    ax.axis("off")

    def box(x, y, w, h, text, edge=MUTED, fs=8.0, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.0,rounding_size=1.4",
                                    linewidth=0.9, edgecolor=edge, facecolor=SURFACE,
                                    zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=INK, zorder=3, linespacing=1.45, fontweight=weight)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=0.9, color=MUTED,
                                     shrinkA=2, shrinkB=2, zorder=1))

    def tie(x1, y1, x2, y2):
        """Shared-reference connector. Deliberately not an arrow: the comparator
        is a reference both estimands are measured against, not a step in a flow,
        and an arrowhead here would read as a causal or procedural direction."""
        ax.plot([x1, x2], [y1, y2], color=MUTED, linewidth=0.9,
                linestyle=(0, (3, 2.5)), zorder=1)

    box(0.5, 23, 19, 9, "Frozen QC predictor\n(label-free features)", edge=INK2)
    arrow(19.5, 27.5, 23.5, 27.5)
    # Widened from 14 to 17 units: at 7.2in across 100 units, "per-case estimate"
    # at 8pt is wider than the box was.
    box(23.5, 23, 17, 9, r"$\hat{q}$" "\nper-case estimate", edge=INK2)

    # Both paths are drawn in neutral ink. Blue and orange are reserved for
    # dataset identity in Figures 2 and 3 and must not also mean "estimand" here.
    arrow(40.5, 30, 45, 40)
    box(45, 36, 21, 9, "used unchanged", edge=INK2)
    arrow(66, 40.5, 71, 40.5)
    box(71, 35, 26, 11, "R5\ndeployment-state\nrelative utility", edge=INK2,
        weight="bold")

    arrow(40.5, 25, 45, 15)
    box(45, 10, 21, 9, "oracle recalibration\nfitted on target labels", edge=INK2)
    arrow(66, 14.5, 71, 14.5)
    box(71, 9, 26, 11, "R8\nrecoverable\nrelative utility", edge=INK2,
        weight="bold")

    # The comparator is drawn once, because both estimands use the same one.
    box(43.5, 23.5, 24, 8, "cross-fitted patient-agnostic\nmedian comparator",
        edge=INK2, fs=7.6)
    tie(67.5, 29.5, 84, 35)
    tie(67.5, 25.5, 84, 20)
    ax.text(84, 27.6, "measured against", ha="center", va="center", fontsize=7.0,
            color=MUTED, style="italic",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.4))

    ax.text(50, 50.5, "Both estimands share the predictor, the comparator and the "
                      "folds.\nThey differ in exactly one step.",
            ha="center", va="center", fontsize=8.2, color=INK2, linespacing=1.5)
    ax.text(50, 3.2, "Target labels are used for evaluation only. The oracle mapping "
                     "is a diagnostic device, not a deployable correction.",
            ha="center", va="center", fontsize=7.4, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    return _save(fig, "figure1_shiftqc_concept")

def figure2() -> Path:
    """Deployment-state and recoverable-utility transfer estimates."""
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.1), sharex=True,
                             gridspec_kw={"hspace": 0.48})

    panels = [
        (axes[0], "a   Deployment-state relative utility (R5, key secondary)",
         [("M&Ms", MNMS, "MM-R5-D", "MM-R5-LO", "MM-R5-HI", "loss resolved"),
          ("AbdomenCT", ABDO, "V2-R5-D", "V2-R5-LO", "V2-R5-HI", "loss not resolved")]),
        (axes[1], "b   Oracle-recalibrated recoverable utility (R8 primary, affine-L1)",
         [("M&Ms", MNMS, "MM-R8-Delta8", "MM-R8-CI_low", "MM-R8-CI_high", "UNRESOLVED"),
          ("AbdomenCT", ABDO, "V2-R8-Delta8", "V2-R8-CI_low", "V2-R8-CI_high",
           "UNRESOLVED")]),
    ]

    for ax, title, entries in panels:
        ax.set_title(title, loc="left", color=INK, pad=7)
        # Registration order, top to bottom. Never sorted by value.
        for index, (label, colour, pid, lo, hi, verdict) in enumerate(entries):
            y = len(entries) - 1 - index
            _interval(ax, y, val(lo), val(hi), val(pid), colour,
                      marker="o" if label == "M&Ms" else "s")
            # Labels live in the gutters outside the plotting box, in axes
            # coordinates, so no interval can ever grow into them.
            ax.text(-0.02, y, f"{label}\n{shown(pid)}  [{shown(lo)}, {shown(hi)}]",
                    transform=ax.get_yaxis_transform(), va="center", ha="right",
                    color=INK, fontsize=8.2, linespacing=1.65)
            ax.text(1.02, y, verdict, transform=ax.get_yaxis_transform(),
                    va="center", ha="left", color=INK2, fontsize=7.8,
                    style="italic")
        ax.axvline(0.0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.set_ylim(-0.62, len(entries) - 0.38)
        ax.set_yticks([])
        ax.set_xlim(-1.52, 0.42)
        ax.spines["left"].set_visible(False)

    axes[1].set_xlabel(r"$\Delta$ = out-of-domain $-$ in-domain relative utility"
                       "\n(negative = the registered direction)", color=INK)
    fig.text(0.008, 0.012,
             "Registered rule: POSITIVE only if the interval lies entirely below "
             "zero; otherwise UNRESOLVED.",
             fontsize=7.2, color=MUTED)
    fig.subplots_adjust(left=0.275, right=0.815, top=0.915, bottom=0.225)
    return _save(fig, "figure2_r5_r8_cross_validation")

def figure3() -> Path:
    """In-domain QC competence, the registered prerequisite."""
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    entries = [("M&Ms", MNMS, "MM-B-RHO", "MM-B-LO", "MM-B-HI", "MM-B-N", "o"),
               ("AbdomenCT", ABDO, "V2-B-RHO", "V2-B-LO", "V2-B-HI", "V2-B-N", "s")]
    for index, (label, colour, rid, lo, hi, nid, marker) in enumerate(entries):
        y = len(entries) - 1 - index
        _interval(ax, y, val(lo), val(hi), val(rid), colour, marker=marker)
        ax.text(-0.02, y, f"{label}\nn = {shown(nid)}",
                transform=ax.get_yaxis_transform(), va="center", ha="right",
                color=INK, fontsize=8.5, linespacing=1.5)
        ax.text(1.02, y, f"{shown(rid)}  [{shown(lo)}, {shown(hi)}]",
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                color=INK2, fontsize=7.8)
    ax.axvline(0.0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.03, 0.70)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Spearman rank association between predicted and observed "
                  "segmentation quality", color=INK)
    ax.set_title("Registered prerequisite: the lower interval bound lies above zero",
                 loc="left", color=INK, pad=7)
    fig.text(0.008, 0.02, "Both magnitudes are reported. They are not compared as "
                          "an endpoint.", fontsize=7.2, color=MUTED)
    fig.subplots_adjust(left=0.175, right=0.765, top=0.845, bottom=0.315)
    return _save(fig, "figure3_baseline_competence")

def figure4() -> Path:
    """Supporting M&Ms ensemble-instability association across the adjustment ladder."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    levels = ("S0", "S1", "S2", "S3")
    panels = [(axes[0], "ID", "a   In-domain population (diagnostic)"),
              (axes[1], "OOD", "b   Out-of-domain population (D-014)")]
    for ax, tag, title in panels:
        ax.set_title(title, loc="left", color=INK, pad=7, fontsize=9)
        for index, level in enumerate(levels):
            y = len(levels) - 1 - index
            _interval(ax, y, val(f"MM-D014-{tag}-{level}-CILOW"),
                      val(f"MM-D014-{tag}-{level}-CIHIGH"),
                      val(f"MM-D014-{tag}-{level}-RHO"), MNMS, marker="o")
        ax.axvline(0.0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xlim(-0.92, 0.06)
        ax.set_ylim(-0.6, len(levels) - 0.4)
        ax.set_xlabel("Residual rank association with macro Dice", color=INK)
    axes[0].set_yticks(range(len(levels)))
    axes[0].set_yticklabels(list(reversed(levels)), fontsize=8.2, color=INK2)
    fig.text(0.008, 0.015, "Association, not mechanism. Supporting evidence only.",
             fontsize=7.2, color=MUTED)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.855, bottom=0.30)
    # The adjustment sets are long; they belong in the caption rather than on the
    # axis, and are emitted here so the caption draws on the same ledger rows.
    _caption_fragments["figure4"] = [
        f"{lv}: {shown(f'MM-D014-OOD-{lv}-CONTROLS')}" for lv in levels]
    return _save(fig, "figure4_d014")

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate selected frozen figures.")
    parser.add_argument(
        "--only", nargs="+",
        choices=("figure1", "figure2", "figure3", "figure4"),
        default=("figure1",),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="optional output directory; defaults to the candidate figures directory",
    )
    parser.add_argument(
        "--ledger", type=Path, default=Path("local_inputs/frozen_result_ledger.json"),
        help="path to a locally regenerated result ledger",
    )
    args = parser.parse_args()
    global FIG, LEDGER
    FIG = args.output_dir if args.output_dir is not None else ROOT / "figures" / "figure1"
    ledger_path = args.ledger if args.ledger.is_absolute() else ROOT / args.ledger
    if any(name != "figure1" for name in args.only):
        try:
            if not ledger_path.is_file():
                parser.exit(2, "Required source-data-derived input not included in this repository. See docs/data_access.md and docs/full_reproduction.md.\n")
            rows = json.loads(ledger_path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows:
                raise ValueError("Expected nonempty locally regenerated result-row list")
            LEDGER = {}
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("result_id"), str):
                    raise ValueError("Invalid local result-row schema")
                if row["result_id"] in LEDGER:
                    raise ValueError("Duplicate local result ID")
                if not math.isfinite(float(row["full_precision"])):
                    raise ValueError("Nonfinite local result value")
                if not isinstance(row.get("display"), str):
                    raise ValueError("Invalid local display value")
                LEDGER[row["result_id"]] = row
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.exit(2, f"Invalid locally regenerated result input: {exc}. See docs/data_access.md and docs/full_reproduction.md.\n")
    else:
        LEDGER = {}

    if any(name != "figure1" for name in args.only) and not LEDGER:
        parser.exit(2, "Required source-data-derived input not included in this repository. See docs/data_access.md and docs/full_reproduction.md.\n")

    builders = {
        "figure1": figure1,
        "figure2": figure2,
        "figure3": figure3,
        "figure4": figure4,
    }
    for name in args.only:
        plt.rcParams.update(matplotlib.rcParamsDefault)
        if name == "figure1":
            plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 8.5, "text.color": INK,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
        else:
            plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.7,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
        plt.rcParams["svg.hashsalt"] = "SHIFT-QC-TASK045-v1"
        try:
            output = builders[name]()
        except (KeyError, ValueError, TypeError, RuntimeError) as exc:
            parser.exit(2, f"Invalid or missing locally regenerated result input: {exc}. See docs/data_access.md and docs/full_reproduction.md.\n")
        print(f"{name}: {output.name}")


if __name__ == "__main__":
    main()
