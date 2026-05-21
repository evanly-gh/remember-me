"""
plot_training_curve.py — parse a beauty-regressor training log into a
print-ready PNG of train/val loss + validation Pearson r over epochs.

Usage
-----
    python plot_training_curve.py logs/beauty-train-12345.out
        --out training_curve.png

Or pipe in via stdin:

    sbatch_output | python plot_training_curve.py - --out curve.png

Defaults:
- 1600×1000 px @ 300 DPI → ~5.3 × 3.3 inches at print, still crisp
  when scaled up to ~10" wide on a poster.
- Colour palette matches the project's chopped-score motif so the
  figure visually ties into the rest of the poster.

Output:
- One twin-axis figure: train + val MSE on the left axis, validation
  Pearson r on the right axis. Best-Pearson epoch annotated.
- Stats printed to stdout: total epochs, best Pearson + its epoch,
  final MAE, final val MSE.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# ── Project palette (matches the chopped-score tier colours) ──────
PALETTE = {
    "train_loss": "#D4A017",   # mustard ("Average Joe" — neutral)
    "val_loss":   "#E53935",   # red     ("Chopped" — what we want to push down)
    "pearson":    "#00C853",   # green   ("Gigachad" — what we want to push up)
    "best_marker":"#1A1A1A",   # ink
    "grid":       "#E0E0E0",
    "axis":       "#333333",
}

# Match lines like:
#   epoch 03  train_mse=0.4582  val_mse=0.4218  val_mae=0.5183  pearson=0.4521
# Tolerant of leading whitespace, extra fields, and either `pearson`
# or `val_pearson` / `val_r` naming.
EPOCH_RE = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s*/?\s*\d*\s+"
    r".*?"
    r"(?:train_mse|train_loss)\s*=\s*(?P<train_mse>[0-9.]+).*?"
    r"(?:val_mse|val_loss)\s*=\s*(?P<val_mse>[0-9.]+).*?"
    r"(?:val_mae|mae)\s*=\s*(?P<val_mae>[0-9.]+).*?"
    r"(?:pearson|val_pearson|val_r)\s*=\s*(?P<pearson>[0-9.]+)",
    re.IGNORECASE,
)


def parse_log(text: str):
    """Pull (epoch, train_mse, val_mse, val_mae, pearson) tuples from raw log text."""
    rows = []
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        rows.append({
            "epoch":     int(m.group("epoch")),
            "train_mse": float(m.group("train_mse")),
            "val_mse":   float(m.group("val_mse")),
            "val_mae":   float(m.group("val_mae")),
            "pearson":   float(m.group("pearson")),
        })
    # De-dupe in case the log emits epoch lines multiple times (e.g.
    # EMA + raw). Keep the last occurrence per epoch since that's
    # typically the EMA-eval line.
    by_epoch = {r["epoch"]: r for r in rows}
    return [by_epoch[e] for e in sorted(by_epoch)]


def render(rows, out_path: Path):
    if not rows:
        sys.exit("No epoch lines found. Check the log file format.")

    epochs   = [r["epoch"] for r in rows]
    train    = [r["train_mse"] for r in rows]
    val      = [r["val_mse"] for r in rows]
    pearson  = [r["pearson"] for r in rows]

    best_idx = max(range(len(rows)), key=lambda i: pearson[i])
    best_epoch = epochs[best_idx]
    best_pearson = pearson[best_idx]

    # Figure setup. constrained_layout removes the need to fiddle with
    # subplots_adjust when the right-side y-axis label runs long.
    fig, ax_loss = plt.subplots(
        figsize=(8, 5),
        dpi=200,
        constrained_layout=True,
    )

    # ── Loss curves (left axis) ──────────────────────────────────
    ax_loss.plot(
        epochs, train,
        color=PALETTE["train_loss"], linewidth=2.0,
        marker="o", markersize=4, label="Train MSE",
    )
    ax_loss.plot(
        epochs, val,
        color=PALETTE["val_loss"], linewidth=2.0,
        marker="s", markersize=4, label="Val MSE",
    )
    ax_loss.set_xlabel("Epoch", fontsize=12, color=PALETTE["axis"])
    ax_loss.set_ylabel("MSE loss (lower = better)", fontsize=12,
                       color=PALETTE["axis"])
    ax_loss.tick_params(axis="x", colors=PALETTE["axis"])
    ax_loss.tick_params(axis="y", colors=PALETTE["axis"])
    ax_loss.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.7)
    ax_loss.set_ylim(bottom=0)

    # ── Pearson r (right axis) ───────────────────────────────────
    ax_r = ax_loss.twinx()
    ax_r.plot(
        epochs, pearson,
        color=PALETTE["pearson"], linewidth=2.6,
        marker="D", markersize=4.5, label="Val Pearson r",
    )
    ax_r.set_ylabel("Validation Pearson r (higher = better)",
                    fontsize=12, color=PALETTE["axis"])
    ax_r.tick_params(axis="y", colors=PALETTE["axis"])
    ax_r.set_ylim(0, 1.0)

    # ── Best-epoch annotation ────────────────────────────────────
    ax_r.scatter(
        [best_epoch], [best_pearson],
        s=120, color=PALETTE["best_marker"], zorder=5,
        edgecolor="white", linewidth=1.5,
    )
    ax_r.annotate(
        f"best epoch {best_epoch}\nr = {best_pearson:.3f}",
        xy=(best_epoch, best_pearson),
        xytext=(10, -22),
        textcoords="offset points",
        fontsize=10,
        color=PALETTE["best_marker"],
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=PALETTE["pearson"],
            linewidth=1.2,
        ),
    )

    # ── Title + combined legend ──────────────────────────────────
    ax_loss.set_title(
        "Beauty regressor — SCUT-FBP5500",
        fontsize=14, fontweight="bold", color=PALETTE["axis"],
    )

    lines_loss, labels_loss = ax_loss.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax_loss.legend(
        lines_loss + lines_r, labels_loss + labels_r,
        loc="upper right", frameon=True, framealpha=0.95,
        edgecolor=PALETTE["grid"], fontsize=10,
    )

    # Clean up axis spines for a poster look.
    for ax in (ax_loss, ax_r):
        for side in ("top",):
            ax.spines[side].set_visible(False)
        for side in ("left", "right", "bottom"):
            ax.spines[side].set_color(PALETTE["axis"])
            ax.spines[side].set_linewidth(1.0)

    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Print summary stats for the brief / poster copy.
    final = rows[-1]
    print(f"\nSummary:")
    print(f"  Total epochs    : {len(rows)}")
    print(f"  Best Pearson r  : {best_pearson:.4f}  (epoch {best_epoch})")
    print(f"  Final val MSE   : {final['val_mse']:.4f}")
    print(f"  Final val MAE   : {final['val_mae']:.4f}")
    print(f"  Final Pearson r : {final['pearson']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot the beauty-regressor training curve from a log file."
    )
    parser.add_argument(
        "log",
        help="path to the training log file, or '-' to read from stdin",
    )
    parser.add_argument(
        "--out", default="training_curve.png",
        help="output PNG path (default: training_curve.png)",
    )
    args = parser.parse_args()

    if args.log == "-":
        text = sys.stdin.read()
    else:
        text = Path(args.log).read_text(encoding="utf-8", errors="replace")

    rows = parse_log(text)
    render(rows, Path(args.out))


if __name__ == "__main__":
    main()
