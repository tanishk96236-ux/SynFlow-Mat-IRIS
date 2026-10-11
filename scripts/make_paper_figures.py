"""Figures for the paper, from results already measured.

    python scripts/make_paper_figures.py

Every number plotted is read from runs/ or quoted from a measurement recorded
in docs/target_output.md; nothing is recomputed here. Static PNGs for print:
light surface, the reference categorical order (blue, orange, aqua, yellow),
values written on the marks because two of those hues sit below 3:1 contrast.
"""

import json
import os
import pathlib

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

OUT = pathlib.Path("paper/figures")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    print("wrote", OUT / name)


def bars(ax, labels, values, colors, fmt, title, xlabel, reference=None):
    y = range(len(labels))
    ax.barh(list(y), values, color=colors, height=0.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    span = max(values) * 1.18
    ax.set_xlim(0, span)
    for i, v in enumerate(values):
        ax.text(v + span * 0.01, i, fmt.format(v), va="center", color=INK,
                fontsize=9)
    if reference is not None:
        ax.axvline(reference, color=INK_2, linestyle="--", linewidth=1)


def fig_recipe():
    """The recipe against its baselines and the oracle ceilings."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    bars(axes[0, 0], ["plain metal rule", "ConditionalLookup (ours)"],
         [47.5, 55.7], [SERIES[1], SERIES[0]], "{:.1f}%",
         "Precursor set exactly right", "% of 981 unseen targets")
    bars(axes[0, 1], ["plain metal rule", "ConditionalLookup (ours)",
                      "oracle ceiling"],
         [65.6, 80.9, 92.9], [SERIES[1], SERIES[0], GRID], "{:.1f}%",
         "Precursor right per metal", "% of metals, validation split")
    bars(axes[1, 0], ["predict the mean", "metal rule", "TemperatureNet (ours)",
                      "k-nearest neighbours"],
         [279, 234, 194, 193], [SERIES[1], SERIES[1], SERIES[0], SERIES[2]],
         "{:.0f} C", "Firing temperature error (lower is better)",
         "mean absolute error, C, 981 unseen targets")
    bars(axes[1, 1], ["predict the median", "dwell network (ours)",
                      "oracle ceiling"],
         [45.7, 48.6, 80.7], [SERIES[1], SERIES[0], GRID], "{:.1f}%",
         "Hold time within a factor of 2", "% of 2,527 test recipes")
    fig.suptitle("DAO-S: what the recipe gets right, against what it is "
                 "compared with", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig_recipe_accuracy.png")


def fig_alab_gates():
    """Agreement with the A-Lab's robot on 81 checked pairwise reactions."""
    import matplotlib.pyplot as plt

    gates = ["thermodynamics,\nbefore the gas fix", "thermodynamics",
             "+ learned kinetic\nmodel", "+ Tammann's rule",
             "+ robots' own evidence\n(shipped)"]
    formed = [88.7, 95.8, 74.6, 93.0, 95.8]
    inert = [50.0, 50.0, 50.0, 80.0, 70.0]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = list(range(len(gates)))
    for i in y:
        ax.plot([inert[i], formed[i]], [i, i], color=GRID, linewidth=2,
                zorder=1)
    ax.scatter(formed, y, s=70, color=SERIES[0], zorder=3,
               label="reactions the robot made (n = 71)",
               edgecolors=SURFACE, linewidths=2)
    ax.scatter(inert, y, s=70, color=SERIES[1], zorder=3,
               label="pairs where nothing formed (n = 10)",
               edgecolors=SURFACE, linewidths=2)
    for i in y:
        ax.text(formed[i] + 1.2, i, "{:.1f}%".format(formed[i]), va="center",
                fontsize=9)
        ax.text(inert[i] - 1.2, i, "{:.0f}%".format(inert[i]), va="center",
                ha="right", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(gates)
    ax.invert_yaxis()
    ax.set_xlim(35, 103)
    ax.set_xlabel("agreement with the robot, %")
    ax.grid(axis="y", visible=False)
    ax.set_title("The feasibility gate against A-Lab Supplementary Table 2",
                 loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "fig_alab_gates.png")


def fig_selectivity():
    """Three attempts to tell the A-Lab's failures from its successes."""
    import matplotlib.pyplot as plt

    labels = ["C1, all rivals (v2)", "C1, pair rivals only (v3)",
              "remaining driving force (v3)"]
    auc = [0.486, 0.459, 0.497]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    bars(ax, labels, auc, [SERIES[0]] * 3, "AUC {:.3f}",
         "Thermodynamic selectivity does not separate the robot's failures",
         "AUC, 14 failed against 42 made targets (0.5 = no signal)",
         reference=0.5)
    ax.set_xlim(0, 1.0)
    fig.tight_layout()
    save(fig, "fig_selectivity_auc.png")


def fig_arrows3():
    """One target, many recipes: YBCO yield against temperature."""
    import matplotlib.pyplot as plt

    path = pathlib.Path("runs/arrows3_YBCO_Ba4Y2Cu6O13_decomposed.jsonl")
    if not path.exists():
        print("skip ARROWS3: no", path)
        return
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("dG_per_atom") is not None:
                rows[r["id"]] = r
    rows = list(rows.values())
    temps = sorted({r["temperature_C"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.2))
    import random
    rng = random.Random(0)
    for r in rows:
        carbonate = "BaCO3" in r["precursors"]
        ax.scatter(r["temperature_C"] + rng.uniform(-18, 18),
                   r["target_fraction"], s=26,
                   color=SERIES[1] if carbonate else SERIES[0],
                   edgecolors=SURFACE, linewidths=1, zorder=3)
    for t in temps:
        values = sorted(r["target_fraction"] for r in rows
                        if r["temperature_C"] == t)
        ax.hlines(values[len(values) // 2], t - 35, t + 35, color=INK,
                  linewidth=2, zorder=4)
    ax.scatter([], [], color=SERIES[0], s=26, label="recipes without BaCO3")
    ax.scatter([], [], color=SERIES[1], s=26, label="recipes with BaCO3")
    ax.plot([], [], color=INK, linewidth=2, label="median at each temperature")
    ax.set_xticks(temps)
    ax.set_xlabel("firing temperature, C (4 h)")
    ax.set_ylabel("YBa2Cu3O6.5 in the product, wt%")
    ax.set_title("ARROWS3: 149 robot runs, 47 recipes for one target",
                 loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_arrows3_ybco.png")


def fig_anchoring():
    """Anchored against calibrated energies for compounds MP does have."""
    import matplotlib.pyplot as plt

    path = pathlib.Path("runs/anchoring_validation.jsonl")
    if not path.exists():
        print("skip anchoring: no", path)
        return
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8")
            .splitlines() if l.strip()]
    rows = [r for r in rows if r.get("anchored_error") is not None
            and r.get("calibrated_error") is not None]
    rows.sort(key=lambda r: abs(r["calibrated_error"]))
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(rows) + 1.4))
    y = list(range(len(rows)))
    ax.scatter([1000 * abs(r["calibrated_error"]) for r in rows], y, s=34,
               color=SERIES[1], label="calibrated (before)", zorder=3,
               edgecolors=SURFACE, linewidths=1)
    ax.scatter([1000 * abs(r["anchored_error"]) for r in rows], y, s=34,
               color=SERIES[0], label="anchored (new)", zorder=3,
               edgecolors=SURFACE, linewidths=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r["formula"] for r in rows], fontsize=8)
    # log scale: one compound (BaGdCrFeO6) is ~3,000 meV/atom off under BOTH
    # methods, and a linear axis would flatten every other point onto zero
    ax.set_xscale("log")
    ax.set_xlabel("|energy error| against DFT, meV/atom, log scale "
                  "(each compound treated as unknown)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Putting a new structure on Materials Project's scale",
                 loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_anchoring.png")


def main():
    style()
    fig_recipe()
    fig_alab_gates()
    fig_selectivity()
    fig_arrows3()
    fig_anchoring()


if __name__ == "__main__":
    main()
