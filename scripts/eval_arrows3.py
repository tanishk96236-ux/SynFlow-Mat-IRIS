"""Does the simulator rank RECIPES the way the robots' results do?

    python scripts/eval_arrows3.py                 # YBCO, 149 runs
    python scripts/eval_arrows3.py --dataset NTMO

THE DATA
--------
Szymanski, Nevatia, Bartel, Zeng & Ceder, "Autonomous and dynamic precursor
selection for solid-state materials synthesis", Nat. Commun. 14, 6956 (2023),
data from github.com/njszym/ARROWS (Examples/*/Exp.json; the repository states
no licence, so the files stay in data/ and are cited, not redistributed).

One target, many recipes: 47 precursor sets for YBa2Cu3O6.5, each heated at
600-900 C (some at 1000 C) for 4 h, and the products of each run measured by
XRD with weight fractions. Only runs marked "Experimentally Verified" are used
-- 149 for YBCO. The label is the target's weight fraction, 0-100%.

WHY THIS TEST IS DIFFERENT
--------------------------
Everything before asked "can this TARGET be made". This asks which RECIPE
works for one target -- the question a chemist actually faces, and the one
the paper answers by experiment: BaCO3 sets never gave pure YBCO, BaO and
BaO2 sets often did.

THE METRICS, AND WHAT EACH SHOULD DO IF IT WORKS
------------------------------------------------
    dG per atom          more negative -> more YBCO
    driving force        larger -> more YBCO (the 50 meV/atom flag)
    C1                   lower -> more YBCO (a rival less favoured)
    remaining dG         more negative -> more YBCO (energy left after the
                         rival has formed)
Stated before the run. Scored as Spearman rank correlation with the YBCO
fraction, over every run each metric could score.

PRECURSORS MATERIALS PROJECT DOES NOT HAVE
------------------------------------------
Y2(CO3)3 and CuCO3 are not in Materials Project, and many sets use them.
With --decompose-missing they are replaced by the oxide they decompose to
(Y2O3, CuO) -- both decompose far below 600 C, the lowest temperature here.
Off by default; the count of runs it rescues is reported either way.
"""

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DATA = "data/arrows3/{}_Exp.json"
TARGETS = {"YBCO": "YBa2Cu3O7", "NTMO": "Na2Te3Mo3O16"}
# the phase label the dataset uses for the target, as written in "products"
TARGET_LABELS = {"YBCO": "Ba2YCu3O7", "NTMO": "Na2Te3Mo3O16"}
DECOMPOSES_TO = {"Y2(CO3)3": "Y2O3", "Y2C3O9": "Y2O3", "CuCO3": "CuO"}


def load_runs(name):
    with open(DATA.format(name), encoding="utf-8") as handle:
        sets = json.load(handle)["Universal File"]
    runs = []
    for key, entry in sets.items():
        precursors = [p.strip() for p in key.split(",")]
        for temperature, record in entry["Temperatures"].items():
            if not record.get("Experimentally Verified"):
                continue
            fraction = 0.0
            for phase, weight in zip(record.get("products") or [],
                                     record.get("product weight fractions")
                                     or []):
                if phase.split("_")[0].replace(".cif", "") \
                        == TARGET_LABELS[name]:
                    fraction += float(weight)
            runs.append({"id": "{}|{}".format(key, temperature),
                         "precursors": precursors,
                         "temperature_C": float(temperature.split()[0]),
                         "target_fraction": fraction})
    return runs


def score_run(run, target, decompose):
    from synflow import kinetics, reaction, selectivity

    precursors = list(run["precursors"])
    substituted = []
    if decompose:
        for i, p in enumerate(precursors):
            if p in DECOMPOSES_TO:
                substituted.append(p)
                precursors[i] = DECOMPOSES_TO[p]
    out = {"substituted": substituted}
    result = reaction.assess(precursors, target, run["temperature_C"])
    force = result.get("driving_force_meV_per_atom")       # -1000 dG / atom
    out.update(dG_per_atom=None if force is None else -force / 1000.0,
               driving_force=force,
               verdict=result.get("verdict_class"),
               error=result.get("error"))
    try:
        sel = selectivity.primary_competition(precursors, target,
                                              run["temperature_C"])
        out["C1"] = sel.get("C1_eV_per_atom")
        out["rival"] = (sel.get("strongest_competitor") or {}).get("product")
        out["remaining"] = (sel.get("remaining") or {}).get("dG_eV_per_atom")
    except Exception as error:                                  # noqa: BLE001
        out["C1"] = out["remaining"] = None
        out["selectivity_error"] = str(error)[:100]
    report = kinetics.check_recipe(precursors, run["temperature_C"], 4)
    out["blocked_pairs"] = report["blocked_pairs"]
    return out


def spearman(x, y):
    """Rank correlation, ties given their average rank."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="YBCO", choices=sorted(TARGETS))
    parser.add_argument("--decompose-missing", action="store_true")
    parser.add_argument("--target", default=None,
                        help="formula to judge; YBCO's default is O7")
    args = parser.parse_args()

    # THE OXYGEN CONTENT MATTERS. The paper's target is YBa2Cu3O6.5, and at
    # 800-900 C in air YBCO is oxygen-poor. Judged as YBa2Cu3O7 the reaction
    # must take up oxygen, which costs entropy at high temperature: five
    # recipes that gave over 50% YBCO came out uphill. Materials Project
    # holds O6.5 as Ba4Y2Cu6O13 (mp-20897, 22 meV/atom above the hull).
    target = args.target or TARGETS[args.dataset]
    # A fractional formula never matches: pymatgen keeps "YBa2Cu3O6.5" as
    # Ba2Y1Cu3O6.5 while Materials Project files the phase as Ba4Y2Cu6O13,
    # and the first --target run scored 0 of 149. Scale to whole numbers.
    from pymatgen.core import Composition
    composition = Composition(target)
    for factor in range(1, 13):
        if all(abs(v * factor - round(v * factor)) < 1e-6
               for v in composition.values()):
            target = (composition * factor).reduced_formula
            break
    runs = load_runs(args.dataset)
    print("{}: {} verified runs, target {}".format(args.dataset, len(runs),
                                                   target))
    # Keyed by the target ACTUALLY judged, after scaling, so "YBa2Cu3O6.5"
    # and "Ba4Y2Cu6O13" share one file and an earlier broken spelling's
    # file is never picked up.
    path = "runs/arrows3_{}_{}{}.jsonl".format(
        args.dataset, target,
        "_decomposed" if args.decompose_missing else "")
    # A row that came back unchecked is scored again rather than resumed.
    # The Colab rerun at O6.5 printed 149 of 149 unchecked twice: the first
    # attempt had failed on the unscaled formula, and resuming kept every
    # one of those failures after the formula was fixed.
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("dG_per_atom") is not None:
                        done[entry["id"]] = entry
    rows = []
    for run in runs:
        if run["id"] in done:
            rows.append(done[run["id"]])
            continue
        entry = dict(run, **score_run(run, target, args.decompose_missing))
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
        rows.append(entry)
        print("  {:45s} {:>5.0f} C  target {:5.1f}%  dG/atom {}  C1 {}"
              .format(run["id"].split("|")[0][:45], run["temperature_C"],
                      run["target_fraction"],
                      "  -   " if entry["dG_per_atom"] is None
                      else "{:+.3f}".format(entry["dG_per_atom"]),
                      "  -   " if entry.get("C1") is None
                      else "{:+.3f}".format(entry["C1"])), flush=True)

    # Within one temperature: across all temperatures the YBCO fraction is
    # driven by heat (rho +0.79 for temperature alone), so a pooled
    # correlation mostly asks whether a metric tracks temperature. The
    # question a chemist has -- which recipe, at this temperature -- is this.
    print("\nWITHIN EACH TEMPERATURE (which recipe works better at the same T)")
    for temperature in sorted({r["temperature_C"] for r in rows}):
        group = [r for r in rows if r["temperature_C"] == temperature]
        cells = []
        for key in ("dG_per_atom", "C1", "remaining"):
            pairs = [(r[key], r["target_fraction"]) for r in group
                     if r.get(key) is not None]
            cells.append("{} {:+.3f} (n={})".format(
                key, spearman([p[0] for p in pairs], [p[1] for p in pairs]),
                len(pairs)) if len(pairs) >= 5 else "{} -".format(key))
        print("   {:5.0f} C   {}".format(temperature, "   ".join(cells)))

    print("\nRANK CORRELATION WITH THE TARGET'S WEIGHT FRACTION, POOLED")
    print("(sign expected if the metric works: dG -, force +, C1 -, "
          "remaining -)")
    for key, label, expect in (("dG_per_atom", "dG per atom", "-"),
                               ("driving_force", "driving force", "+"),
                               ("C1", "C1", "-"),
                               ("remaining", "remaining dG", "-"),
                               ("temperature_C", "temperature alone", "+")):
        pairs = [(r[key], r["target_fraction"]) for r in rows
                 if r.get(key) is not None]
        if len(pairs) < 5:
            print("   {:18s} too few scored ({})".format(label, len(pairs)))
            continue
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
        print("   {:18s} rho {:+.3f}   n={:3d}   expected {}".format(
            label, rho, len(pairs), expect))
    unchecked = sum(1 for r in rows if r.get("dG_per_atom") is None)
    print("   unchecked by the simulator: {} of {}".format(unchecked,
                                                         len(rows)))
    blocked = [r for r in rows if r.get("blocked_pairs")]
    if blocked:
        mean_b = sum(r["target_fraction"] for r in blocked) / len(blocked)
        rest = [r for r in rows if not r.get("blocked_pairs")]
        mean_r = sum(r["target_fraction"] for r in rest) / max(len(rest), 1)
        print("   runs with a pair the Precursor Genome saw unreacted: {} "
              "(mean target {:.1f}%, others {:.1f}%)".format(
                  len(blocked), mean_b, mean_r))


if __name__ == "__main__":
    main()
