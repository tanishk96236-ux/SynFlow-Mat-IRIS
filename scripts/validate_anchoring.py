"""Is anchoring a novel structure better than calibrating it? Measured.

    python scripts/validate_anchoring.py --limit 25

THE TEST
--------
Take compounds Materials Project DOES have -- the A-Lab's targets, each a
real phase with a DFT energy -- and pretend each is new: remove every entry
of its formula from its chemical system, relax its MP structure with the
potential as a generated CIF would be, and estimate its energy two ways:

    calibrated   hull's per-system correction, as _novel_entry does today
    anchored     MP's energy of the phases it decomposes into, plus the
                 potential's gap above them (hull.anchor)

Both are compared with the DFT energy MP actually has. The one with the
smaller error is the one a novel target should use, and its error is the
band its verdicts should carry. Nothing is fitted to these compounds; the
calibration and the anchor see only the OTHER phases of each system.
"""

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

CHECKPOINT = "runs/anchoring_validation.jsonl"


def one(formula, critic, max_atoms=None):
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition

    from synflow import dao_g, hull

    reduced = Composition(formula).reduced_formula
    elements = sorted(str(e) for e in Composition(formula))
    diagram, entries = hull.phase_diagram(elements)
    own = [e for e in entries if e.composition.reduced_formula == reduced]
    if not own:
        return {"error": "not in Materials Project"}
    truth = min(own, key=lambda e: e.energy_per_atom)
    if getattr(truth, "structure", None) is None \
            or not truth.structure.is_ordered:
        return {"error": "no ordered structure"}
    if max_atoms and len(truth.structure) > max_atoms:
        return {"error": "skipped: {} atoms".format(len(truth.structure))}
    others = tuple(e for e in entries
                   if e.composition.reduced_formula != reduced)
    loo = PhaseDiagram(list(others))
    structure = dao_g.relax(truth.structure, critic, steps=80)

    out = {"truth_eV_per_atom": float(truth.energy_per_atom),
           "e_above_hull_mp": float(diagram.get_e_above_hull(truth))}

    anchored = hull.anchor(structure, critic, loo)
    if anchored is not None:
        out["anchored_error"] = (anchored["energy_per_atom"]
                                 - truth.energy_per_atom)
        out["anchors"] = anchored["anchors"]
    else:
        out["anchored_error"] = None

    mp_refs, orb_refs = hull._references(loo, others, critic)
    if all(el in orb_refs for el in elements):
        correction = hull.calibrate(loo, others, critic, orb_refs, sample=30)
        fitted = correction["calibration_per_element"]
        fractions = structure.composition.fractional_composition
        bias = sum(fractions.get(el, 0.0) * c for el, c in fitted.items())
        formation = hull.formation_energy(structure, critic, orb_refs) - bias
        on_mp = formation + sum(fractions.get(el, 0.0) * mp_refs[el]
                                for el in elements)
        out["calibrated_error"] = on_mp - truth.energy_per_atom
        out["calibration_error_reported"] = correction["calibration_error"]
    else:
        out["calibrated_error"] = None
    return out


def summarise(rows):
    import statistics

    print("\n{:24s} {:>10s} {:>10s}".format("", "anchored", "calibrated"))
    both = [r for r in rows if r.get("anchored_error") is not None
            and r.get("calibrated_error") is not None]
    for label, key in (("mean |error| eV/atom", "abs"),
                       ("median |error|", "median"),
                       ("mean error (bias)", "mean")):
        cells = []
        for field in ("anchored_error", "calibrated_error"):
            values = [r[field] for r in both]
            if not values:
                cells.append("-")
                continue
            if key == "abs":
                v = statistics.mean(abs(x) for x in values)
            elif key == "median":
                v = statistics.median(abs(x) for x in values)
            else:
                v = statistics.mean(values)
            cells.append("{:+.4f}".format(v) if key == "mean"
                         else "{:.4f}".format(v))
        print("{:24s} {:>10s} {:>10s}".format(label, *cells))
    print("compounds compared on both: {}   anchored unavailable: {}   "
          "skipped: {}".format(
              len(both),
              sum(1 for r in rows if r.get("anchored_error") is None
                  and not r.get("error")),
              sum(1 for r in rows if r.get("error"))))
    if both:
        wins = sum(abs(r["anchored_error"]) < abs(r["calibrated_error"])
                   for r in both)
        print("anchored closer to DFT on {}/{}".format(wins, len(both)))
        stated = [r["calibration_error_reported"] for r in both
                  if r.get("calibration_error_reported") is not None]
        if stated:
            print("calibration's STATED error, mean {:.4f} eV/atom -- the "
                  "band verdicts use".format(statistics.mean(stated)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-atoms", type=int, default=None,
                        help="skip cells larger than this, for speed")
    args = parser.parse_args()

    from synflow import dao_p

    with open("runs/alab_targets.json", encoding="utf-8") as handle:
        targets = [t["target"] for t in json.load(handle)]
    done = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    done[entry["formula"]] = entry
    critic = dao_p.load()
    rows = []
    for formula in targets[:args.limit]:
        if formula in done:
            rows.append(done[formula])
            continue
        try:
            entry = dict(one(formula, critic, args.max_atoms),
                         formula=formula)
        except Exception as error:                              # noqa: BLE001
            entry = {"formula": formula,
                     "error": "{}: {}".format(type(error).__name__,
                                              str(error)[:100])}
        with open(CHECKPOINT, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
        rows.append(entry)
        print("  {:24s} anchored {}   calibrated {}   {}".format(
            formula[:24],
            "   -   " if entry.get("anchored_error") is None
            else "{:+.3f}".format(entry["anchored_error"]),
            "   -   " if entry.get("calibrated_error") is None
            else "{:+.3f}".format(entry["calibrated_error"]),
            entry.get("error", "")), flush=True)
    summarise(rows)


if __name__ == "__main__":
    main()
