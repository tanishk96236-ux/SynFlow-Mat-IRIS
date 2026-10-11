"""The simulator against robotic experiments: A-Lab Supplementary Table 2.

    python scripts/eval_alab.py

THE DATA
--------
Szymanski et al., "An autonomous laboratory for the accelerated synthesis of
inorganic materials", Nature 624, 86-91 (2023), Supplementary Table 2: every
pairwise reaction the A-Lab's robots observed during active learning, read off
by XRD, with the temperature at which it had happened:

    CaCO3|TiO2     CaTiO3            <= 800 C
    TiO2|NiO       NiTiO3            800 < T <= 900 C
    TiO2|NiO       No rxn (inert)    <= 800 C

88 rows: 78 where a product formed and 10 where nothing did. These are not
text-mined from literature and were never seen by any part of this project,
and the simulator has no parameters to have been fitted to them anyway.

(The A-Lab paper's novelty claims were corrected in January 2026 after
Leeman et al. showed its "new" compounds were known. That concerned its
TARGETS. These rows are pairwise reactions between ordinary precursors,
identified by XRD as known phases, and are not what the correction was about.)

THE TWO TESTS
-------------
Where a product formed by T: the simulator should find the reaction downhill
at T, with a thermodynamic floor at or below T.

Where nothing formed by T, there is no product to test, so the question is
asked of every stable compound in the chemical system that contains a metal
from each reactant. If none is downhill -- or every downhill one is below the
50 meV/atom kinetic threshold -- the gate agrees with the robot. If a strongly
downhill product exists, the gate predicted a reaction that did not happen.

A gate that accepts everything would score perfectly on the first test and
fail the second. The second is the one that measures whether it discriminates.

TWO EDITS TO THE SOURCE, BOTH STATED
------------------------------------
* "NH4H4PO4" appears twice. No such compound exists (it would be NH8PO4); the
  same table uses NH4H2PO4 for the identical reaction with MoO2 and MoO3, so
  it is read as that.
* Rows listing two products ("Mo(PO3)3, Mo2P4O15") are tested on the first.
"""

import json
import os
import pathlib
import re
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

SI_TEXT = "data/alab/alab_SI.txt"
TYPOS = {"NH4H4PO4": "NH4H2PO4"}
UBIQUITOUS = frozenset({"O", "C", "H", "N"})


def parse_table2(path=SI_TEXT):
    """88 rows as dicts: reactants, products, lower and upper temperature."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    block = text[text.index("Supplementary Table 2 |"):
                 text.index("Supplementary Table 3 |")]
    row = re.compile(
        r"^\s*(\S+\|\S+)\s+(.+?)\s+((?:\d+\s*<\s*T\s*)?≤\s*\d+)\s*$",
        re.M)
    rows = []
    for pair, products, temperature in row.findall(block):
        reactants = [TYPOS.get(r, r) for r in pair.split("|")
                     if r.lower() != "air"]
        numbers = [int(n) for n in re.findall(r"\d+", temperature)]
        inert = products.startswith("No rxn")
        rows.append({
            "pair": pair,
            "reactants": reactants,
            "air": "air" in pair.split("|"),
            "products": [] if inert else [p.strip()
                                          for p in products.split(",")],
            "inert": inert,
            "lower_C": numbers[0] if len(numbers) == 2 else None,
            "upper_C": numbers[-1],
            "corrected": any(r in TYPOS for r in pair.split("|")),
        })
    return rows


def metals(formula):
    from synflow.recipes import composition
    return frozenset(composition(formula)) - UBIQUITOUS


def test_formed(row, enthalpies="measured"):
    """A product formed by T: is it downhill at T, floor at or below T?"""
    from synflow import reaction

    target = row["products"][0]
    result = reaction.assess(row["reactants"], target, row["upper_C"],
                             enthalpies=enthalpies)
    feasible = result.get("feasible")
    onset = result.get("onset_temperature_C")
    out = {"target": target, "feasible": feasible,
           "enthalpy_eV": result.get("enthalpy_eV"),
           "entropy_term_eV": result.get("entropy_term_eV"),
           "free_energy_eV": result.get("free_energy_eV"),
           "driving_force_meV_per_atom":
               result.get("driving_force_meV_per_atom"),
           "kinetic_risk": result.get("kinetic_risk"),
           "onset_C": onset,
           "ceiling_C": result.get("ceiling_temperature_C"),
           "gases": result.get("gases"),
           "enthalpy_sources": result.get("enthalpy_sources"),
           "error": result.get("error"),
           "reaction": result.get("reaction")}
    out["outcome"] = _formed_outcome(feasible, onset, row["upper_C"])
    out["enthalpy_source"] = result.get("enthalpy_source")
    out["not_measured"] = result.get("not_measured")

    # Where calorimetry covered every species, the all-MP answer for the same
    # row too: the only rows where the two sources can disagree, so one run
    # gives the comparison instead of two full runs.
    if (result.get("enthalpy_source") or "").startswith("measured"):
        mp = reaction.assess(row["reactants"], target, row["upper_C"],
                             enthalpies="mp")
        out["mp"] = {"enthalpy_eV": mp.get("enthalpy_eV"),
                     "free_energy_eV": mp.get("free_energy_eV"),
                     "feasible": mp.get("feasible"), "error": mp.get("error")}
        out["outcome_mp"] = _formed_outcome(
            mp.get("feasible"), mp.get("onset_temperature_C"), row["upper_C"])

    # the same row with CO2 and O2 at their partial pressures in air
    air = result.get("in_air")
    if air:
        out["air"] = {key: air.get(key) for key in (
            "feasible", "free_energy_eV", "onset_temperature_C",
            "ceiling_temperature_C", "driving_force_meV_per_atom",
            "kinetic_risk")}
        out["outcome_air"] = _formed_outcome(
            air["feasible"], air["onset_temperature_C"], row["upper_C"])
    else:
        out["outcome_air"] = out["outcome"]
    return out


def _formed_outcome(feasible, onset, upper_C):
    if feasible is None:
        return "unchecked"
    if not feasible:
        return "rejected"                    # robot made it; gate says no
    if onset is not None and onset != float("inf") and onset > upper_C:
        return "floor_too_high"              # downhill, but floor above T
    return "agrees"


def test_inert(row, enthalpies="measured"):
    """Nothing formed by T: does the gate find no strong driving force?"""
    from synflow import hull, reaction
    from synflow.recipes import composition

    elements = set()
    for formula in row["reactants"]:
        elements |= set(composition(formula))
    sides = [metals(f) for f in row["reactants"]]
    try:
        diagram, _ = hull.phase_diagram(sorted(elements))
    except Exception as error:                                  # noqa: BLE001
        return {"outcome": "unchecked", "error": str(error)[:120]}

    candidates = sorted({
        e.composition.reduced_formula for e in diagram.stable_entries
        if all(metals(e.composition.reduced_formula) & side
               for side in sides)})
    tried = []
    for product in candidates:
        result = reaction.assess(row["reactants"], product, row["upper_C"],
                                 enthalpies=enthalpies)
        air = result.get("in_air") or {}
        tried.append({"product": product,
                      "feasible": result.get("feasible"),
                      "reaction": result.get("reaction"),
                      "gases": result.get("gases"),
                      "free_energy_eV": result.get("free_energy_eV"),
                      "driving_force_meV_per_atom":
                          result.get("driving_force_meV_per_atom"),
                      "kinetic_risk": result.get("kinetic_risk"),
                      "feasible_air": air.get("feasible"),
                      "free_energy_air_eV": air.get("free_energy_eV"),
                      "kinetic_risk_air": air.get("kinetic_risk"),
                      "error": result.get("error")})

    return {"outcome": _inert_outcome(candidates, tried, ""),
            "outcome_air": _inert_outcome(candidates, tried, "_air"),
            "candidates": candidates, "tried": tried}


def _inert_outcome(candidates, tried, suffix):
    """suffix "" reads the 1 bar verdicts, "_air" the ones in air."""
    feasible = [t.get("feasible" + suffix) for t in tried]
    risk = [t.get("kinetic_risk" + suffix) for t in tried]
    strong = [f and not r for f, r in zip(feasible, risk)]
    weak = [f and r for f, r in zip(feasible, risk)]
    if not candidates:
        return "agrees_nothing_to_form"
    if any(strong):
        return "predicted_reaction"          # gate expected a product
    if any(weak):
        return "agrees_kinetic_risk"
    if all(f is None for f in feasible):
        return "unchecked"
    return "agrees_uphill"


# Each row is appended here the moment it finishes. The first full run was
# cut off by a power failure at 55 of 88 rows, an hour in, and lost all of
# them: the results were only written at the very end.
#
# One file per enthalpy source. The run before the gas fix is kept as
# runs/alab_table2_before_gas_fix.jsonl: its verdicts used gases counted
# from the precursors, and resuming from it would mix two simulators.
CHECKPOINT = "runs/alab_table2_{}.jsonl"


def _load_checkpoint(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    done[entry["key"]] = entry
    return done


def _append(entry, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, default=str) + "\n")


def seed_from_log(log_path, rows, path):
    """Recover the rows a killed run had already finished, from its log.

    Only rows whose outcome was "agrees" are recovered. Their outcome is all
    the summary needs, and the log states it exactly. The rejections and the
    unchecked rows are left to be re-run, because explaining WHY the gate
    disagreed with the robot needs the free energy and the balanced
    equation, which the log line does not carry. Recovered entries say so.
    """
    formed = [r for r in rows if not r["inert"]]
    line = re.compile(r"^\s+formed\s+(\d+)/\d+\s+(.{22})\s+->\s+(\S+)\s+"
                      r"(\d+)\s+C\s+(\S+)")
    done = _load_checkpoint(path)
    seeded = 0
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for text in handle:
            match = line.match(text)
            if not match:
                continue
            position, pair, _, _, outcome = match.groups()
            index = int(position)
            key = "formed:{}".format(index)
            row = formed[index - 1]
            if outcome != "agrees" or key in done:
                continue
            # the log printed the pair padded to 22 characters
            if pair != row["pair"][:22].ljust(22):
                continue                  # the log is not this table
            _append({"key": key, "outcome": "agrees",
                     "target": row["products"][0],
                     "recovered_from_log": True}, path)
            seeded += 1
    return seeded


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--enthalpies", choices=("measured", "mp"),
                        default="measured",
                        help="measured where it exists (default), or all-MP")
    parser.add_argument("--seed-from-log", default=None,
                        help="recover 'agrees' rows from a killed run's log")
    args = parser.parse_args()
    checkpoint = CHECKPOINT.format(args.enthalpies)

    rows = parse_table2()
    formed = [r for r in rows if not r["inert"]]
    inert = [r for r in rows if r["inert"]]
    print("A-Lab Supplementary Table 2: {} rows, {} formed, {} inert".format(
        len(rows), len(formed), len(inert)))
    print("corrected for the NH4H4PO4 typo: {} rows".format(
        sum(r["corrected"] for r in rows)))
    print("enthalpies: {}   checkpoint: {}".format(args.enthalpies,
                                                   checkpoint))

    if args.seed_from_log:
        print("recovered {} rows from {}".format(
            seed_from_log(args.seed_from_log, rows, checkpoint),
            args.seed_from_log))
    done = _load_checkpoint(checkpoint)
    print("already finished: {} rows -- skipped\n".format(len(done)))

    for i, row in enumerate(formed, 1):
        key = "formed:{}".format(i)
        if key in done:
            row["result"] = done[key]
            continue
        row["result"] = test_formed(row, args.enthalpies)
        _append(dict(row["result"], key=key), checkpoint)
        res = row["result"]
        print("  formed {:2d}/{}  {:22s} -> {:12s} {:>4d} C  {:15s}{}".format(
            i, len(formed), row["pair"][:22], res["target"][:12],
            row["upper_C"], res["outcome"],
            "  (flag)" if res.get("kinetic_risk") else ""), flush=True)

    print()
    for i, row in enumerate(inert, 1):
        key = "inert:{}".format(i)
        if key in done:
            row["result"] = done[key]
            continue
        row["result"] = test_inert(row, args.enthalpies)
        _append(dict(row["result"], key=key), checkpoint)
        res = row["result"]
        print("  inert  {:2d}/{}  {:22s} {:>4d} C  {:24s} {}".format(
            i, len(inert), row["pair"][:22], row["upper_C"], res["outcome"],
            ", ".join(res.get("candidates", []))[:40]), flush=True)

    def tally(group, field):
        out = {}
        for row in group:
            key = row["result"].get(field, row["result"]["outcome"])
            out[key] = out.get(key, 0) + 1
        return out

    totals = {}
    for field, label in (("outcome", "gases at 1 bar"),
                         ("outcome_air", "in air: CO2 4.2e-4, O2 0.21 bar")):
        tf, ti = tally(formed, field), tally(inert, field)
        totals[field] = {"formed": tf, "inert": ti}
        checked_f = sum(v for k, v in tf.items() if k != "unchecked")
        checked_i = sum(v for k, v in ti.items() if k != "unchecked")
        agree_f = tf.get("agrees", 0)
        agree_i = sum(v for k, v in ti.items() if k.startswith("agrees"))

        print("\n" + "=" * 66)
        print("{} -- enthalpies {}".format(label.upper(), args.enthalpies))
        print("FORMED ({} rows) -- should be downhill with the floor at or "
              "below T".format(len(formed)))
        for key, value in sorted(tf.items(), key=lambda kv: -kv[1]):
            print("   {:22s} {:3d}".format(key, value))
        if checked_f:
            print("   agreement on checked rows   {}/{} = {:.1%}".format(
                agree_f, checked_f, agree_f / checked_f))
        print("\nINERT ({} rows) -- should find no strong driving force"
              .format(len(inert)))
        for key, value in sorted(ti.items(), key=lambda kv: -kv[1]):
            print("   {:24s} {:3d}".format(key, value))
        if checked_i:
            print("   agreement on checked rows   {}/{} = {:.1%}".format(
                agree_i, checked_i, agree_i / checked_i))
    print("=" * 66)

    pathlib.Path("runs").mkdir(exist_ok=True)
    summary = "runs/alab_table2_{}.json".format(args.enthalpies)
    pathlib.Path(summary).write_text(json.dumps(
        {"enthalpies": args.enthalpies, "rows": rows, "totals": totals},
        indent=2, default=str), "utf-8")
    print("written to {}".format(summary))


if __name__ == "__main__":
    main()
