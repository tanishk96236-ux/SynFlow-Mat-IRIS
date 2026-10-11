"""The whole system against the A-Lab's own targets and recipes.

    python scripts/eval_alab_targets.py              # recipes + simulator
    python scripts/eval_alab_targets.py --no-simulator   # recipes only, fast

THE DATA
--------
Szymanski et al., Nature 624, 86-91 (2023), Supplementary Data,
"20230502 Synthesis Results with Recipes.csv": 57 targets, most of them
compounds predicted stable by Materials Project and Google DeepMind, each with
the robot's result and, where it made something, the recipe it used --
precursors with masses in mg, temperature, hours.

    Success 33   Success (offline) 2   Partial 7   Failed 15

The 15 failures carry no recipe: the robot tried several and none worked.

WHAT IS COMPARED, FIELD BY FIELD
--------------------------------
For the 42 targets with a recipe:
  precursors    ours against theirs: exact set, Jaccard, per metal
  temperature   |ours - theirs|, in C
  dwell         ours against their 4 h, as a factor
  grams         THEIR precursors through our balancer and weigh_out, against
                their masses -- the derived field, tested on numbers a robot
                actually weighed
  simulator     their recipe at their temperature should not be rejected
For all 57, our recipe through the simulator, split by the robot's result.

WHAT THIS DOES NOT MEASURE
--------------------------
The robot's recipe is one recipe that worked, chosen by its own models and
active learning -- not the best recipe. A different temperature from ours is
a disagreement, not necessarily our error. And the A-Lab's novelty claims
were corrected in January 2026 (Leeman et al.): several targets were known,
so each one is checked against our training corpus and the overlap reported.
"""

import argparse
import csv
import io
import json
import math
import os
import pathlib
import re
import sys
import zipfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ZIP = "data/alab/alab_supplementary_data.zip"
TABLE = "20230502 Synthesis Results with Recipes.csv"
CHECKPOINT = "runs/alab_targets.jsonl"
UBIQUITOUS = frozenset({"O", "C", "H", "N"})


def canonical(formula):
    from pymatgen.core import Composition
    try:
        return Composition(formula).reduced_formula
    except Exception:                                           # noqa: BLE001
        return None


def parse_targets(path=ZIP):
    """57 rows: target, result, mp id, precursors {formula: mg}, C, hours."""
    with zipfile.ZipFile(path) as archive:
        raw = archive.read(TABLE)
    # One hydrate is written with a middle dot stored as byte 0xE1 (code page
    # 437): "Y2(CO3)3<E1>3H2O". Rewritten as Y2(CO3)3(H2O)3, which pymatgen
    # reads, so the water is weighed and released like any other.
    text = raw.replace(b"\xe1", b"*").decode("utf-8", "replace")
    text = re.sub(r"(\S+)\*(\d+)H2O", r"\1(H2O)\2", text)
    rows = []
    for record in csv.DictReader(io.StringIO(text)):
        masses = {formula: float(mg) for mg, formula in re.findall(
            r"([\d.]+)\s*mg\s+(\S+)", record["Precursors + Masses"])}
        hours = re.findall(r"[\d.]+", record["Duration (hours)"])
        temperature = record["Temperature (C)"].strip()
        rows.append({
            "target": record["Target"].strip(),
            "result": record["Result"].strip(),
            "mp_id": record["Materials Project ID"].strip(),
            "masses_mg": masses,
            "temperature_C": float(temperature) if temperature else None,
            # "4 -> regrind -> 4" is two firings of 4 h each
            "hours_per_firing": float(hours[0]) if hours else None,
        })
    return rows


def metals(formula):
    from synflow.recipes import composition
    return frozenset(composition(formula)) - UBIQUITOUS


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def per_metal(ours, theirs, target):
    """For each metal of the target, did we pick the precursor they used?"""
    hits = total = 0
    for metal in metals(target):
        mine = {canonical(p) for p in ours if metal in metals(p)}
        used = {canonical(p) for p in theirs if metal in metals(p)}
        if used:
            total += 1
            hits += mine == used
    return hits, total


def gram_check(row):
    """Their precursors, our balancer: do our masses match what they weighed?

    Compared as mass FRACTIONS, so the batch size drops out. Returns the
    largest absolute difference in fraction, or why it could not be made.
    """
    from synflow import protocol, reaction

    theirs = row["masses_mg"]
    try:
        energy = reaction.reaction_enthalpy(list(theirs), row["target"])
    except reaction.NotAllMeasured:
        energy = None
    except Exception as error:                                  # noqa: BLE001
        return {"error": str(error)[:100]}
    equation = energy["reaction"] if energy else None
    if equation is None:
        # balance without energies: the masses need only the stoichiometry
        from pymatgen.analysis.reaction_calculator import Reaction
        from pymatgen.core import Composition
        gases = sorted(set(reaction.released_gases(list(theirs))) | {"O2"})
        try:
            target = Composition(row["target"]).reduced_composition
            balanced = Reaction([Composition(f) for f in theirs],
                                [target] + [Composition(g) for g in gases])
            balanced.normalize_to(target, 1.0)
            equation = str(balanced)
        except Exception as error:                              # noqa: BLE001
            return {"error": "could not balance: {}".format(str(error)[:80])}

    ours = protocol.weigh_out(equation, row["target"], batch_g=1.0)
    if not ours:
        return {"error": "weigh_out failed", "equation": equation}
    ours = {canonical(r["precursor"]): r["grams"] for r in ours}
    theirs_c = {canonical(f): mg for f, mg in theirs.items()}
    total_ours, total_theirs = sum(ours.values()), sum(theirs_c.values())
    fractions = {}
    for formula in theirs_c:
        fractions[formula] = (ours.get(formula, 0.0) / total_ours,
                              theirs_c[formula] / total_theirs)
    worst = max(abs(a - b) for a, b in fractions.values())
    return {"equation": equation, "fractions_ours_theirs": fractions,
            "worst_fraction_difference": worst}


def assess_brief(precursors, target, temperature):
    from synflow import reaction
    result = reaction.assess(precursors, target, temperature)
    air = result.get("in_air") or {}
    return {key: result.get(key) for key in (
        "reaction", "enthalpy_source", "enthalpy_eV", "free_energy_eV",
        "feasible", "onset_temperature_C", "ceiling_temperature_C",
        "driving_force_meV_per_atom", "kinetic_risk", "error")} | {
        "feasible_air": air.get("feasible"),
        "free_energy_air_eV": air.get("free_energy_eV")}


def _load(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    done[entry["target"]] = entry
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-simulator", action="store_true",
                        help="recipe fields only; no Materials Project calls")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    args = parser.parse_args()

    from synflow import recipes_llm
    from synflow.dao_s import load_shipped

    rows = parse_targets()
    print("{} A-Lab targets: {}".format(len(rows), ", ".join(
        "{} {}".format(sum(r["result"] == k for r in rows), k)
        for k in sorted({r["result"] for r in rows}))))

    # Leakage: a target already in the corpus the lookup was fitted on is not
    # unseen, and is reported separately rather than dropped.
    corpus = {canonical(r.target) for r in recipes_llm.load_recipes(
        verbose=False)}
    model = load_shipped(verbose=False)

    done = {} if args.no_simulator else _load(args.checkpoint)
    out = []
    for row in rows:
        if row["target"] in done:
            out.append(done[row["target"]])
            continue
        entry = dict(row, in_training_corpus=canonical(row["target"]) in corpus)
        recipe = model.predict_recipe(row["target"])
        entry["ours"] = recipe
        theirs = list(row["masses_mg"])
        if recipe and theirs:
            ours = recipe["precursors"]
            entry["exact"] = ({canonical(p) for p in ours}
                              == {canonical(p) for p in theirs})
            entry["jaccard"] = jaccard({canonical(p) for p in ours},
                                       {canonical(p) for p in theirs})
            entry["per_metal"] = per_metal(ours, theirs, row["target"])
            if row["temperature_C"] is not None:
                entry["temperature_error_C"] = abs(
                    recipe["firing_temperature_C"] - row["temperature_C"])
            if recipe.get("dwell_hours") and row["hours_per_firing"]:
                entry["dwell_factor"] = math.exp(abs(math.log(
                    recipe["dwell_hours"] / row["hours_per_firing"])))
            entry["grams"] = gram_check(row)
        if not args.no_simulator:
            if recipe:
                entry["simulator_ours"] = assess_brief(
                    recipe["precursors"], row["target"],
                    recipe["firing_temperature_C"])
            if theirs and row["temperature_C"] is not None:
                entry["simulator_theirs"] = assess_brief(
                    theirs, row["target"], row["temperature_C"])
            pathlib.Path(args.checkpoint).parent.mkdir(exist_ok=True)
            with open(args.checkpoint, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=str) + "\n")
        out.append(entry)
        print("  {:22s} {:9s} {}  ours {} at {} C".format(
            row["target"][:22], row["result"][:9],
            "TRAIN" if entry["in_training_corpus"] else "unseen",
            "+".join(recipe["precursors"]) if recipe else "REFUSED",
            recipe["firing_temperature_C"] if recipe else "-"), flush=True)

    summarise(out, args.no_simulator)
    path = "runs/alab_targets.json"
    pathlib.Path(path).write_text(json.dumps(out, indent=2, default=str),
                                  "utf-8")
    print("written to {}".format(path))


def summarise(out, no_simulator):
    def line(label, values, fmt):
        if values:
            print("   {:34s} {}   (n={})".format(label, fmt(values),
                                                 len(values)))

    def median(values):
        values = sorted(values)
        return values[len(values) // 2]

    for group, keep in (("ALL", lambda e: True),
                        ("UNSEEN ONLY", lambda e: not e["in_training_corpus"])):
        rows = [e for e in out if keep(e)]
        compared = [e for e in rows if "jaccard" in e]
        print("\n{} -- {} targets, {} with a recipe to compare".format(
            group, len(rows), len(compared)))
        line("precursor set exact", [e["exact"] for e in compared],
             lambda v: "{:.1%}".format(sum(v) / len(v)))
        line("precursor Jaccard (mean)", [e["jaccard"] for e in compared],
             lambda v: "{:.3f}".format(sum(v) / len(v)))
        hits = sum(e["per_metal"][0] for e in compared)
        total = sum(e["per_metal"][1] for e in compared)
        if total:
            print("   {:34s} {:.1%}   ({}/{} metals)".format(
                "precursor per metal", hits / total, hits, total))
        errors = [e["temperature_error_C"] for e in compared
                  if "temperature_error_C" in e]
        line("temperature |ours-theirs| MAE", errors,
             lambda v: "{:.0f} C, median {:.0f} C".format(
                 sum(v) / len(v), median(v)))
        factors = [e["dwell_factor"] for e in compared if "dwell_factor" in e]
        line("dwell within 2x of their 4 h", factors,
             lambda v: "{:.1%}".format(sum(f <= 2 for f in v) / len(v)))
        worst = [e["grams"]["worst_fraction_difference"] for e in compared
                 if "worst_fraction_difference" in e.get("grams", {})]
        line("grams: worst mass-fraction gap", worst,
             lambda v: "median {:.4f}, max {:.4f}".format(median(v), max(v)))
        if no_simulator:
            continue
        theirs = [e["simulator_theirs"] for e in compared
                  if "simulator_theirs" in e
                  and e["simulator_theirs"]["feasible"] is not None]
        line("their recipe not rejected (1 bar)",
             [s["feasible"] for s in theirs],
             lambda v: "{:.1%}".format(sum(v) / len(v)))
        line("their recipe not rejected (air)",
             [s["feasible_air"] for s in theirs],
             lambda v: "{:.1%}".format(sum(v) / len(v)))
        for result in ("Success", "Partial", "Failed"):
            ours = [e["simulator_ours"] for e in rows
                    if e["result"].startswith(result)
                    and e.get("simulator_ours", {}).get("feasible") is not None]
            line("our recipe feasible | {}".format(result),
                 [s["feasible"] for s in ours],
                 lambda v: "{:.1%}".format(sum(v) / len(v)))
            line("  ...and kinetic risk flagged",
                 [bool(s["kinetic_risk"]) for s in ours],
                 lambda v: "{:.1%}".format(sum(v) / len(v)))


if __name__ == "__main__":
    main()
