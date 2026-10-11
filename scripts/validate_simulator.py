"""How often is the simulator right? Measured against real recipes.

    python scripts/validate_simulator.py --sample 150

WHY THIS SCRIPT IS THE ONE THAT MATTERS
---------------------------------------
reaction.py passes three controls -- BaTiO3, LiFePO4, LiCoO2 -- and three
examples are not a measurement. The first question anyone will ask is "how
often is it right", and until this runs there is no answer to give.

There is a ready-made positive set sitting in the repository: 19,488 recipes
text-mined from the literature. Every one of them was actually carried out by
a chemist who then published it, so every one must come out feasible. A real
recipe the simulator rejects is a false negative, and counting them is the
accuracy number.

AND WHY THE NEGATIVE SET IS THE PART THAT DECIDES IT
----------------------------------------------------
A gate that says yes to everything scores 100% on the positive set and is
worthless. So the harder test is a set of recipes that should FAIL.

These are built by scrambling: keep a real target, and give it the precursors
of a different, unrelated target. Ba2TiO4 made from CuO and Nd2O3 is not a
synthesis, and a useful gate must say so.

Scrambling is honest but imperfect, and the imperfection runs one way. A
scrambled recipe may occasionally be accidentally reasonable -- swap Li2CO3
for Na2CO3 and the thermodynamics barely notices -- so some of what looks like
a false positive is really a mislabelled negative. That makes the measured
false-positive rate an OVERestimate of the simulator's error, which is the
safe direction. It is stated rather than hidden.

THE SCRAMBLED SET DOES NOT TEST THE GATE (found 2026-10-08)
-----------------------------------------------------------
Every one of 40 scrambled recipes came back "could not balance". It has to:
the donor is required to share no metal with the target, so the target's
metals have no source and no equation exists. The set measures mass
balance, which needs no simulator, and its "rejections" say nothing about
thermodynamics. Negatives that do test the gate come from experiments where
nothing formed -- A-Lab Table 2's inert rows and the Precursor Genome's
unreacted pairs (scripts/eval_alab.py, scripts/train_kinetics.py). This
script's true-positive rate stands; its negative set is reported as what it
is.

WHAT THE NUMBERS MEAN
---------------------
    true positive rate   real recipes accepted        want high
    false positive rate  scrambled recipes accepted   want low

If both are near 100% the simulator is not discriminating, it is just
agreeing. The gap between them is the whole value of the thing.

Needs no GPU and no potential: every compound in a real recipe is a real
compound, so Materials Project has all of them and the energies come straight
from the database. That is why this can run on a laptop.
"""

import argparse
import json
import os
import pathlib
import random
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def scramble(recipes, rng):
    """Real targets, wrong precursors -- the set that should be rejected.

    The donor is required to share NO element with the target apart from the
    ubiquitous ones. Without that check the scramble quietly produces valid
    chemistry: drawing a donor at random from an oxide corpus lands on another
    oxide, and "BaTiO3 from BaCO3 and TiO2" can be reconstructed by accident.
    """
    ubiquitous = {"O", "C", "H", "N"}
    from synflow.recipes import composition

    def metals(formula):
        return frozenset(composition(formula)) - ubiquitous

    out = []
    for recipe in recipes:
        target_metals = metals(recipe.target)
        if not target_metals:
            continue
        for _ in range(20):                    # a few tries, then give up
            donor = rng.choice(recipes)
            donor_metals = set()
            for precursor in donor.precursors:
                donor_metals |= metals(precursor)
            if donor_metals and not (donor_metals & target_metals):
                out.append((recipe.target, sorted(donor.precursors),
                            recipe.max_temperature or 800.0))
                break
    return out


def run(cases, label, checkpoint=None):
    """Assess a list of (target, precursors, temperature) and tally.

    Each row is appended to `checkpoint` as it finishes and skipped on a
    rerun: on this machine a large chemical system takes minutes, and a run
    of a few hours is cut off by the two-hour limit on background jobs.
    """
    from synflow import reaction

    done = {}
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    done[(entry["set"], entry["target"],
                          tuple(entry["precursors"]))] = entry

    rows, accepted, rejected, unchecked = [], 0, 0, 0
    for i, (target, precursors, temperature) in enumerate(cases, 1):
        key = (label, target, tuple(precursors))
        if key in done:
            result = {"feasible": done[key]["feasible"],
                      "free_energy_eV": done[key]["free_energy_eV"],
                      "enthalpy_eV": done[key]["enthalpy_eV"],
                      "error": done[key]["error"]}
        else:
            result = reaction.assess(precursors, target, temperature)
        verdict = result.get("feasible")
        if verdict is None:
            unchecked += 1
        elif verdict:
            accepted += 1
        else:
            rejected += 1
        rows.append({"set": label, "target": target,
                     "precursors": precursors,
                     "temperature_C": temperature,
                     "feasible": verdict,
                     "free_energy_eV": result.get("free_energy_eV"),
                     "enthalpy_eV": result.get("enthalpy_eV"),
                     "error": result.get("error")})
        if checkpoint and key not in done:
            with open(checkpoint, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(rows[-1], default=str) + "\n")
        if i % 10 == 0:
            print("   {:4d}/{:<4d} accepted {:3d}  rejected {:3d}  "
                  "unchecked {:3d}".format(i, len(cases), accepted,
                                           rejected, unchecked),
                  flush=True)
    return rows, accepted, rejected, unchecked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=100,
                        help="recipes per set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/simulator_validation.json")
    args = parser.parse_args()

    from synflow.recipes import load_recipes, split_by_target

    rng = random.Random(args.seed)

    recipes = load_recipes(verbose=False)
    # The TEST split, so this is measured on recipes no part of this project
    # was tuned on. DAO-S was fitted on train; the simulator was fitted on
    # nothing, but using the same held-out split keeps one story.
    held_out = split_by_target(recipes, seed=0)["test"]
    usable = [r for r in held_out if r.precursors and r.target]
    print("{} recipes in the test split, {} usable".format(
        len(held_out), len(usable)))

    positives = [(r.target, sorted(r.precursors), r.max_temperature or 800.0)
                 for r in rng.sample(usable, min(args.sample, len(usable)))]
    negatives = scramble(rng.sample(usable, min(args.sample, len(usable))),
                         rng)

    checkpoint = os.path.splitext(args.out)[0] + ".jsonl"
    print("\nPOSITIVE set -- real recipes, should be accepted ({})".format(
        len(positives)))
    pos_rows, pos_ok, pos_no, pos_un = run(positives, "real", checkpoint)

    print("\nNEGATIVE set -- scrambled recipes, should be rejected ({})".format(
        len(negatives)))
    neg_rows, neg_ok, neg_no, neg_un = run(negatives, "scrambled",
                                           checkpoint)

    pos_checked = pos_ok + pos_no
    neg_checked = neg_ok + neg_no

    print("\n" + "=" * 62)
    print("{:28s} {:>10s} {:>10s} {:>10s}".format(
        "", "accepted", "rejected", "unchecked"))
    print("{:28s} {:10d} {:10d} {:10d}".format(
        "real recipes", pos_ok, pos_no, pos_un))
    print("{:28s} {:10d} {:10d} {:10d}".format(
        "scrambled recipes", neg_ok, neg_no, neg_un))
    print("-" * 62)
    if pos_checked:
        print("true  positive rate  {:.1%}   ({}/{} real accepted)".format(
            pos_ok / pos_checked, pos_ok, pos_checked))
    if neg_checked:
        print("false positive rate  {:.1%}   ({}/{} scrambled accepted)"
              .format(neg_ok / neg_checked, neg_ok, neg_checked))
    if pos_checked and neg_checked:
        print("separation           {:+.1f} points".format(
            100 * (pos_ok / pos_checked - neg_ok / neg_checked)))

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(
        json.dumps({"positives": pos_rows, "negatives": neg_rows,
                    "sample": args.sample, "seed": args.seed},
                   indent=2), "utf-8")
    print("\nwritten to {}".format(args.out))


if __name__ == "__main__":
    main()
