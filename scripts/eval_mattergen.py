"""The whole system on MatterGen's four lab-tested candidates.

    python scripts/eval_mattergen.py

THE CASE
--------
Zeni et al., "A generative model for inorganic materials design", Nature
(2025), Supplementary Information D.10.4-D.10.6. MatterGen generated
candidates for target bulk moduli; four passed the filters and were made in
a real lab. The CIFs are from github.com/microsoft/mattergen
(data-release/cifs.zip, cifs/mattergen/experimental/).

    TaCr2O6   Cr + Ta2O5 (4:1), ball-milled 12 h, 1200 C 12 h in air,
              pelleted, 1500 C 36 h            -> MADE: 98% disordered rutile
                                                  TaCr2O6, 2% Cr2O3
    Cr2MoO6   Cr2O3 + MoO3, 750 C 24 h; 900 C    -> FAILED: no crystalline
                                                  product; Mo lost at 900 C
    LaMoO4    La2O3 + Mo, 750 C 24 h; 900 C      -> FAILED: likewise
    Mn3NiO6   Mn2O3 + Ni2O3, 1100 C 24 h         -> FAILED: the known
                                                  Mn2NiO4 formed instead

This is the project's whole question on a generator's own output, with the
answers known: a CIF goes in, a protocol and a verdict come out, and each is
held against what the lab did and got.

For each candidate: OUR recipe through the full pipeline (relax, DAO-S,
simulator with the structure scored, kinetics, protocol), and the LAB's
recipe through the same simulator at the lab's temperatures. Nothing here was
tuned on these four.
"""

import json
import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

CIFS = "data/mattergen/experimental"
LAB = {
    "TaCr2O6": {"precursors": ["Cr", "Ta2O5"], "temperatures_C": [1200, 1500],
                "hours": [12, 36], "outcome": "made",
                "found": "98% disordered rutile TaCr2O6 + 2% Cr2O3"},
    "Cr2MoO6": {"precursors": ["Cr2O3", "MoO3"], "temperatures_C": [750, 900],
                "hours": [24, None], "outcome": "failed",
                "found": "no crystalline product at 750 C; Mo lost at 900 C"},
    "LaMoO4": {"precursors": ["La2O3", "Mo"], "temperatures_C": [750, 900],
               "hours": [24, None], "outcome": "failed",
               "found": "no crystalline product at 750 C; Mo lost at 900 C"},
    "Mn3NiO6": {"precursors": ["Mn2O3", "Ni2O3"], "temperatures_C": [1100],
                "hours": [24], "outcome": "failed",
                "found": "the known Mn2NiO4 formed instead"},
}


def brief(result):
    keys = ("reaction", "verdict_class", "enthalpy_eV", "free_energy_eV",
            "driving_force_meV_per_atom", "error_band_meV_per_atom",
            "onset_temperature_C", "kinetic_risk", "polymorph",
            "target_stability", "error")
    return {k: result.get(k) for k in keys}


def main():
    from assess_generated import load_dao_s, load_structures
    from synflow import dao_g, dao_p, kinetics, protocol, reaction, selectivity

    model, reason = load_dao_s()
    if model is None:
        raise SystemExit("DAO-S unavailable: {}".format(reason))
    critic = dao_p.load()

    rows = []
    for name, structure in load_structures(CIFS):
        target = structure.composition.reduced_formula
        lab = LAB.get(target)
        print("\n" + "=" * 72)
        print("{}  (MatterGen; lab: {} -- {})".format(
            target, lab["outcome"].upper(), lab["found"]))
        print("=" * 72)
        row = {"target": target, "lab": lab}
        try:
            relaxed = dao_g.relax(structure, critic, steps=80)
        except Exception as error:                              # noqa: BLE001
            print("relaxation failed: {}".format(error))
            rows.append(dict(row, error=str(error)))
            continue

        # ---- ours
        recipe = model.predict_recipe(target)
        row["ours"] = recipe
        if recipe:
            ours = reaction.assess(recipe["precursors"], target,
                                   recipe["firing_temperature_C"],
                                   critic=critic, structure=relaxed)
            row["ours_assessment"] = brief(ours)
            lab_protocol = protocol.build(
                target, recipe["precursors"], recipe["firing_temperature_C"],
                equation=ours.get("reaction"), assessment=ours,
                dwell_hours=recipe.get("dwell_hours"))
            print("OUR RECIPE: {} at {} C, ~{} h".format(
                " + ".join(recipe["precursors"]),
                recipe["firing_temperature_C"], recipe.get("dwell_hours")))
            print("   verdict: {}".format(reaction.verdict(ours)))
            for note in lab_protocol.get("temperature_checks") or []:
                print("   CHECK: {}".format(note))
            for pair in lab_protocol["kinetics"].get("pairs", []):
                if pair["evidence"]:
                    print("   kinetics: {} + {} {} {}".format(
                        *pair["pair"], pair["evidence"], pair["seen"]))
            try:
                sel = selectivity.primary_competition(
                    recipe["precursors"], target,
                    recipe["firing_temperature_C"])
                row["ours_selectivity"] = {
                    "C1": sel.get("C1_eV_per_atom"),
                    "rival": (sel.get("strongest_competitor") or {})
                    .get("product"),
                    "remaining": (sel.get("remaining") or {})
                    .get("dG_eV_per_atom")}
                print("   selectivity: {}".format(selectivity.explain(sel)))
            except Exception as error:                          # noqa: BLE001
                print("   selectivity: not computed ({})".format(
                    str(error)[:80]))

        # ---- the lab's
        print("LAB RECIPE: {} at {} C".format(
            " + ".join(lab["precursors"]),
            " then ".join(str(t) for t in lab["temperatures_C"])))
        row["lab_assessment"] = []
        for temperature in lab["temperatures_C"]:
            theirs = reaction.assess(lab["precursors"], target, temperature,
                                     critic=critic, structure=relaxed)
            row["lab_assessment"].append(dict(brief(theirs),
                                              temperature_C=temperature))
            print("   at {} C: {}".format(temperature,
                                          reaction.verdict(theirs)))
        report = kinetics.check_recipe(lab["precursors"],
                                       max(lab["temperatures_C"]),
                                       lab["hours"][0])
        for pair in report["pairs"]:
            if pair["evidence"]:
                print("   kinetics: {} + {} {} {}".format(
                    *pair["pair"], pair["evidence"], pair["seen"]))
        rows.append(row)

    pathlib.Path("runs").mkdir(exist_ok=True)
    pathlib.Path("runs/mattergen_showcase.json").write_text(
        json.dumps(rows, indent=2, default=str), "utf-8")
    print("\nwritten to runs/mattergen_showcase.json")


if __name__ == "__main__":
    main()
