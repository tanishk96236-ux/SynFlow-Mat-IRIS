"""Take generated structures, predict a recipe for each, and test that recipe.

    python scripts/assess_generated.py --input generated_crystals_cif.zip

This is the join between the two halves of the project. Everything before it
answers "what should we make"; this answers "can it be made, and how".

WHY THE INPUT IS SOMEBODY ELSE'S STRUCTURES
--------------------------------------------
MatterGen and GNoME generate structures well and have already published
hundreds of thousands of them. What nobody publishes is whether any of those
can actually be synthesised, because finding out costs months of furnace time
per compound. So the input here is deliberately generator-agnostic: a folder of
CIFs is a folder of CIFs whether MatterGen, GNoME or our own DAO-G wrote it.

MatterGen writes generated_crystals_cif.zip (one CIF per structure) and
generated_crystals.extxyz (all of them as frames). Both are read here, along
with a plain .cif or a directory of them.

WHY --relax IS ON BY DEFAULT
-----------------------------
A diffusion model's output is a prediction of a structure, not a relaxed one.
MatterGen's own evaluation runs DFT relaxation before it reports anything, and
the energies here are meaningless if the atoms sit where a denoiser left them
rather than where the forces vanish. DAO-P relaxes with FIRE instead of DFT,
which is the whole point of having a potential.

Note what relaxation here does NOT do: the cell is held fixed. That is correct
enough for a generated structure, whose lattice the generator predicted, and
wrong for a structure built by substituting a large ion into a small template
-- see the volume finding in docs/. Generated input sidesteps that problem.

WHAT COMES OUT
--------------
One row per structure: the predicted precursors, the predicted firing
temperature, the reaction enthalpy, the gas-entropy term, the free energy, the
thermodynamic floor, and a verdict in plain words. Written as JSON so the next
stage can read it, and printed so a person can.

A refusal is a result. DAO-S returns None for a composition it cannot parse,
and the simulator returns feasible=None when a precursor is missing from
Materials Project. Both are recorded as what they are rather than dropped,
because a screen that silently discards its failures reports a success rate it
did not measure.
"""

import argparse
import io
import json
import os
import pathlib
import sys
import zipfile

# Two Intel OpenMP runtimes abort the process when the second one loads, with
# no traceback and exit code 4. synflow/__init__.py sets this too; kept here so
# the guard does not depend on import order if this file grows imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def load_structures(path: str, limit: int = 0):
    """Structures from a .cif, a directory of them, a .zip, or an .extxyz.

    Yields (name, structure). The name is whatever identifies the structure in
    its source, so a verdict can be traced back to the file it came from.
    """
    from pymatgen.core import Structure

    target = pathlib.Path(path)
    found = []

    if target.is_dir():
        for cif in sorted(target.rglob("*.cif")):
            found.append((cif.stem, Structure.from_file(str(cif))))

    elif target.suffix == ".zip":
        with zipfile.ZipFile(target) as archive:
            for name in sorted(archive.namelist()):
                if not name.lower().endswith(".cif"):
                    continue
                text = archive.read(name).decode("utf-8", "replace")
                found.append((pathlib.Path(name).stem,
                              Structure.from_str(text, fmt="cif")))

    elif target.suffix in (".extxyz", ".xyz"):
        from ase.io import read
        from pymatgen.io.ase import AseAtomsAdaptor
        for i, atoms in enumerate(read(str(target), index=":")):
            found.append(("frame_{}".format(i),
                          AseAtomsAdaptor.get_structure(atoms)))

    else:
        found.append((target.stem, Structure.from_file(str(target))))

    return found[:limit] if limit else found


def load_dao_s():
    """The shipped recipe predictor: ConditionalLookup for precursors,
    TemperatureNet for heat, the dwell network for the hold.

    Returns (model, reason). The model is None when something is missing, and
    the reason says which -- this used to raise SystemExit from inside
    load_recipes, which inherits BaseException and therefore walked straight
    past `except Exception` and killed four discovery runs after their
    expensive stages had already finished.
    """
    # This used to assemble HybridDAOS by hand from runs/dao_s/best.pt -- the
    # September RecipeNet, trained on the corpus the parser was corrupting and
    # selected on precursor Jaccard although only its temperature was used --
    # plus the plain rule. The locked-output demo of 2026-10-06 ran on that,
    # after the better temperature model had already been trained and was
    # sitting unused. The shipped model is now assembled in exactly one place.
    from synflow.dao_s import load_shipped

    path = "runs/dao_s_temperature/best.pt"
    if not os.path.exists(path):
        return None, ("checkpoint {} is missing -- train it with "
                      "scripts/train_temperature.py".format(path))
    try:
        return load_shipped(path), "ok"
    except BaseException as error:                          # noqa: BLE001
        return None, "{}: {}".format(type(error).__name__,
                                     str(error).splitlines()[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help=".cif, a directory of them, a .zip, or an .extxyz")
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many structures (0 = all)")
    parser.add_argument("--relax", dest="relax", action="store_true",
                        default=True)
    parser.add_argument("--no-relax", dest="relax", action="store_false",
                        help="score the structure as given; only correct if it "
                             "is already relaxed")
    parser.add_argument("--relax-steps", type=int, default=80)
    parser.add_argument("--out", default="runs/assessed.json")
    parser.add_argument("--batch-g", type=float, default=2.0,
                        help="grams of target the protocol weighs out for")
    parser.add_argument("--show-protocol", action="store_true", default=True,
                        help="print the full lab protocol for each structure")
    parser.add_argument("--quiet", dest="show_protocol", action="store_false",
                        help="one line per structure only")
    args = parser.parse_args()

    from synflow import dao_g, dao_p, protocol, reaction

    structures = load_structures(args.input, args.limit)
    print("{} structure(s) from {}".format(len(structures), args.input))

    model, reason = load_dao_s()
    if model is None:
        raise SystemExit("DAO-S unavailable: {}".format(reason))
    print("DAO-S: {}".format(model))

    critic = dao_p.load()
    print("DAO-P: {}\n".format(critic))

    rows = []
    for name, structure in structures:
        target = structure.composition.reduced_formula
        row = {"name": name, "target": target,
               "atoms": len(structure), "relaxed": False}

        # 1. a recipe, or a refusal
        recipe = model.predict_recipe(target)
        if recipe is None:
            row["stage"] = "recipe"
            row["error"] = "composition could not be parsed"
            rows.append(row)
            print("{:20s} {:12s} REFUSED (unparseable)".format(name, target))
            continue
        row["precursors"] = recipe["precursors"]
        row["firing_temperature_C"] = recipe["firing_temperature_C"]

        # 2. relax, so the energy is the structure's own minimum and not
        #    wherever the generator's last denoising step happened to stop
        scored = structure
        if args.relax:
            try:
                scored = dao_g.relax(structure, critic, steps=args.relax_steps)
                row["relaxed"] = True
                row["energy_eV_per_atom"] = float(critic.score(scored))
            except Exception as error:                      # noqa: BLE001
                row["relax_error"] = "{}: {}".format(
                    type(error).__name__, str(error)[:120])

        # 3. does that recipe work
        #
        # A structure whose relaxation FAILED is not scored. Its energy is
        # wherever the generator's last step left the atoms, and with the
        # polymorph check that would now be compared against a relaxed known
        # phase -- reporting a gap that is only the missing relaxation. A
        # known formula is then judged as its known phase and says so; a
        # novel one is reported unchecked rather than scored wrong.
        usable = scored if (row["relaxed"] or not args.relax) else None
        if usable is None:
            row["scored"] = False
        result = reaction.assess(
            recipe["precursors"], target, recipe["firing_temperature_C"],
            critic=critic if usable is not None else None, structure=usable)
        row.update({k: v for k, v in result.items()
                    if k not in ("target", "precursors", "temperature_C")})
        row["verdict"] = reaction.verdict(result)

        # 4. the lab protocol -- the output this project is FOR (locked in
        #    docs/target_output.md). Grams, atmosphere and cycles are derived
        #    from the balanced equation, so they exist only once the
        #    simulator has balanced it; a recipe it could not check still
        #    gets a protocol, with the masses marked unavailable.
        lab = protocol.build(target, recipe["precursors"],
                             recipe["firing_temperature_C"],
                             equation=result.get("reaction"),
                             assessment=result, batch_g=args.batch_g,
                             dwell_hours=recipe.get("dwell_hours"))
        row["protocol"] = lab

        if result.get("feasible") is None:
            print("{:20s} {:12s} could not check: {}".format(
                name, target, result.get("error", "?")))
        else:
            print("{:20s} {:12s} {:>5s}  dG {:+7.3f} eV  {}{}".format(
                name, target,
                "UNCRT" if result.get("within_error")
                else "OK" if result["feasible"] else "NO",
                result["free_energy_eV"],
                " + ".join(recipe["precursors"]),
                "   KINETIC RISK" if result.get("kinetic_risk") else ""))
        if args.show_protocol:
            print()
            for line in protocol.render(lab).splitlines():
                print("    " + line)
            print("    VERDICT  " + row["verdict"])
            print()
        rows.append(row)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(rows, indent=2), "utf-8")

    checked = [r for r in rows if r.get("feasible") is not None]
    # Inside the error band the sign of dG is not known, so those rows are
    # "uncertain", not feasible -- the per-row line and the protocol say so.
    uncertain = [r for r in checked if r.get("within_error")]
    feasible = [r for r in checked
                if r["feasible"] and not r.get("within_error")]
    print("\n{} structures, {} checked, {} feasible, {} uncertain".format(
        len(rows), len(checked), len(feasible), len(uncertain)))
    if len(rows) != len(checked):
        print("{} could not be checked -- counted, not discarded".format(
            len(rows) - len(checked)))
    print("written to {}".format(args.out))


if __name__ == "__main__":
    main()
