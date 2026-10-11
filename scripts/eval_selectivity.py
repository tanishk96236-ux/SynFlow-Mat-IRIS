"""Does primary competition (C1) tell success from failure?

    python scripts/eval_selectivity.py --alab        # the A-Lab's 57 targets
    python scripts/eval_selectivity.py --corpus 60   # 60 + 60 corpus recipes

TWO TESTS, BOTH ON OUTCOMES NOBODY TUNED C1 ON
----------------------------------------------
A-Lab   57 targets with the robot's result. "Is the target reaction downhill"
        was yes for all of them and separated nothing; the question is
        whether C1 is higher for the 15 it failed on.

corpus  the LLM-extracted corpus records what else came out of the furnace
        (Sci. Data 2025). From the TEST split, recipes whose impurity is a
        DIFFERENT phase -- not leftover precursor, which is incomplete reaction
        and a kinetic matter -- against recipes with no impurity recorded,
        matched in number. C1 should be higher for the first group.

"No impurity recorded" is weaker than "clean": many papers do not report
minor phases, and the extraction finds impurity phases with precision 0.86
and recall 0.90. So the clean group is partly mislabelled, and that can only
pull the measured separation toward 0.5.

A recipe whose system cannot be scored is counted as such, not dropped.
Each result is checkpointed, because Materials Project fetches of large
chemical systems take minutes each on this machine.
"""

import argparse
import json
import os
import pathlib
import random
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def reduced(formula):
    from pymatgen.core import Composition
    try:
        return Composition(formula).reduced_formula
    except Exception:                                           # noqa: BLE001
        return None


def _done(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    entry = json.loads(line)
                    out[entry["id"]] = entry
    return out


def _score_one(case):
    """One recipe; top level so a process pool can call it."""
    from synflow import selectivity
    try:
        result = selectivity.primary_competition(
            case["precursors"], case["target"], case["temperature_C"])
    except Exception as error:                                  # noqa: BLE001
        result = {"C1_eV_per_atom": None,
                  "error": "{}: {}".format(type(error).__name__,
                                           str(error)[:100])}
    remaining = result.get("remaining") or {}
    return dict(case, C1=result.get("C1_eV_per_atom"),
                target_dG=result.get("target_dG_eV_per_atom"),
                competitor=(result.get("strongest_competitor") or {})
                .get("product"),
                remaining=remaining.get("dG_eV_per_atom"),
                remaining_from=remaining.get("from_"),
                error=result.get("error"))


def score(cases, path, workers=1):
    """Score every case not already in the checkpoint at `path`.

    The work is waiting on Materials Project, not computing, so on Colab a
    few processes in parallel finish several times sooner.
    """
    done = _done(path)
    todo = [c for c in cases if c["id"] not in done]
    rows = [done[c["id"]] for c in cases if c["id"] in done]

    def record(entry):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
        rows.append(entry)
        print("  {:24s} {:10s} C1 {}  {}".format(
            entry["target"][:24], entry["label"],
            "   -   " if entry["C1"] is None
            else "{:+.3f}".format(entry["C1"]),
            entry["competitor"] or entry["error"] or ""), flush=True)

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(workers) as pool:
            for future in as_completed([pool.submit(_score_one, c)
                                        for c in todo]):
                record(future.result())
    else:
        for case in todo:
            record(_score_one(case))
    return rows


def report(rows, positive, negative):
    """Each metric per group, and the rank AUC.

    Stated before this run (2026-10-08, v3), the two hypotheses:
      C1         higher for the failures -- a rival was more favoured
      remaining  higher (closer to zero, or positive) for the failures --
                 too little left to drive the target once the rival formed,
                 the A-Lab's own account of its stalls
    """
    for key, label in (("C1", "C1"), ("remaining", "remaining dG")):
        groups = {}
        for group in (positive, negative):
            values = [r[key] for r in rows if r["label"] == group
                      and r.get(key) is not None]
            unscored = sum(1 for r in rows if r["label"] == group
                           and r.get(key) is None)
            groups[group] = values
            if values:
                ordered = sorted(values)
                print("   {:12s} {:12s} n={:3d} (none {:2d})  mean {:+.3f}  "
                      "median {:+.3f}".format(
                          label, group, len(values), unscored,
                          sum(values) / len(values),
                          ordered[len(ordered) // 2]))
        pos, neg = groups[positive], groups[negative]
        if pos and neg:
            wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
            print("   AUC, higher {} for '{}': {:.3f}   (0.5 = no "
                  "signal)\n".format(label, positive,
                                     wins / (len(pos) * len(neg))))


def alab_cases(path="runs/alab_targets.json"):
    """OUR recipe for every target, from the A-Lab run already made.

    The robot recorded a recipe only where it succeeded, so scoring its
    recipe for the successes and ours for the failures would compare two
    recipe sources, not two outcomes.
    """
    with open(path, encoding="utf-8") as handle:
        targets = json.load(handle)
    cases = []
    for row in targets:
        recipe = row.get("ours")
        if not recipe:
            continue
        cases.append({"id": "ours:" + row["target"],
                      "label": "failed" if row["result"] == "Failed"
                      else "made",
                      "target": row["target"],
                      "precursors": recipe["precursors"],
                      "temperature_C": recipe["firing_temperature_C"]})
    return cases


def corpus_cases(n, seed=0):
    from synflow import recipes_llm
    from synflow.recipes import split_by_target

    test = split_by_target(recipes_llm.load_recipes(verbose=False),
                           seed=seed)["test"]
    usable = [r for r in test if r.max_temperature and 2 <= len(r.precursors)
              <= 3 and r.mp_id]
    other, clean = [], []
    for r in usable:
        precursors = {reduced(p) for p in r.precursors}
        phases = {reduced(p) for p in r.impurity_phases} - {None}
        if not r.impurity_phases:
            clean.append(r)
        elif phases and not phases <= precursors:
            other.append(r)            # a phase that is not a precursor
    rng = random.Random(seed)
    picked = rng.sample(other, min(n, len(other))) + \
        rng.sample(clean, min(n, len(clean)))
    return [{"id": "corpus:{}:{}".format(r.target, "+".join(r.precursors)),
             "label": "impurity" if r.impurity_phases else "clean",
             "target": r.target, "precursors": list(r.precursors),
             "temperature_C": r.max_temperature} for r in picked]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alab", action="store_true")
    parser.add_argument("--corpus", type=int, default=0)
    parser.add_argument("--export-cases", default=None,
                        help="write the cases to this JSON and stop")
    parser.add_argument("--cases", default=None,
                        help="score cases from this JSON (e.g. on Colab)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only", choices=("alab", "corpus"), default=None,
                        help="score only one group of the cases file")
    args = parser.parse_args()

    if args.cases:
        with open(args.cases, encoding="utf-8") as handle:
            cases = json.load(handle)
        prefix = {"alab": "ours:", "corpus": "corpus:"}.get(args.only)
        if prefix:
            cases = [c for c in cases if c["id"].startswith(prefix)]
    else:
        cases = (alab_cases() if args.alab else []) + \
            (corpus_cases(args.corpus) if args.corpus else [])

    if args.export_cases:
        pathlib.Path(args.export_cases).parent.mkdir(parents=True,
                                                     exist_ok=True)
        pathlib.Path(args.export_cases).write_text(
            json.dumps(cases, indent=1), "utf-8")
        print("{} cases written to {}".format(len(cases), args.export_cases))
        return

    pathlib.Path("runs").mkdir(exist_ok=True)
    rows = score(cases, "runs/selectivity_v3.jsonl", args.workers)
    alab = [r for r in rows if r["id"].startswith("ours:")]
    corpus = [r for r in rows if r["id"].startswith("corpus:")]
    if alab:
        print("\nA-LAB TARGETS -- our recipe for each; failed vs made")
        report(alab, "failed", "made")
    if corpus:
        print("\nCORPUS TEST SPLIT -- impurity of another phase vs none")
        report(corpus, "impurity", "clean")


if __name__ == "__main__":
    main()
