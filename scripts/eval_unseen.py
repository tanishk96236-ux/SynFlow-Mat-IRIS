"""DAO-S against data from a different corpus, extracted a different way.

    python scripts/eval_unseen.py

WHY NOT JUST USE THE HELD-OUT SPLIT
-----------------------------------
A split by target inside one corpus answers "does this generalise to compounds
it was not trained on". It cannot answer "does this generalise beyond the
quirks of one extraction pipeline", and that second question is the one a
reviewer asks about any text-mined result.

So the test set here is from the other corpus entirely:

    train / select   LLM corpus, gpt-4o extracted      46,344 recipes
    test             Kononova et al. 2019, hand-built   1,047 recipes
                     pipeline, MINUS every target that
                     appears anywhere in training

Two things are unseen at once: the compounds, and the way the text was read.
1,047 recipes over 973 distinct targets survive that filter.

Formulas are canonicalised through pymatgen before the overlap is taken.
Comparing raw strings would have left "Li3NiMnCoO6" and "Li3MnCoNiO6" looking
like different compounds, which is a mistake this project has already made
once -- it concluded Materials Project held no NMC cathodes.

WHAT IS REPORTED, AND AGAINST WHAT
----------------------------------
Every number comes with the baselines computed on the SAME unseen recipes,
because a figure with nothing beside it cannot be read:

    most-common      always predict the training set's commonest precursors
    predict the mean always predict the training set's mean temperature
    metal-lookup     the zero-parameter rule
    ours             the shipped hybrid

The test set is touched once, at the end, with the checkpoint chosen on the
LLM validation split.
"""

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch                                                    # noqa: E402


def canonical(formula):
    from pymatgen.core import Composition
    try:
        return Composition(formula).reduced_formula
    except Exception:                                           # noqa: BLE001
        return None


def unseen_test_set(train_recipes, candidate_recipes):
    """Candidates whose target never appears in training, canonically."""
    seen = {c for c in (canonical(r.target) for r in train_recipes) if c}
    out = []
    for recipe in candidate_recipes:
        key = canonical(recipe.target)
        if key and key not in seen and recipe.max_temperature is not None:
            out.append(recipe)
    return out


def jaccard(predicted, reference):
    predicted, reference = set(predicted), set(reference)
    if not predicted and not reference:
        return 1.0
    union = predicted | reference
    return len(predicted & reference) / len(union) if union else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature-checkpoint",
                        default="runs/dao_s_temperature/best.pt")
    parser.add_argument("--out", default="runs/unseen_evaluation.json")
    parser.add_argument("--k", type=int, default=25,
                        help="neighbours; 25 was chosen on validation")
    args = parser.parse_args()

    from synflow import recipes_llm as RL
    from synflow.dao_s import TemperatureNet, composition_tensor
    from synflow.recipe_metrics import MetalLookupBaseline
    from synflow.recipes import load_recipes as load_kononova
    from synflow.recipes import group_by_target, precursor_vocabulary

    train = RL.load_recipes(verbose=True)
    kononova = load_kononova(verbose=False)
    test = unseen_test_set(train, kononova)

    print("\ntrain (LLM corpus)      {:6,} recipes".format(len(train)))
    print("candidate (Kononova)    {:6,} recipes".format(len(kononova)))
    print("UNSEEN test set         {:6,} recipes, {:,} targets".format(
        len(test), len({canonical(r.target) for r in test})))
    if not test:
        raise SystemExit("no unseen recipes -- nothing to measure")

    # A target can have several recorded recipes, and 62% of multi-recipe
    # targets disagree on the precursors. So a prediction is credited against
    # the CLOSEST recorded recipe for that target rather than against one
    # arbitrary row, which would mark a valid route wrong for not being the
    # one that happened to be indexed first.
    references = group_by_target(test)

    rule = MetalLookupBaseline(train)
    vocabulary = precursor_vocabulary(train)
    index = {p: i for i, p in enumerate(vocabulary)}

    # most-common baseline: the commonest precursor set size in training,
    # filled with the globally commonest precursors
    import collections
    counts = collections.Counter(p for r in train for p in r.precursors)
    common = [p for p, _ in counts.most_common(3)]
    train_mean_C = (sum(r.max_temperature for r in train
                        if r.max_temperature is not None)
                    / max(sum(1 for r in train
                              if r.max_temperature is not None), 1))
    print("training mean temperature {:.0f} C".format(train_mean_C))

    model = None
    if os.path.exists(args.temperature_checkpoint):
        blob = torch.load(args.temperature_checkpoint, weights_only=False,
                          map_location="cpu")
        model = TemperatureNet(len(blob["vocabulary"]),
                               hidden=blob["hidden"], layers=blob["layers"],
                               dropout=blob["dropout"],
                               mean_temperature=blob["mean_temperature"])
        model.load_state_dict(blob["model"])
        model.eval()
        ckpt_vocabulary = blob["vocabulary"]
        ckpt_index = {p: i for i, p in enumerate(ckpt_vocabulary)}
        print("temperature model: {:,} parameters, val MAE {:.1f} C "
              "at epoch {}".format(
                  sum(p.numel() for p in model.parameters()),
                  blob["val_mae_C"], blob["epoch"]))
    else:
        blob = {}
        print("no temperature checkpoint at {} -- temperature columns will "
              "show baselines only".format(args.temperature_checkpoint))

    # The precursors that are scored, and fed to the network, come from the
    # predictor the checkpoint was TRAINED against -- read from the
    # checkpoint, not chosen here, so the two cannot be mismatched. The plain
    # rule is still scored alongside as the baseline it has to beat.
    if blob.get("precursors") == "conditional":
        from synflow.recipe_metrics import ConditionalLookup
        shipped = ConditionalLookup(train, min_count=blob.get("min_count", 3))
        shipped_name = "ConditionalLookup(min_count={})".format(
            blob.get("min_count", 3))
    else:
        shipped, shipped_name = rule, "metal-lookup rule"
    print("shipped precursors: {}".format(shipped_name))

    # k-nearest neighbours on the SAME training rows the network saw (the
    # train split of the LLM corpus), with k chosen on its validation split
    # by scripts/knn_baseline.py. Unseen targets are new to both.
    from synflow.recipes import split_by_target
    knn_rows = [(r.target, r.max_temperature)
                for r in split_by_target(train, seed=0)["train"]
                if r.max_temperature is not None]
    knn_x = torch.stack([composition_tensor(t) for t, _ in knn_rows])
    knn_y = torch.tensor([v for _, v in knn_rows], dtype=torch.float32)

    def knn_temperature(target):
        d = torch.cdist(composition_tensor(target).unsqueeze(0), knn_x)[0]
        idx = d.topk(min(args.k, len(knn_y)), largest=False).indices
        return float(knn_y[idx].median())

    rows = []
    for target, group in references.items():
        recorded = [set(r.precursors) for r in group]
        truth_T = [r.max_temperature for r in group
                   if r.max_temperature is not None]
        if not truth_T:
            continue

        plain = set(rule.predict(target))
        predicted = set(shipped.predict(target))
        row = {
            "target": target,
            "n_recorded_recipes": len(group),
            "predicted": sorted(predicted),
            "jaccard_rule": max(jaccard(plain, r) for r in recorded),
            "exact_rule": any(plain == r for r in recorded),
            "jaccard_shipped": max(jaccard(predicted, r) for r in recorded),
            "exact_shipped": any(predicted == r for r in recorded),
            "jaccard_common": max(jaccard(common, r) for r in recorded),
            "truth_temperature_C": max(truth_T),
        }

        row["error_mean_C"] = abs(train_mean_C - row["truth_temperature_C"])
        rule_T = rule.predict_temperature(target)
        row["error_rule_C"] = (abs(rule_T - row["truth_temperature_C"])
                               if rule_T else None)

        if model is not None:
            multi_hot = torch.zeros(len(ckpt_vocabulary))
            for precursor in predicted:
                if precursor in ckpt_index:
                    multi_hot[ckpt_index[precursor]] = 1.0
            with torch.no_grad():
                guess = float(model(composition_tensor(target).unsqueeze(0),
                                    multi_hot.unsqueeze(0))[0]) * 100.0
            row["predicted_temperature_C"] = guess
            row["error_model_C"] = abs(guess - row["truth_temperature_C"])
        near = knn_temperature(target)
        row["error_knn_C"] = abs(near - row["truth_temperature_C"])
        if model is not None:
            row["error_average_C"] = abs((near + guess) / 2
                                         - row["truth_temperature_C"])
        rows.append(row)

    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    def median(key):
        values = sorted(r[key] for r in rows if r.get(key) is not None)
        return values[len(values) // 2] if values else None

    print("\n{}".format("=" * 64))
    print("UNSEEN EVALUATION -- {} targets from the other corpus".format(
        len(rows)))
    print("{}".format("=" * 64))
    print("\nPRECURSORS")
    print("   most-common baseline    jaccard {:.3f}".format(
        mean("jaccard_common")))
    print("   metal-lookup rule       jaccard {:.3f}   exact {:.1%}".format(
        mean("jaccard_rule"),
        sum(1 for r in rows if r["exact_rule"]) / len(rows)))
    if shipped is not rule:
        print("   SHIPPED ({})".format(shipped_name))
        print("                           jaccard {:.3f}   exact {:.1%}".format(
            mean("jaccard_shipped"),
            sum(1 for r in rows if r["exact_shipped"]) / len(rows)))
    print("\nTEMPERATURE  (mean | median absolute error)")
    print("   predict the mean        {:5.0f} C | {:5.0f} C".format(
        mean("error_mean_C"), median("error_mean_C")))
    if mean("error_rule_C") is not None:
        print("   metal-lookup rule       {:5.0f} C | {:5.0f} C".format(
            mean("error_rule_C"), median("error_rule_C")))
    if mean("error_model_C") is not None:
        print("   OUR MODEL               {:5.0f} C | {:5.0f} C".format(
            mean("error_model_C"), median("error_model_C")))
    print("   k-NN, k={:<3d}            {:5.0f} C | {:5.0f} C".format(
        args.k, mean("error_knn_C"), median("error_knn_C")))
    if mean("error_average_C") is not None:
        print("   average of the two      {:5.0f} C | {:5.0f} C".format(
            mean("error_average_C"), median("error_average_C")))

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps({
        "n_targets": len(rows),
        "n_recipes": len(test),
        "train_corpus": "llm", "test_corpus": "kononova_2019_unseen",
        "precursors": {
            "most_common_jaccard": mean("jaccard_common"),
            "rule_jaccard": mean("jaccard_rule"),
            "rule_exact": sum(1 for r in rows if r["exact_rule"]) / len(rows),
            "shipped": shipped_name,
            "shipped_jaccard": mean("jaccard_shipped"),
            "shipped_exact":
                sum(1 for r in rows if r["exact_shipped"]) / len(rows),
        },
        "temperature_C": {
            "mean_baseline": mean("error_mean_C"),
            "rule": mean("error_rule_C"),
            "model": mean("error_model_C"),
            "model_median": median("error_model_C"),
            "knn": mean("error_knn_C"),
            "knn_median": median("error_knn_C"),
            "average": mean("error_average_C"),
            "average_median": median("error_average_C"),
        },
        "rows": rows,
    }, indent=2), "utf-8")
    print("\nwritten to {}".format(args.out))


if __name__ == "__main__":
    main()
