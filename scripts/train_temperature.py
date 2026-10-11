"""Train the firing-temperature regressor, on the larger corpus.

    python scripts/train_temperature.py --hidden 512 --layers 3 --epochs 60

WHAT THIS FIXES THAT train_dao_s.py DID NOT
-------------------------------------------
1. MODEL SELECTION WAS ON THE WRONG METRIC. train_dao_s.py keeps the epoch
   with the best precursor Jaccard:

       if scores["jaccard"] > best["jaccard"]: torch.save(...)

   and `HybridDAOS` then discards the network's precursors entirely, using the
   metal-lookup rule instead because it wins 0.733 to 0.596. So the shipped
   checkpoint was chosen by a score for a job it does not do, and the score
   for the job it does do -- temperature -- had no say. There is no reason the
   best-Jaccard epoch is also the best-temperature epoch.

2. THE PRECURSORS WERE NOT AN INPUT. They are chosen before the temperature
   is wanted, and they are one of the strongest determinants of it. See
   TemperatureNet.

3. THE TWO LOSSES WERE UNWEIGHTED AND SHARED AN ENCODER. A 463-way head with
   pos_weight near 165 and a single scalar regression both wrote into the same
   representation. Here there is one task.

TRAINED ON THE RULE'S PRECURSORS, NOT THE TRUE ONES
---------------------------------------------------
The obvious thing is to feed the recorded precursors during training. That
would be teacher forcing, and at inference the model gets the rule's
predictions instead, which are right about 59% of the time exactly. Training
on inputs it will never see again is how a model learns to depend on a
precision it will not be given.

So the rule's prediction is the input in training too. The rule is fitted on
the training split only, so this leaks nothing into validation or test, and
train and inference then see distributions that match.
"""

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402


def build_inputs(recipes, rule, vocabulary, index):
    """(composition, precursor multi-hot, temperature) for a split."""
    from synflow.dao_s import composition_tensor

    x, p, t = [], [], []
    for recipe in recipes:
        if recipe.max_temperature is None:
            continue
        predicted = rule.predict(recipe.target)
        if not predicted:
            continue
        row = torch.zeros(len(vocabulary))
        for precursor in predicted:
            if precursor in index:
                row[index[precursor]] = 1.0
        x.append(composition_tensor(recipe.target))
        p.append(row)
        t.append(recipe.max_temperature / 100.0)
    if not x:
        raise RuntimeError("no usable rows")
    return torch.stack(x), torch.stack(p), torch.tensor(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpus", default="llm", choices=["llm", "kononova"])
    parser.add_argument("--precursors", default="conditional",
                        choices=["conditional", "rule"],
                        help="which precursor predictor supplies the "
                             "network's precursor input; must match the one "
                             "used at inference")
    parser.add_argument("--min-count", type=int, default=3,
                        help="ConditionalLookup's trust threshold, chosen on "
                             "validation by scripts/eval_conditional.py")
    parser.add_argument("--out", default="runs/dao_s_temperature")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    from synflow.dao_s import TemperatureNet
    from synflow.recipe_metrics import MetalLookupBaseline
    from synflow.recipes import precursor_vocabulary, split_by_target

    if args.corpus == "llm":
        from synflow.recipes_llm import load_recipes
    else:
        from synflow.recipes import load_recipes

    recipes = load_recipes(verbose=True)
    splits = split_by_target(recipes, seed=0)
    print("split by target:", {k: len(v) for k, v in splits.items()})

    overlap = ({r.target for r in splits["train"]}
               & {r.target for r in splits["test"]})
    if overlap:
        raise RuntimeError("target leak: {}".format(sorted(overlap)[:5]))

    # The network's precursor input must come from the SAME predictor that
    # will supply it at inference. It was first trained on the plain rule's
    # output; switching the shipped system to ConditionalLookup without
    # retraining would hand it a distribution it never saw -- the same
    # train/inference mismatch the docstring warns about for teacher forcing.
    # Fitted on the training split only, so no validation or test recipe
    # shapes the inputs it is scored on.
    if args.precursors == "conditional":
        from synflow.recipe_metrics import ConditionalLookup
        rule = ConditionalLookup(splits["train"], min_count=args.min_count)
    else:
        rule = MetalLookupBaseline(splits["train"])
    print("precursor input from: {}".format(
        "ConditionalLookup(min_count={})".format(args.min_count)
        if args.precursors == "conditional" else "plain metal-lookup rule"))
    vocabulary = precursor_vocabulary(splits["train"])
    index = {p: i for i, p in enumerate(vocabulary)}
    print("precursor vocabulary:", len(vocabulary))

    train = build_inputs(splits["train"], rule, vocabulary, index)
    val = build_inputs(splits["val"], rule, vocabulary, index)
    test = build_inputs(splits["test"], rule, vocabulary, index)
    print("rows: train {:,}  val {:,}  test {:,}".format(
        len(train[2]), len(val[2]), len(test[2])))

    mean = float(train[2].mean())
    print("mean firing temperature: {:.0f} C".format(mean * 100))

    model = TemperatureNet(len(vocabulary), hidden=args.hidden,
                           layers=args.layers, dropout=args.dropout,
                           mean_temperature=mean)
    parameters = sum(p.numel() for p in model.parameters())
    print("model: {:,} parameters against {:,} training rows".format(
        parameters, len(train[2])))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.SmoothL1Loss()

    # THE BASELINE THAT HAS TO BE BEATEN, printed before training starts.
    # Predicting the training mean for everything is what "no model" scores,
    # and a regressor that cannot beat it has learned nothing whatever its
    # loss curve looks like.
    floor = float((val[2] - mean).abs().mean()) * 100
    rule_temps = []
    for recipe in splits["val"]:
        if recipe.max_temperature is None:
            continue
        guess = rule.predict_temperature(recipe.target)
        if guess:
            rule_temps.append(abs(guess - recipe.max_temperature))
    print("\nBASELINES on validation")
    print("   predict the mean   {:.0f} C".format(floor))
    if rule_temps:
        print("   metal-lookup rule  {:.0f} C".format(
            sum(rule_temps) / len(rule_temps)))

    def scores(inputs):
        """Everything worth watching, in one pass.

        MAE alone is a poor thing to watch a regression by: it is dominated by
        the tail, so it can sit still while the bulk of the predictions get
        better. The median says where the typical prediction is, and the two
        hit rates are the only figures here a chemist reads directly -- "nine
        in ten of these land within 200 C of what was actually used" is a
        sentence; "MAE 178" is not.

        A hit rate is also the honest way to show accuracy for a regression.
        There is no such thing as a correct firing temperature to the degree:
        the same compound is made at different temperatures by different
        groups, and within this corpus the median disagreement between two
        recorded recipes for one target is 55 C.
        """
        model.eval()
        with torch.no_grad():
            error = (model(inputs[0], inputs[1]) - inputs[2]).abs() * 100
        return {
            "mae_C": float(error.mean()),
            "median_C": float(error.median()),
            "within_100_C": float((error <= 100).float().mean()),
            "within_200_C": float((error <= 200).float().mean()),
        }

    n = len(train[2])
    best = {"val_mae_C": float("inf")}
    history = []
    os.makedirs(args.out, exist_ok=True)

    print("\n{:>5s} {:>9s} {:>8s} {:>8s} {:>9s} {:>9s}".format(
        "epoch", "loss", "MAE C", "med C", "<=100C", "<=200C"))
    print("-" * 54)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n)
        total = 0.0
        for start in range(0, n, args.batch_size):
            idx = order[start:start + args.batch_size]
            optimizer.zero_grad()
            predicted = model(train[0][idx], train[1][idx])
            loss = criterion(predicted, train[2][idx])
            loss.backward()
            optimizer.step()
            # .detach() first: the loss still carries the graph it was
            # backpropagated through, and reading a scalar off a tensor that
            # requires grad warns for a real reason -- it keeps the graph
            # alive for as long as the Python float does.
            total += float(loss.detach()) * len(idx)
        scheduler.step()

        measured = scores(val)
        val_mae = measured["mae_C"]
        history.append(dict(epoch=epoch, train_loss=total / n,
                            val_mae_C=val_mae, **measured))

        flag = ""
        # SELECTED ON THE METRIC THIS MODEL IS ACTUALLY USED FOR.
        if val_mae < best["val_mae_C"]:
            best = {"epoch": epoch, "val_mae_C": val_mae, **measured}
            flag = "  <- best"
            torch.save({"model": model.state_dict(),
                        "vocabulary": vocabulary,
                        "hidden": args.hidden, "layers": args.layers,
                        "dropout": args.dropout,
                        "mean_temperature": mean,
                        "val_mae_C": val_mae, "scores": measured,
                        "epoch": epoch,
                        # what the precursor input was drawn from, so a
                        # loader can refuse a mismatched pairing
                        "precursors": args.precursors,
                        "min_count": args.min_count,
                        "corpus": args.corpus},
                       os.path.join(args.out, "best.pt"))
        if epoch % 2 == 0 or epoch == 1 or flag:
            print("{:5d} {:9.4f} {:8.1f} {:8.1f} {:8.1%} {:8.1%}{}".format(
                epoch, total / n, val_mae, measured["median_C"],
                measured["within_100_C"], measured["within_200_C"], flag))

    # Test is touched once, with the selected checkpoint, at the very end.
    blob = torch.load(os.path.join(args.out, "best.pt"),
                      weights_only=False, map_location="cpu")
    model.load_state_dict(blob["model"])
    on_test = scores(test)

    # The same four figures for the two things that need no training, on the
    # same test rows, so the model's numbers can be read against something.
    mean_error = (test[2] - mean).abs() * 100
    baseline = {
        "mae_C": float(mean_error.mean()),
        "median_C": float(mean_error.median()),
        "within_100_C": float((mean_error <= 100).float().mean()),
        "within_200_C": float((mean_error <= 200).float().mean()),
    }

    print("\n{}".format("=" * 68))
    print("TEST SET -- {:,} rows, touched once, with the epoch-{} checkpoint"
          .format(len(test[2]), blob["epoch"]))
    print("{}".format("=" * 68))
    print("{:22s} {:>9s} {:>9s} {:>10s} {:>10s}".format(
        "", "MAE C", "median C", "within100", "within200"))
    print("-" * 68)
    print("{:22s} {:9.1f} {:9.1f} {:10.1%} {:10.1%}".format(
        "predict the mean", baseline["mae_C"], baseline["median_C"],
        baseline["within_100_C"], baseline["within_200_C"]))
    print("{:22s} {:9.1f} {:9.1f} {:10.1%} {:10.1%}".format(
        "OUR MODEL", on_test["mae_C"], on_test["median_C"],
        on_test["within_100_C"], on_test["within_200_C"]))
    print("-" * 68)
    print("{:22s} {:,} parameters, best at epoch {}".format(
        "", parameters, blob["epoch"]))
    print("{:22s} MAE is {:.0f} C better than the floor ({:+.0%})".format(
        "", baseline["mae_C"] - on_test["mae_C"],
        on_test["mae_C"] / baseline["mae_C"] - 1.0))

    pathlib.Path(os.path.join(args.out, "history.json")).write_text(
        json.dumps({"history": history, "best": best,
                    "test": on_test, "test_baseline_mean": baseline,
                    "test_mae_C": on_test["mae_C"],
                    "test_median_C": on_test["median_C"],
                    "floor_C": floor, "parameters": parameters,
                    "hidden": args.hidden, "layers": args.layers,
                    "corpus": args.corpus}, indent=2), "utf-8")


if __name__ == "__main__":
    main()
