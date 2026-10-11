"""Train the dwell-time model: how long the firing step is held.

    python scripts/train_dwell.py --epochs 30

The last PREDICTED field of the locked output that was missing. Measured on
the corpus before anything was trained:

    32,373 usable recipes; median 6 h, 5th-95th percentile 0.5-72 h
    commonest values 2, 4, 12, 24, 3, 1, 10, 6, 5, 48 h

Round numbers. Dwell time is often set by habit -- "overnight", "a day" --
rather than by the chemistry, so this field may be close to unpredictable,
and the baseline decides whether a model earns its place.

WHY THE ERROR IS A FACTOR, NOT HOURS
------------------------------------
The values span two orders of magnitude. Being 5 h off is a disaster for a
1 h firing and noise for a 72 h one. So the target is ln(hours), and the
error is reported as how many TIMES off the prediction is:

    within 2x   a 6 h prediction for anything from 3 to 12 h

BASELINE, measured before training: predict the training median, 6 h, for
everything -- within 2x 45.7%, within 3x 60.9%, median 3.00x off.

The same architecture and precursor inputs as the temperature model, for the
same reason: the lookup's own predictions are the input at inference, so they
are the input in training.
"""

import argparse
import json
import math
import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402


def build(recipes, lookup, index):
    from synflow.dao_s import composition_tensor

    x, p, y = [], [], []
    for recipe in recipes:
        if not recipe.dwell_hours:
            continue
        predicted = lookup.predict(recipe.target)
        if not predicted:
            continue
        row = torch.zeros(len(index))
        for precursor in predicted:
            if precursor in index:
                row[index[precursor]] = 1.0
        x.append(composition_tensor(recipe.target))
        p.append(row)
        y.append(math.log(recipe.dwell_hours))
    return torch.stack(x), torch.stack(p), torch.tensor(y)


def factor_scores(log_predicted, log_true):
    """within 2x, within 3x, and the median factor off."""
    factor = torch.exp((log_predicted - log_true).abs())
    return {"within_2x": float((factor <= 2).float().mean()),
            "within_3x": float((factor <= 3).float().mean()),
            "median_factor": float(factor.median())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/dao_s_dwell")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    from synflow import recipes_llm as RL
    from synflow.dao_s import TemperatureNet
    from synflow.recipe_metrics import ConditionalLookup
    from synflow.recipes import precursor_vocabulary, split_by_target

    recipes = RL.load_recipes(verbose=False)
    splits = split_by_target(recipes, seed=0)
    lookup = ConditionalLookup(splits["train"], min_count=args.min_count)
    vocabulary = precursor_vocabulary(splits["train"])
    index = {p: i for i, p in enumerate(vocabulary)}

    train = build(splits["train"], lookup, index)
    val = build(splits["val"], lookup, index)
    test = build(splits["test"], lookup, index)
    print("rows: train {:,}  val {:,}  test {:,}".format(
        len(train[2]), len(val[2]), len(test[2])))

    median_log = float(train[2].median())
    print("training median {:.1f} h".format(math.exp(median_log)))
    floor = factor_scores(torch.full_like(test[2], median_log), test[2])

    model = TemperatureNet(len(vocabulary), hidden=args.hidden,
                           layers=args.layers, dropout=args.dropout,
                           mean_temperature=median_log, minimum=None)
    print("model: {:,} parameters".format(
        sum(p.numel() for p in model.parameters())))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.SmoothL1Loss()

    print("\n{:>5s} {:>9s} {:>10s} {:>10s} {:>10s}".format(
        "epoch", "loss", "within2x", "within3x", "median x"))
    print("-" * 50)
    os.makedirs(args.out, exist_ok=True)
    best = {"within_2x": -1.0}
    n = len(train[2])
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n)
        total = 0.0
        for start in range(0, n, args.batch_size):
            idx = order[start:start + args.batch_size]
            loss = criterion(model(train[0][idx], train[1][idx]),
                             train[2][idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            scores = factor_scores(model(val[0], val[1]), val[2])
        flag = ""
        if scores["within_2x"] > best["within_2x"]:
            best = dict(scores, epoch=epoch)
            flag = "  <- best"
            torch.save({"model": model.state_dict(), "vocabulary": vocabulary,
                        "hidden": args.hidden, "layers": args.layers,
                        "dropout": args.dropout,
                        "mean_temperature": median_log, "minimum": None,
                        "target": "ln_hours", "precursors": "conditional",
                        "min_count": args.min_count, "epoch": epoch,
                        "val": scores},
                       os.path.join(args.out, "best.pt"))
        if epoch % 2 == 0 or epoch == 1 or flag:
            print("{:5d} {:9.4f} {:10.1%} {:10.1%} {:9.2f}x{}".format(
                epoch, total / n, scores["within_2x"], scores["within_3x"],
                scores["median_factor"], flag))

    blob = torch.load(os.path.join(args.out, "best.pt"), weights_only=False,
                      map_location="cpu")
    model.load_state_dict(blob["model"])
    model.eval()
    with torch.no_grad():
        result = factor_scores(model(test[0], test[1]), test[2])

    print("\n" + "=" * 60)
    print("TEST -- {:,} recipes, once, epoch-{} checkpoint".format(
        len(test[2]), blob["epoch"]))
    print("{:24s} {:>9s} {:>9s} {:>10s}".format(
        "", "within2x", "within3x", "median x"))
    print("{:24s} {:9.1%} {:9.1%} {:9.2f}x".format(
        "predict the median", floor["within_2x"], floor["within_3x"],
        floor["median_factor"]))
    print("{:24s} {:9.1%} {:9.1%} {:9.2f}x".format(
        "OUR MODEL", result["within_2x"], result["within_3x"],
        result["median_factor"]))
    print("=" * 60)

    pathlib.Path(os.path.join(args.out, "result.json")).write_text(
        json.dumps({"test": result, "baseline": floor, "best_val": best,
                    "n_test": len(test[2])}, indent=2), "utf-8")


if __name__ == "__main__":
    main()
