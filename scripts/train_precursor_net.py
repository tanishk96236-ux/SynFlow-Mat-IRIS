"""Train the learned precursor picker and compare it with the lookup.

    python scripts/train_precursor_net.py --epochs 25

THE QUESTION
------------
"Precursor choice is largely not a learning problem" was concluded on 6,128
training recipes. This re-tests it on 36,787, against the strongest
non-learned system available, ConditionalLookup:

    unseen   ConditionalLookup   0.724 Jaccard   55.7% exact

Either outcome is reportable. A win means the larger corpus made precursor
choice learnable after all; a tie means the lookup is close to what this
data supports, and the effort belongs elsewhere.

TRAINING EXAMPLES
-----------------
One per (recipe, metal) where exactly one recorded precursor supplies that
metal -- the same filter the per-metal hit rate uses, so the network is
trained on precisely the question it is scored on. Labels outside the
vocabulary (precursors seen fewer than --min-precursor-count times in
training) are dropped rather than mapped to a catch-all.

SELECTION AND TESTING
---------------------
The checkpoint is chosen on validation set-Jaccard, the number that gets
reported. The unseen set is scored once, at the end, and the lookup is scored
on it in the same run so the two numbers cannot drift apart.
"""

import argparse
import collections
import json
import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402


def examples(recipes, vocabulary):
    """(features, label index, metal) for every usable (recipe, metal)."""
    from synflow.precursor_net import featurize
    from synflow.recipe_metrics import MetalLookupBaseline
    from synflow.recipes import composition

    ubiquitous = MetalLookupBaseline.UBIQUITOUS
    index = {p: i for i, p in enumerate(vocabulary)}
    features, labels, metals = [], [], []
    for recipe in recipes:
        for metal in frozenset(composition(recipe.target)) - ubiquitous:
            used = [p for p in recipe.precursors
                    if metal in frozenset(composition(p)) - ubiquitous]
            if len(used) != 1 or used[0] not in index:
                continue
            row = featurize(recipe.target, metal)
            if row is None:
                continue
            features.append(row)
            labels.append(index[used[0]])
            metals.append(metal)
    return torch.stack(features), torch.tensor(labels), metals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-precursor-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/precursor_net")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    from synflow import recipes_llm as RL
    from synflow.precursor_net import (NeuralPrecursorPicker, PrecursorNet,
                                       supply_masks)
    from synflow.recipe_metrics import ConditionalLookup
    from synflow.recipes import load_recipes as load_kononova
    from synflow.recipes import split_by_target
    from scripts.eval_conditional import per_metal_hit_rate, set_scores
    from scripts.eval_unseen import unseen_test_set

    recipes = RL.load_recipes(verbose=False)
    splits = split_by_target(recipes, seed=0)
    train, val = splits["train"], splits["val"]

    counts = collections.Counter(p for r in train for p in r.precursors)
    vocabulary = sorted(p for p, n in counts.items()
                        if n >= args.min_precursor_count)
    print("vocabulary: {:,} precursors seen >= {} times in training".format(
        len(vocabulary), args.min_precursor_count))

    print("featurising (oxidation states for every training target) ...")
    x_train, y_train, metals_train = examples(train, vocabulary)
    print("training examples: {:,} (recipe, metal) pairs".format(len(y_train)))

    masks = supply_masks(vocabulary)
    mask_train = torch.stack([masks[m] for m in metals_train])

    lookup = ConditionalLookup(train, min_count=3)
    jac_l, exa_l, n_val = set_scores(lookup, val)
    hit_l, _ = per_metal_hit_rate(lookup, val)
    print("\nBAR on validation -- ConditionalLookup: per-metal {:.1%}  "
          "jaccard {:.3f}  exact {:.1%}  ({:,} targets)\n".format(
              hit_l, jac_l, exa_l, n_val))

    network = PrecursorNet(len(vocabulary), hidden=args.hidden,
                           dropout=args.dropout)
    print("network: {:,} parameters".format(
        sum(p.numel() for p in network.parameters())))
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr,
                                  weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    print("\n{:>5s} {:>9s} {:>10s} {:>9s} {:>9s}".format(
        "epoch", "loss", "per-metal", "jaccard", "exact"))
    print("-" * 48)

    os.makedirs(args.out, exist_ok=True)
    best = {"jaccard": -1.0}
    history = []
    n = len(y_train)
    for epoch in range(1, args.epochs + 1):
        network.train()
        order = torch.randperm(n)
        total = 0.0
        for start in range(0, n, args.batch_size):
            idx = order[start:start + args.batch_size]
            logits = network(x_train[idx])
            # Masked before the loss as well as at inference: the network is
            # never asked to rank a precursor that cannot supply the metal,
            # so no capacity is spent learning that TiO2 does not supply Ba.
            logits = logits.masked_fill(~mask_train[idx], float("-inf"))
            loss = nn.functional.cross_entropy(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        scheduler.step()

        picker = NeuralPrecursorPicker(network, vocabulary, fallback=lookup)
        hit, _ = per_metal_hit_rate(picker, val)
        jac, exa, _ = set_scores(picker, val)
        history.append({"epoch": epoch, "loss": total / n,
                        "per_metal": hit, "jaccard": jac, "exact": exa})
        flag = ""
        if jac > best["jaccard"]:
            best = {"epoch": epoch, "jaccard": jac, "exact": exa,
                    "per_metal": hit}
            flag = "  <- best"
            torch.save({"model": network.state_dict(),
                        "vocabulary": vocabulary, "hidden": args.hidden,
                        "dropout": args.dropout, "epoch": epoch,
                        "val": best}, os.path.join(args.out, "best.pt"))
        print("{:5d} {:9.4f} {:10.1%} {:9.3f} {:9.1%}{}".format(
            epoch, total / n, hit, jac, exa, flag))

    # ---------------------------------------------------------------- unseen
    blob = torch.load(os.path.join(args.out, "best.pt"), weights_only=False,
                      map_location="cpu")
    network.load_state_dict(blob["model"])

    # Both fitted on the training split for this comparison, so the network
    # and the lookup see exactly the same evidence. Unseen targets appear in
    # neither.
    unseen = unseen_test_set(recipes, load_kononova(verbose=False))
    picker = NeuralPrecursorPicker(network, vocabulary, fallback=lookup)
    jac_n, exa_n, n_u = set_scores(picker, unseen)
    jac_lu, exa_lu, _ = set_scores(lookup, unseen)

    print("\n" + "=" * 60)
    print("VALIDATION (chose the epoch)   best epoch {}".format(best["epoch"]))
    print("   lookup    jaccard {:.3f}   exact {:.1%}".format(jac_l, exa_l))
    print("   network   jaccard {:.3f}   exact {:.1%}".format(
        best["jaccard"], best["exact"]))
    print("UNSEEN ({:,} targets), scored once".format(n_u))
    print("   lookup    jaccard {:.3f}   exact {:.1%}".format(jac_lu, exa_lu))
    print("   network   jaccard {:.3f}   exact {:.1%}".format(jac_n, exa_n))
    print("=" * 60)

    pathlib.Path(os.path.join(args.out, "result.json")).write_text(
        json.dumps({"history": history, "best": best,
                    "validation_lookup": {"jaccard": jac_l, "exact": exa_l},
                    "unseen_lookup": {"jaccard": jac_lu, "exact": exa_lu},
                    "unseen_network": {"jaccard": jac_n, "exact": exa_n},
                    "n_unseen": n_u, "n_train_examples": n,
                    "vocabulary": len(vocabulary)}, indent=2), "utf-8")


if __name__ == "__main__":
    main()
