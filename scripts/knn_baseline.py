"""Temperature and dwell time from the most similar known compounds.

    python scripts/knn_baseline.py

The network compresses 33,620 recipes into 445,185 weights and loses detail
on the way; its validation stops improving after about 20 epochs while the
training loss keeps falling, which is the network memorising rather than
learning. Something that does not compress at all is the opposite bet: for a
new compound, look up the k training compounds closest in composition and
take the median of what they were fired at.

That is what a chemist does first -- "what did people use for things like
this?" -- and it is the idea behind the similarity-based recommender in Ceder
et al. 2023 (82% top-5 on precursors).

The oracle ceilings it is compared against, measured on the same splits:

    temperature MAE   79.2 C (119.8 C on multi-recipe targets)
    dwell within 2x   80.7%
"""

import math
import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch                                                    # noqa: E402


def main():
    from synflow import recipes_llm as RL
    from synflow.dao_s import composition_tensor
    from synflow.recipes import split_by_target

    from synflow.dao_s import load_shipped

    recipes = RL.load_recipes(verbose=False)
    splits = split_by_target(recipes, seed=0)
    model = load_shipped(verbose=False)

    def table(group, field):
        rows = [(r.target, getattr(r, field) if field != "temperature"
                 else r.max_temperature) for r in group]
        rows = [(t, v) for t, v in rows if v]
        x = torch.stack([composition_tensor(t) for t, _ in rows])
        y = torch.tensor([v for _, v in rows], dtype=torch.float32)
        return x, y

    def predict(x_query, x_train, y_train, k):
        preds = []
        for start in range(0, len(x_query), 512):
            d = torch.cdist(x_query[start:start + 512], x_train)
            idx = d.topk(k, largest=False).indices
            preds.append(y_train[idx].median(dim=1).values)
        return torch.cat(preds)

    def scores(field, p, y):
        if field == "dwell_hours":
            factor = torch.exp((p - y).abs())
            return {"within2": float((factor <= 2).float().mean()),
                    "within3": float((factor <= 3).float().mean()),
                    "median_x": float(factor.median())}
        err = (p - y).abs()
        return {"mae": float(err.mean()), "median": float(err.median()),
                "within100": float((err <= 100).float().mean())}

    def line(field, s):
        if field == "dwell_hours":
            return ("within 2x {:5.1%}   within 3x {:5.1%}   median {:.2f}x"
                    .format(s["within2"], s["within3"], s["median_x"]))
        return ("MAE {:6.1f} C   median {:6.1f} C   within 100 C {:5.1%}"
                .format(s["mae"], s["median"], s["within100"]))

    # k is chosen on VALIDATION and the test split is touched once, with
    # that k -- the rule the network was held to. Printing every k on test
    # and quoting the best would be choosing on the test set.
    for field, label in (("temperature", "TEMPERATURE"),
                         ("dwell_hours", "DWELL")):
        x_tr, y_tr = table(splits["train"], field)
        x_va, y_va = table(splits["val"], field)
        x_te, y_te = table(splits["test"], field)
        if field == "dwell_hours":
            y_tr, y_va, y_te = y_tr.log(), y_va.log(), y_te.log()

        print("{} -- {:,} training, {:,} validation, {:,} test".format(
            label, len(y_tr), len(y_va), len(y_te)))
        best_k, best = None, None
        for k in (1, 3, 5, 10, 25, 50):
            s = scores(field, predict(x_va, x_tr, y_tr, k), y_va)
            print("   validation k={:3d}   {}".format(k, line(field, s)))
            better = (s["within2"] > best["within2"] if field == "dwell_hours"
                      else s["mae"] < best["mae"]) if best else True
            if better:
                best_k, best = k, s
        # THE HYBRID IS DECIDED HERE, ON VALIDATION. The rule, stated before
        # it was computed: ship the average of the network and the k-NN only
        # if it lowers validation MAE by at least 5 C (temperature) or raises
        # within-2x by at least a point (dwell). Its test number below was
        # seen before this rule existed, so a hybrid that passes is confirmed
        # on the unseen set (scripts/eval_unseen.py), not on test.
        def network_on(group):
            values = []
            for r in group:
                truth = (r.max_temperature if field == "temperature"
                         else r.dwell_hours)
                if not truth:
                    continue
                precursors = model.predict(r.target)
                value = (model.predict_temperature(r.target, precursors)
                         if field == "temperature"
                         else model.predict_dwell(r.target, precursors))
                values.append(float("nan") if value is None else value)
            values = torch.tensor(values, dtype=torch.float32)
            return values.log() if field == "dwell_hours" else values

        net_va = network_on(splits["val"])
        knn_va = predict(x_va, x_tr, y_tr, best_k)
        ok_va = ~torch.isnan(net_va)
        s_net = scores(field, net_va[ok_va], y_va[ok_va])
        s_avg = scores(field, (net_va[ok_va] + knn_va[ok_va]) / 2,
                       y_va[ok_va])
        print("   VALIDATION, network               {}".format(
            line(field, s_net)))
        print("   VALIDATION, average               {}".format(
            line(field, s_avg)))
        passes = (s_avg["within2"] >= s_net["within2"] + 0.01
                  if field == "dwell_hours"
                  else s_avg["mae"] <= s_net["mae"] - 5.0)
        print("   hybrid {} the rule".format("PASSES" if passes else
                                             "does not pass"))

        knn = predict(x_te, x_tr, y_tr, best_k)
        test = scores(field, knn, y_te)
        print("   TEST, k={} (chosen on validation)   {}".format(
            best_k, line(field, test)))

        # The network on the SAME test rows. Its own training scripts
        # filtered rows slightly differently (2,527 dwell rows there against
        # 2,652 here), so their numbers are not quoted beside these. Shipped
        # behaviour: precursors from the lookup, dwell rounded to
        # conventional values.
        targets = [r.target for r in splits["test"]
                   if (r.max_temperature if field == "temperature"
                       else r.dwell_hours)]
        net = []
        for target in targets:
            precursors = model.predict(target)
            value = (model.predict_temperature(target, precursors)
                     if field == "temperature"
                     else model.predict_dwell(target, precursors))
            net.append(float("nan") if value is None else value)
        net = torch.tensor(net, dtype=torch.float32)
        if field == "dwell_hours":
            net = net.log()
        ok = ~torch.isnan(net)
        print("   TEST, network, same rows (n={})   {}".format(
            int(ok.sum()), line(field, scores(field, net[ok], y_te[ok]))))
        print("   TEST, k={}, those same n rows       {}".format(
            best_k, line(field, scores(field, knn[ok], y_te[ok]))))
        both = (knn[ok] + net[ok]) / 2
        print("   TEST, average of the two          {}".format(
            line(field, scores(field, both, y_te[ok]))))
        print()


if __name__ == "__main__":
    main()
