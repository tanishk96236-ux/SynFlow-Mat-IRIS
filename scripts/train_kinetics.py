"""Train the kinetic gate on the Precursor Genome; test it on the A-Lab.

    python scripts/train_kinetics.py

TRAIN  Precursor Genome (Walters et al. 2026, CC BY 4.0): 1,035 pairwise
       reactions by the A-Lab, each heated at its Tammann temperature for 1 or
       4 h and classified by XRD. Label: did a MIXED phase form (partially or
       completely reacted) or not (unreacted, or only transformed on its own).
       Only pairs downhill by more than 5 meV/atom -- the kinetic question is
       asked only where thermodynamics allows a reaction.

TEST   A-Lab Supplementary Table 2 (Szymanski et al. 2023): 75 pairwise rows,
       65 where a product formed and 10 where none did. Every Precursor
       Genome pair that also appears in Table 2 is REMOVED from training, so
       no test pair has been seen, at any temperature.

THE THRESHOLD is chosen on out-of-fold training predictions (the one that
maximises balanced accuracy), before Table 2 is looked at, and not changed
afterwards.
"""

import json
import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

REACTED = ("partially_reacted", "completely_reacted")
NOT_REACTED = ("unreacted", "transformed")
DOWNHILL_EV = -0.005
TABLE2_DWELL_MIN = 240.0     # the A-Lab's standard 4 h hold


def reduced(formula):
    from pymatgen.core import Composition
    return Composition(formula).reduced_formula


def load_genome(path):
    with open(path, encoding="utf-8") as handle:
        ledger = json.load(handle)
    rows = []
    for sample in ledger["samples"]:
        category = (sample.get("outcome") or {}).get("reaction_category")
        synthesis = sample.get("synthesis") or {}
        if category not in REACTED + NOT_REACTED:
            continue
        if synthesis.get("heating_temperature") is None \
                or synthesis.get("heating_time") is None:
            continue
        a, b = (p["formula"] for p in sample["precursors"])
        rows.append({"a": a, "b": b,
                     "pair": frozenset((reduced(a), reduced(b))),
                     "T": synthesis["heating_temperature"],
                     "t": synthesis["heating_time"],
                     "E": sample.get("reaction_energy_ev_per_atom"),
                     "y": int(category in REACTED),
                     "category": category})
    return rows


def matrix(rows):
    import numpy as np
    from synflow import kinetics

    X, y, kept = [], [], []
    for row in rows:
        f = kinetics.features(row["a"], row["b"], row["T"], row["t"])
        if f is None:
            continue
        X.append([f[name] for name in kinetics.FEATURES])
        y.append(row["y"])
        kept.append(row)
    return np.array(X), np.array(y), kept


def balanced_accuracy(y, predicted):
    tp = sum(1 for a, b in zip(y, predicted) if a and b)
    tn = sum(1 for a, b in zip(y, predicted) if not a and not b)
    pos, neg = sum(y), len(y) - sum(y)
    return 0.5 * (tp / pos + tn / neg) if pos and neg else float("nan")


def main():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    from eval_alab import parse_table2
    from synflow import kinetics

    genome = load_genome(kinetics.LEDGER)
    print("Precursor Genome: {} classified rows ({} reacted)".format(
        len(genome), sum(r["y"] for r in genome)))
    downhill = [r for r in genome if r["E"] is not None and r["E"] < DOWNHILL_EV]
    print("downhill by > 5 meV/atom: {} ({} reacted, {:.1%})".format(
        len(downhill), sum(r["y"] for r in downhill),
        sum(r["y"] for r in downhill) / len(downhill)))

    table2 = [r for r in parse_table2() if len(r["reactants"]) == 2]
    test_pairs = {frozenset(reduced(x) for x in r["reactants"])
                  for r in table2}
    train = [r for r in downhill if r["pair"] not in test_pairs]
    print("removed {} Precursor Genome rows whose pair is in Table 2; "
          "training on {}".format(len(downhill) - len(train), len(train)))

    X, y, kept = matrix(train)
    print("with every feature known: {} rows, {} reacted\n".format(
        len(y), int(y.sum())))

    def model():
        return LogisticRegression(C=1.0, max_iter=1000)

    # ------------------------------------------------- cross-validation
    oof = np.zeros(len(y))
    for fit, held in StratifiedKFold(10, shuffle=True,
                                     random_state=0).split(X, y):
        scaler = StandardScaler().fit(X[fit])
        clf = model().fit(scaler.transform(X[fit]), y[fit])
        oof[held] = clf.predict_proba(scaler.transform(X[held]))[:, 1]

    # leave one PRECURSOR out: every row containing it is held out together
    precursors = sorted({reduced(r["a"]) for r in kept}
                        | {reduced(r["b"]) for r in kept})
    lopo = np.full(len(y), np.nan)
    for held_out in precursors:
        held = np.array([held_out in r["pair"] for r in kept])
        if held.all() or not held.any() or len(set(y[~held])) < 2:
            continue
        scaler = StandardScaler().fit(X[~held])
        clf = model().fit(scaler.transform(X[~held]), y[~held])
        p = clf.predict_proba(scaler.transform(X[held]))[:, 1]
        # a row has two precursors; keep the stricter (first) hold-out score
        idx = np.where(held)[0]
        for i, value in zip(idx, p):
            if np.isnan(lopo[i]):
                lopo[i] = value

    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [balanced_accuracy(y, oof >= t) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(scores))])

    h_min = X[:, kinetics.FEATURES.index("h_min")]
    print("PRECURSOR GENOME, out-of-fold (n={})".format(len(y)))
    print("   {:44s} {:>6s} {:>10s}".format("", "AUC", "bal. acc"))
    print("   {:44s} {:6.3f} {:10.1%}".format(
        "logistic, 10-fold", roc_auc_score(y, oof),
        balanced_accuracy(y, oof >= threshold)))
    ok = ~np.isnan(lopo)
    print("   {:44s} {:6.3f} {:10.1%}".format(
        "logistic, leave one precursor out (n={})".format(int(ok.sum())),
        roc_auc_score(y[ok], lopo[ok]),
        balanced_accuracy(y[ok], lopo[ok] >= threshold)))
    print("   {:44s} {:6.3f} {:10.1%}".format(
        "Tammann alone (h_min >= 2/3)", roc_auc_score(y, h_min),
        balanced_accuracy(y, h_min >= 2 / 3)))
    print("   {:44s} {:6.3f} {:10.1%}".format(
        "always 'no reaction'", 0.5, 0.5))
    print("   threshold (max balanced accuracy, chosen here): {:.2f}".format(
        threshold))

    scaler = StandardScaler().fit(X)
    clf = model().fit(scaler.transform(X), y)
    blob = {
        "features": list(kinetics.FEATURES),
        "mean": dict(zip(kinetics.FEATURES, scaler.mean_.tolist())),
        "scale": dict(zip(kinetics.FEATURES, scaler.scale_.tolist())),
        "coef": dict(zip(kinetics.FEATURES, clf.coef_[0].tolist())),
        "intercept": float(clf.intercept_[0]),
        "threshold": threshold,
        "trained_on": "Precursor Genome, {} downhill pairs, Table 2 pairs "
                      "removed".format(len(y)),
        "cv_auc": float(roc_auc_score(y, oof)),
        "lopo_auc": float(roc_auc_score(y[ok], lopo[ok])),
    }
    pathlib.Path(kinetics.MODEL).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(kinetics.MODEL).write_text(json.dumps(blob, indent=2),
                                            "utf-8")
    kinetics._model.cache_clear()
    print("\ncoefficients (per standard deviation of each feature):")
    for name in kinetics.FEATURES:
        print("   {:8s} {:+.3f}".format(name, blob["coef"][name]))

    test_on_table2(blob)


def test_on_table2(blob):
    """Every checked Table 2 row, under five gates.

    A kinetic gate can only veto: it turns "the thermodynamics says this
    forms" into "it will not form here". Where it has nothing to say -- no
    melting point, no observation, or a single precursor heated in air --
    the thermodynamic verdict stands.
    """
    from eval_alab import parse_table2
    from synflow import kinetics

    checkpoint = "runs/alab_table2_measured.jsonl"
    results = {}
    with open(checkpoint, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entry = json.loads(line)
                results[entry["key"]] = entry

    rows = parse_table2()
    formed = [r for r in rows if not r["inert"]]
    inert = [r for r in rows if r["inert"]]
    keyed = [("formed:{}".format(i), r) for i, r in enumerate(formed, 1)] + \
            [("inert:{}".format(i), r) for i, r in enumerate(inert, 1)]

    def first_version(a, b, temperature):
        """The evidence rule as first written: "transformed" counted as no
        reaction. Kept only to report what the change did."""
        from pymatgen.core import Composition
        pair = frozenset(Composition(f).reduced_formula for f in (a, b))
        for seen_T, seen_t, category in kinetics._observations().get(pair, []):
            margin = (0.0 if seen_t >= TABLE2_DWELL_MIN
                      else kinetics.TIME_MARGIN_K)
            if category not in kinetics.REACTED \
                    and seen_T >= temperature + margin:
                return True
        return None

    gates = ("thermodynamics only",
             "+ learned model (PG)",
             "+ Tammann (2/3 Tm, K)",
             "+ robot evidence, v1",
             "+ robot evidence")
    tally = {}
    print("\nA-LAB TABLE 2 -- every checked row; pairs held out of training; "
          "4 h hold assumed")
    for key, row in keyed:
        result = results.get(key)
        if result is None or result["outcome"] == "unchecked":
            continue
        thermo = result["outcome"].startswith("agrees")
        blocks = {gate: None for gate in gates}
        if len(row["reactants"]) == 2:
            a, b = row["reactants"]
            T = row["upper_C"]
            p = kinetics.probability(a, b, T, TABLE2_DWELL_MIN, blob)
            x = kinetics.features(a, b, T, TABLE2_DWELL_MIN)
            found = kinetics.evidence(a, b, T, TABLE2_DWELL_MIN)
            blocks["+ learned model (PG)"] = (None if p is None
                                              else p < blob["threshold"])
            blocks["+ Tammann (2/3 Tm, K)"] = (None if x is None
                                               else x["h_min"] < 2 / 3)
            blocks["+ robot evidence, v1"] = first_version(a, b, T)
            blocks["+ robot evidence"] = (
                None if found is None else found["verdict"] == "blocked")
        group = "inert" if row["inert"] else "formed"
        for gate in gates:
            block = blocks[gate]
            if block is None:
                agrees = thermo
            elif row["inert"]:
                agrees = thermo or block
            else:
                agrees = thermo and not block
            hit, n = tally.get((gate, group), (0, 0))
            tally[(gate, group)] = (hit + bool(agrees), n + 1)

    print("   {:26s} {:>16s} {:>16s} {:>10s}".format(
        "gate", "formed", "inert", "balanced"))
    for gate in gates:
        f_hit, f_n = tally[(gate, "formed")]
        i_hit, i_n = tally[(gate, "inert")]
        print("   {:26s} {:>9s} {:5.1%} {:>9s} {:5.1%} {:9.1%}".format(
            gate, "{}/{}".format(f_hit, f_n), f_hit / f_n,
            "{}/{}".format(i_hit, i_n), i_hit / i_n,
            0.5 * (f_hit / f_n + i_hit / i_n)))
    print("   v1 counted 'transformed' as no reaction; see kinetics.evidence "
          "for why that\n   was wrong (MgO + NiO).")


if __name__ == "__main__":
    main()
