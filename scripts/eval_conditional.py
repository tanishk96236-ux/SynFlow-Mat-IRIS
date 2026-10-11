"""Does looking at the target make the precursor rule better?

    python scripts/eval_conditional.py

Three systems, compared on the same recipes:

    rule            one precursor per metal, whatever the target
    + anion         ...chosen per (metal, is the target an oxide, a sulfide,
                    an intermetallic...)
    + anion + state ...and per oxidation state of that metal in the target

The ablation is the point. If "+ anion" carries all the gain, the story is
"oxides and intermetallics need different precursors" -- obvious once said.
If "+ state" adds on top, the story is "the rule ignores redox", which is the
Fe(2+)/Fe(3+) result and the more interesting one.

TWO SPLITS, KEPT APART
----------------------
min_count -- how often a key must have been seen before it is trusted -- is
chosen on the LLM corpus's VALIDATION split. The unseen set (Kononova targets
that appear nowhere in the LLM corpus) is scored once, at the end, with that
choice fixed. Choosing min_count on the unseen set would make it seen.
"""

import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def per_metal_hit_rate(model, recipes):
    """For each metal with exactly one supplying precursor in the recorded
    recipe: is that the one the model picks for it?"""
    from synflow.recipes import composition

    ubiquitous = model.UBIQUITOUS
    hits = total = 0
    for recipe in recipes:
        predicted = model.predict(recipe.target)
        for metal in frozenset(composition(recipe.target)) - ubiquitous:
            used = [p for p in recipe.precursors
                    if metal in frozenset(composition(p)) - ubiquitous]
            if len(used) != 1:
                continue
            total += 1
            mine = [p for p in predicted
                    if metal in frozenset(composition(p)) - ubiquitous]
            hits += bool(mine) and mine[0] == used[0]
    return hits / max(total, 1), total


def set_scores(model, recipes):
    """Jaccard and exact match against the CLOSEST recorded recipe."""
    from synflow.recipes import group_by_target

    def jaccard(a, b):
        a, b = set(a), set(b)
        union = a | b
        return len(a & b) / len(union) if union else 1.0

    groups = group_by_target(recipes)
    j = e = 0.0
    for target, group in groups.items():
        predicted = set(model.predict(target))
        recorded = [set(r.precursors) for r in group]
        j += max(jaccard(predicted, r) for r in recorded)
        e += any(predicted == r for r in recorded)
    n = len(groups)
    return j / n, e / n, n


def main():
    from synflow import recipes_llm as RL
    from synflow.recipe_metrics import ConditionalLookup, MetalLookupBaseline
    from synflow.recipes import load_recipes as load_kononova
    from synflow.recipes import split_by_target

    recipes = RL.load_recipes(verbose=False)
    splits = split_by_target(recipes, seed=0)
    train, val = splits["train"], splits["val"]
    print("LLM corpus: train {:,}  validation {:,}\n".format(
        len(train), len(val)))

    # ---------------------------------------------------------- validation
    print("VALIDATION (LLM corpus) -- choosing min_count here")
    print("{:30s} {:>9s} {:>9s} {:>9s}".format(
        "", "per-metal", "jaccard", "exact"))
    print("-" * 62)

    plain = MetalLookupBaseline(train)
    hit, n_metal = per_metal_hit_rate(plain, val)
    jac, exa, n_target = set_scores(plain, val)
    print("{:30s} {:9.1%} {:9.3f} {:9.1%}".format("rule", hit, jac, exa))

    anion_only = ConditionalLookup(train, use_state=False)
    with_state = ConditionalLookup(train, use_state=True)

    best = None
    for label, model in (("+ anion", anion_only),
                         ("+ anion + state", with_state)):
        for min_count in (3, 5, 10, 20, 50):
            model.min_count = min_count
            hit, _ = per_metal_hit_rate(model, val)
            jac, exa, _ = set_scores(model, val)
            print("{:30s} {:9.1%} {:9.3f} {:9.1%}".format(
                "{}  (min_count {})".format(label, min_count), hit, jac, exa))
            if label == "+ anion + state" and (best is None or jac > best[1]):
                best = (min_count, jac)
    print("-" * 62)
    print("({:,} metal assignments, {:,} targets)".format(n_metal, n_target))
    chosen = best[0]
    print("\nmin_count chosen on validation: {}".format(chosen))

    # ---------------------------------------------------------------- unseen
    # Fitted on the WHOLE LLM corpus, as eval_unseen.py does: no unseen
    # target appears anywhere in it, so nothing leaks.
    from scripts.eval_unseen import unseen_test_set     # noqa: E402

    unseen = unseen_test_set(recipes, load_kononova(verbose=False))
    print("\nUNSEEN (Kononova targets absent from the LLM corpus) -- scored once")
    print("{:30s} {:>9s} {:>9s}".format("", "jaccard", "exact"))
    print("-" * 52)
    for label, model in (
            ("rule", MetalLookupBaseline(recipes)),
            ("+ anion", ConditionalLookup(recipes, min_count=chosen,
                                          use_state=False)),
            ("+ anion + state", ConditionalLookup(recipes, min_count=chosen,
                                                  use_state=True))):
        jac, exa, n = set_scores(model, unseen)
        print("{:30s} {:9.3f} {:9.1%}".format(label, jac, exa))
    print("-" * 52)
    print("({:,} unseen targets)".format(n))


if __name__ == "__main__":
    main()
