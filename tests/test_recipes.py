"""DAO-S data handling and metrics.

Two measured properties of the synthesis data drive every design choice here,
and getting either wrong produces a number that looks fine and means nothing:

* 19,488 recipes cover 13,009 targets. LiFePO4 has 118 of them. A random split
  scatters those across train and test and the model scores by recall.
* Of the targets with several recipes, only 38% agree on the precursors. Exact
  match would mark a genuine alternative route wrong.
"""

import pytest

from synflow.recipe_metrics import (
    MostCommonBaseline,
    RetrievalBaseline,
    best_over_references,
    composition_similarity,
    evaluate,
    jaccard,
    precision_recall_f1,
)
from synflow.recipes import (
    Recipe,
    composition,
    group_by_target,
    precursor_vocabulary,
    split_by_target,
)


def _r(target, precursors, temps=(), ops=()):
    return Recipe(target, precursors, ops, temps)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def test_composition_parses_a_clean_formula():
    assert composition("Fe2O3") == {"Fe": 2.0, "O": 3.0}
    assert composition("LiFePO4") == {"Li": 1.0, "Fe": 1.0, "P": 1.0, "O": 4.0}


def test_composition_rejects_what_names_no_single_compound():
    """56% of raw targets are like this -- free variables, composites, trade
    names -- and a model cannot be conditioned on them."""
    for bad in ("Sr2Ir1-xRuxO4", "(Ta2O5)1-x(TiO2)x", "YAG",
                "B2O3-Bi2O3-SiO2", ""):
        assert composition(bad) == {}, bad


def test_max_temperature_is_the_firing_step():
    """The hottest step forms the phase; drying and calcination are lower and
    are not what a chemist means by 'the temperature'."""
    assert _r("X", ["A"], temps=(120.0, 600.0, 1200.0)).max_temperature == 1200.0
    assert _r("X", ["A"]).max_temperature is None


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------

def test_every_recipe_for_a_target_stays_together():
    recipes = [_r("T{}".format(t), ["P{}".format(i)])
               for t in range(50) for i in range(3)]
    splits = split_by_target(recipes, seed=0)
    seen = [{r.target for r in splits[s]} for s in ("train", "val", "test")]
    assert seen[0] & seen[1] == set()
    assert seen[0] & seen[2] == set()
    assert seen[1] & seen[2] == set()


def test_split_keeps_every_recipe():
    recipes = [_r("T{}".format(t), ["P"]) for t in range(40) for _ in range(2)]
    splits = split_by_target(recipes, seed=0)
    assert sum(len(v) for v in splits.values()) == len(recipes)


def test_heavily_repeated_targets_do_not_leak():
    """LiFePO4 in miniature: one target with many recipes must not straddle."""
    recipes = ([_r("LiFePO4", ["Li2CO3", "FeC2O4"]) for _ in range(30)] +
               [_r("T{}".format(i), ["X"]) for i in range(60)])
    splits = split_by_target(recipes, seed=0)
    holding = [s for s in splits if any(r.target == "LiFePO4" for r in splits[s])]
    assert len(holding) == 1


# --------------------------------------------------------------------------
# metrics -- one target, many valid answers
# --------------------------------------------------------------------------

def test_jaccard_is_set_overlap():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"a", "c"}) == pytest.approx(1 / 3)
    assert jaccard({"a"}, {"b"}) == 0.0


def test_a_valid_alternative_route_is_not_marked_wrong():
    """The reason exact match is not the headline metric. 62% of repeated
    targets disagree on precursors, all of them real syntheses."""
    references = [{"Li2CO3", "FeC2O4"}, {"LiOH", "Fe2O3"}]
    assert best_over_references({"LiOH", "Fe2O3"}, references) == 1.0
    assert best_over_references({"Li2CO3", "FeC2O4"}, references) == 1.0


def test_evaluate_credits_the_closest_reference():
    refs = {"T": [{"a", "b"}, {"c", "d"}]}
    perfect = evaluate({"T": {"c", "d"}}, refs)
    assert perfect["jaccard"] == 1.0
    assert perfect["exact_match"] == 1.0


def test_precision_and_recall_separate_over_and_under_prediction():
    p, r, _ = precision_recall_f1({"a", "b", "c"}, {"a"})
    assert p == pytest.approx(1 / 3) and r == 1.0      # predicted too much
    p, r, _ = precision_recall_f1({"a"}, {"a", "b", "c"})
    assert p == 1.0 and r == pytest.approx(1 / 3)      # predicted too little


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def test_similar_compositions_score_higher():
    assert (composition_similarity("LiFePO4", "LiMnPO4") >
            composition_similarity("LiFePO4", "BaTiO3"))


def test_retrieval_uses_the_nearest_target_not_the_same_one():
    train = [_r("LiMnPO4", ["Li2CO3", "MnO"]), _r("BaTiO3", ["BaCO3", "TiO2"])]
    baseline = RetrievalBaseline(train)
    # a lithium phosphate should pull the lithium phosphate, not the titanate
    assert baseline.predict("LiFePO4") == frozenset({"Li2CO3", "MnO"})


def test_most_common_ignores_the_target_entirely():
    """The true floor: no chemistry at all."""
    train = ([_r("A", ["X", "Y"]) for _ in range(5)] + [_r("B", ["Z"])])
    baseline = MostCommonBaseline(train)
    assert baseline.predict("anything") == frozenset({"X", "Y"})
    assert baseline.predict("LiFePO4") == baseline.predict("BaTiO3")


def test_vocabulary_is_ordered_by_frequency():
    recipes = [_r("A", ["common"]), _r("B", ["common"]), _r("C", ["rare"])]
    assert precursor_vocabulary(recipes)[0] == "common"
    assert precursor_vocabulary(recipes, min_count=2) == ["common"]


def test_group_by_target_collects_alternatives():
    recipes = [_r("T", ["a"]), _r("T", ["b"]), _r("U", ["c"])]
    groups = group_by_target(recipes)
    assert len(groups["T"]) == 2 and len(groups["U"]) == 1


# --------------------------------------------------------------------------
# parenthesised formulas
# --------------------------------------------------------------------------

def test_pymatgen_style_parentheses_parse():
    """pymatgen writes Composition("BaSb2O4").reduced_formula as "Ba(SbO2)2",
    and the discovery loop hands those strings straight to DAO-S. Returning
    {} for them made the metal lookup find no metals and the network fall
    back to its prior, so thirty spinels came back with one identical recipe
    containing neither barium nor antimony."""
    from synflow.recipes import composition

    assert composition("Ba(SbO2)2") == composition("BaSb2O4")
    assert composition("Ba(SbO2)2") == {"Ba": 1.0, "Sb": 2.0, "O": 4.0}
    assert composition("Cs(WO2)2") == {"Cs": 1.0, "W": 2.0, "O": 4.0}
    assert composition("K(MoO2)2") == {"K": 1.0, "Mo": 2.0, "O": 4.0}


def test_a_group_without_a_multiplier_is_still_expanded():
    from synflow.recipes import composition

    assert composition("Ca(OH)2") == {"Ca": 1.0, "O": 2.0, "H": 2.0}


def test_expansion_does_not_disturb_plain_formulas():
    """Every recipe target in the corpus goes through this path, so the
    common case must come out untouched."""
    from synflow.recipes import composition

    for formula in ("LiFePO4", "BaTiO3", "LiMn2O4", "Fe2O3", "O2"):
        assert composition(formula), formula
    assert composition("LiFePO4") == {"Li": 1.0, "Fe": 1.0, "P": 1.0, "O": 4.0}


def test_site_labels_and_trade_names_are_still_refused():
    """The reason the parser is hand-rolled rather than pymatgen's: it has to
    reject formulas that only look like formulas."""
    from synflow.recipes import composition

    assert composition("YAG") == {}
    assert composition("A0.5Ti0.5NbO4") == {}
    assert composition("Sr2Ir1-xRuxO4") == {}
    assert composition("") == {}


def test_a_malformed_group_does_not_hang():
    from synflow.recipes import composition

    assert composition("Ba((SbO2)") == {}


def test_a_formula_template_is_refused_rather_than_partly_read():
    """The expensive one. These are formulas copied from paper titles with the
    doping variable left unresolved, and the corpus is full of them: 82 of the
    563 targets in the test split, 14.6%.

    `_expand_groups` used to keep whatever it could match inside a group and
    DROP the rest, which turned an unparseable string into a plausible wrong
    one. "(Ba1-xCax)" became "Ba1Ca1": "1-x" was read as 1 and "x" vanished,
    so an A site holding one atom between the two elements came out holding
    two. The string then passed the clean-formula check, because by that point
    nothing suspicious was left in it, and went on to become a DAO-S input
    vector and a reaction to balance.

    Nothing crashed. The composition was simply wrong, and wrong in a way that
    looks like a real doped perovskite.
    """
    from synflow.recipes import composition

    for formula in ("(Ba1-xCax)(Ti0.85Sn0.15)O3",
                    "(Ba1-xEux)2MgSi2O7",
                    "((Mg1-x-yCax)Lay)(Ti1-yAly)O3",
                    "(Bi1-xPrx)(Fe1-xTix)O3",
                    "Ba(1-x)Nd(2x/3)ZrO3",
                    "LaxSr1-xMnO3"):
        assert composition(formula) == {}, formula


def test_the_specific_corruption_that_was_happening():
    """Named separately because the old behaviour is worth pinning down: it is
    not that these came out empty, it is that they came out looking fine."""
    from synflow.recipes import composition

    # what the old parser produced, which must never come back
    assert composition("(Ba1-xCax)(Ti0.85Sn0.15)O3") != {
        "Ba": 1.0, "Ca": 1.0, "Ti": 0.85, "Sn": 0.15, "O": 3.0}
    assert composition("(Ba1-xEux)2MgSi2O7") != {
        "Ba": 2.0, "Eu": 2.0, "Mg": 1.0, "Si": 2.0, "O": 7.0}


def test_real_doped_compositions_still_parse():
    """The refusal must be aimed at the free variable, not at decimals. Real
    solid solutions are written with explicit numbers and are perfectly
    determinate; refusing those would throw away most of the corpus."""
    from synflow.recipes import composition

    assert composition("(Bi0.5Na0.5)0.93Ba0.07TiO3") == {
        "Bi": 0.465, "Na": 0.465, "Ba": 0.07, "Ti": 1.0, "O": 3.0}
    for formula in ("(Ba0.85Ca0.15)(Zr0.10Ti0.90)O3",
                    "Ba0.9Sr0.1Ti1O3",
                    "(Ba0.7Bi0.3)(Ti0.7Sc0.3)O3"):
        assert composition(formula), formula


def test_the_rule_never_proposes_the_target_as_its_own_precursor():
    """Asked for Al2O3 the lookup returned {Al2O3}, because Al2O3 really is the
    commonest source of aluminium. True, and useless: "make Al2O3 from Al2O3"
    is not a recipe, and it did not look like a failure -- it looked like a
    confident one-step synthesis. Handed on, the simulator would have balanced
    X -> X, found dH = 0 and reported it feasible.

    Same for TiO2, Fe2O3, ZnO and SiO2: all are both common targets and common
    precursors.
    """
    from synflow.recipe_metrics import MetalLookupBaseline
    from synflow.recipes import Recipe

    train = [
        Recipe("BaTiO3", ["BaCO3", "TiO2"], (), (1100,)),
        Recipe("Al2O3", ["Al2O3"], (), (1400,)),
        Recipe("SrTiO3", ["SrCO3", "TiO2"], (), (1300,)),
    ]
    rule = MetalLookupBaseline(train)

    assert rule.predict("Al2O3") == frozenset()
    assert rule.predict("TiO2") == frozenset()
    # and the real routes are untouched
    assert rule.predict("BaTiO3") == frozenset({"BaCO3", "TiO2"})


def test_only_the_target_itself_is_excluded():
    """Not every precursor that is a target somewhere else. Ba2TiO4 from
    BaTiO3 and BaCO3 is a real route and BaTiO3 is a target in its own right;
    excluding it would throw away correct chemistry."""
    from synflow.recipe_metrics import MetalLookupBaseline
    from synflow.recipes import Recipe

    rule = MetalLookupBaseline([
        Recipe("Ba2TiO4", ["BaTiO3", "BaCO3"], (), (1200,)),
        Recipe("BaTiO3", ["BaCO3", "TiO2"], (), (1100,)),
    ])
    assert "BaTiO3" in rule.predict("Ba2TiO4")


def test_the_same_compound_written_two_ways_still_counts_as_circular():
    """Compared by composition, not by string. Fe2O3 and Fe4O6 are one
    compound, and this project has already been caught comparing raw formula
    strings -- Li3NiMnCoO6 against Materials Project's Li3MnCoNiO6 -- and
    concluding the database held no NMC cathodes."""
    from synflow.recipe_metrics import MetalLookupBaseline as M

    assert M._same_compound("Fe2O3", "Fe4O6")
    assert M._same_compound("TiO2", "Ti2O4")
    assert M._same_compound("LiCoO2", "Li2Co2O4")
    assert not M._same_compound("BaTiO3", "BaCO3")
    assert not M._same_compound("Fe2O3", "FeO")
