"""DAO-S model, decoding and the priors built into it."""

import pytest
import torch

from synflow.dao_s import DAOS, RecipeNet, composition_tensor, encode_batch
from synflow.recipes import Recipe


def _r(target, precursors, temps=()):
    return Recipe(target, precursors, (), temps)


def test_composition_is_scale_invariant():
    """Fe2O3 and Fe4O6 are the same compound written twice."""
    assert torch.allclose(composition_tensor("Fe2O3"),
                          composition_tensor("Fe4O6"))


def test_composition_of_nonsense_is_empty():
    assert composition_tensor("YAG").sum() == 0.0


def test_encode_marks_missing_temperatures():
    """24% of recipes record no temperature; training a regressor against a
    filled-in value would teach it the filling-in."""
    _, _, temps, mask = encode_batch(
        [_r("Fe2O3", ["Fe"], temps=(900.0,)), _r("TiO2", ["Ti"])],
        ["Fe", "Ti"])
    assert mask.tolist() == [True, False]
    assert temps[0] == pytest.approx(9.0)      # 900 C in units of 100


def test_encode_builds_multi_label_targets():
    _, y, _, _ = encode_batch([_r("BaTiO3", ["BaCO3", "TiO2"])],
                              ["BaCO3", "TiO2", "Fe2O3"])
    assert y[0].tolist() == [1.0, 1.0, 0.0]


def test_out_of_vocabulary_precursors_are_dropped_not_crashed():
    _, y, _, _ = encode_batch([_r("X", ["unseen"])], ["BaCO3"])
    assert y.sum() == 0.0


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------

def _daos(vocabulary, logits, threshold=0.5, chemical_mask=True):
    model = RecipeNet(len(vocabulary), hidden=8)
    with torch.no_grad():
        model.precursor_head.weight.zero_()
        model.precursor_head.bias.copy_(torch.tensor(logits))
    return DAOS(model, vocabulary, threshold, chemical_mask)


def test_never_returns_an_empty_recipe():
    """A recipe with no ingredients is not a cautious answer, it is unusable."""
    daos = _daos(["BaCO3", "TiO2"], [-10.0, -10.0], threshold=0.9)
    assert len(daos.predict("BaTiO3")) >= 1


def test_caps_the_number_of_precursors():
    """Real recipes average 2.8 and rarely exceed six. Returning fifty wrecks
    precision while flattering recall."""
    vocabulary = ["Ba{}".format(i) for i in range(30)]
    daos = _daos(vocabulary, [10.0] * 30, threshold=0.1, chemical_mask=False)
    assert len(daos.predict("BaTiO3")) <= DAOS.MAX_PRECURSORS


def test_chemical_mask_rejects_unrelated_precursors():
    """91.4% of real precursors share an element with their target. The mask
    supplies that rule instead of making the network rediscover it."""
    vocabulary = ["BaCO3", "CuO"]
    daos = _daos(vocabulary, [0.0, 5.0], threshold=0.1, chemical_mask=True)
    # CuO scores far higher, but BaTiO3 contains no copper
    assert "CuO" not in daos.predict("BaTiO3")


def test_mask_can_be_disabled():
    vocabulary = ["BaCO3", "CuO"]
    daos = _daos(vocabulary, [0.0, 5.0], threshold=0.1, chemical_mask=False)
    assert "CuO" in daos.predict("BaTiO3")


def test_temperature_comes_back_in_celsius():
    model = RecipeNet(2, hidden=8)
    with torch.no_grad():
        model.temperature_head.weight.zero_()
        model.temperature_head.bias.fill_(10.0)     # 10 * 100 C
    daos = DAOS(model, ["a", "b"])
    assert daos.predict_temperature("Fe2O3") == pytest.approx(1000.0)


def test_batch_and_single_prediction_agree():
    vocabulary = ["BaCO3", "TiO2", "CuO"]
    daos = _daos(vocabulary, [2.0, 1.0, -3.0], threshold=0.5)
    single = daos.predict("BaTiO3")
    batched = daos.predict_batch(["BaTiO3"])[0][0]
    assert single == batched


def test_an_unparseable_target_gets_a_refusal_not_a_prior():
    """The failure mode this project keeps producing: confident nonsense in
    place of an honest refusal.

    Given a formula it cannot turn into a composition, the rule finds no
    metals and returns nothing, and the network receives an all-zero
    composition vector and answers with its unconditional prior -- the
    training set's most common precursors, the same set every time,
    containing elements the compound does not have. That is not a
    conservative answer, it is an unusable one dressed as an answer.
    """
    from synflow.dao_s import HybridDAOS

    class Rule:
        def predict(self, target):
            return frozenset()

    class Network:
        def predict(self, target):
            return frozenset({"Bi2O3", "CuO", "Fe2O3", "Li2CO3"})

        def predict_temperature(self, target):
            return 749.0

    model = HybridDAOS(Rule(), Network())

    # A site label, not a compound: nothing can be said about it.
    assert model.predict_recipe("A0.5Ti0.5NbO4") is None
    assert model.predict_recipe("YAG") is None

    # A real formula still gets an answer, including the parenthesised
    # spelling pymatgen produces.
    for formula in ("BaSb2O4", "Ba(SbO2)2"):
        recipe = model.predict_recipe(formula)
        assert recipe is not None, formula
        assert recipe["precursors"]
        assert recipe["firing_temperature_C"] == 749


# --------------------------------------------------------------------------
# the shipped predictor
# --------------------------------------------------------------------------

def test_a_network_trained_on_other_precursor_inputs_is_refused(tmp_path):
    """TemperatureNet reads the precursors as part of its input, so it must be
    paired with the predictor it was trained against. The first one was
    trained on the plain rule's output; putting ConditionalLookup in front of
    it would hand it inputs it never saw -- the same mismatch as teacher
    forcing, and it would not announce itself."""
    import pytest
    import torch

    from synflow.dao_s import load_shipped

    path = tmp_path / "best.pt"
    torch.save({"precursors": "rule"}, path)
    with pytest.raises(RuntimeError, match="retrain"):
        load_shipped(str(path), lookup_cache=str(tmp_path / "lookup.json"),
                     verbose=False)


def test_the_shipped_predictor_refuses_an_unreadable_formula():
    """Same refusal HybridDAOS made: an unreadable formula gives the network
    an all-zero composition and it answers with its prior -- one recipe for
    every such target."""
    from synflow.dao_s import RecipePredictor, TemperatureNet
    from synflow.recipe_metrics import ConditionalLookup
    from synflow.recipes import Recipe

    lookup = ConditionalLookup([Recipe("BaTiO3", ["BaCO3", "TiO2"], (),
                                       (1100,))] * 5, min_count=3)
    network = TemperatureNet(2, hidden=8, layers=1, mean_temperature=10.0)
    predictor = RecipePredictor(lookup, network, ["BaCO3", "TiO2"])

    assert predictor.predict_recipe("(Ba1-xCax)TiO3") is None
    recipe = predictor.predict_recipe("BaTiO3")
    assert recipe["precursors"] == ["BaCO3", "TiO2"]
    assert recipe["firing_temperature_C"] > 0
