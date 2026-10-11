"""DAO-P and the swappable potential layer.

DAO-P is ORB-v3 because that is what measured best on our own polymorph
benchmark, not because a leaderboard said so. It will be replaced -- the
OMat-trained eSEN leading Matbench Discovery is not publicly downloadable yet,
and the field moves quarterly. These tests pin the seam that makes replacing
it a config change rather than a rewrite.
"""

import pytest

from synflow import dao_p, potentials


def test_registry_lists_the_expected_potentials():
    assert set(potentials.REGISTRY) >= {"orb", "m3gnet", "uma", "mace", "chgnet"}
    assert potentials.DEFAULT in potentials.REGISTRY


def test_unknown_potential_names_the_alternatives():
    with pytest.raises(ValueError) as exc:
        potentials.load("nonexistent")
    assert "orb" in str(exc.value)


def test_measured_numbers_come_from_our_own_benchmark():
    """Guards against someone pasting leaderboard figures in here."""
    for name, m in potentials.MEASURED.items():
        assert name in potentials.REGISTRY
        assert 0.5 <= m["ranking"] <= 1.0
        assert 0.0 < m["gap_mae"] < 0.1
    # ORB must be recorded as better than M3GNet, which is why it is default
    assert (potentials.MEASURED["orb"]["ranking"] >
            potentials.MEASURED["m3gnet"]["ranking"])
    assert (potentials.MEASURED["orb"]["gap_mae"] <
            potentials.MEASURED["m3gnet"]["gap_mae"])


def test_default_is_the_best_measured_model():
    best = max(potentials.MEASURED.items(), key=lambda kv: kv[1]["ranking"])[0]
    assert potentials.DEFAULT == best, (
        "the default should be whichever model actually measured best")


# --------------------------------------------------------------------------
# behaviour, against a stub -- no model download, no network
# --------------------------------------------------------------------------

def _stub(energies, with_forces=True):
    """A potential whose energy is looked up per structure id."""
    import numpy as np

    return potentials.Potential(
        name="stub",
        energy=lambda s: energies[id(s)],
        forces=(lambda s: np.zeros((3, 3))) if with_forces else None,
        is_total_energy=True,
        note="stub")


def test_ranks_most_stable_first():
    a, b, c = object(), object(), object()
    p = dao_p.DAOP(_stub({id(a): -1.0, id(b): -3.0, id(c): -2.0}))
    assert p.rank([a, b, c]) == [1, 2, 0]


def test_better_compares_by_energy():
    a, b = object(), object()
    p = dao_p.DAOP(_stub({id(a): -2.0, id(b): -1.0}))
    assert p.better(a, b)
    assert not p.better(b, a)


def test_missing_forces_raise_rather_than_return_zeros():
    """A generator silently steered by nothing is worse than one that stops.

    M3GNet has no force output, so asking it for a Phase III gradient must
    fail loudly.
    """
    a = object()
    p = dao_p.DAOP(_stub({id(a): -1.0}, with_forces=False))
    with pytest.raises(NotImplementedError) as exc:
        p.forces(a)
    assert "orb" in str(exc.value)


def test_m3gnet_is_declared_forceless():
    """Recorded in the registry so Phase III cannot pick it by accident."""
    import inspect
    src = inspect.getsource(potentials.load_m3gnet)
    assert "forces=None" in src


def test_gap_is_a_difference_not_an_absolute():
    """Absolute energies differ between potentials -- formation vs total --
    so everything downstream must be built on differences."""
    a, b = object(), object()
    p = potentials.Potential(name="s", energy=lambda s: {id(a): 5.0, id(b): 3.0}[id(s)],
                             forces=None, is_total_energy=True)
    assert p.gap(a, b) == pytest.approx(2.0)
    # shifting both by a constant, as a different reference would, changes nothing
    q = potentials.Potential(name="s", energy=lambda s: {id(a): -95.0, id(b): -97.0}[id(s)],
                             forces=None, is_total_energy=True)
    assert q.gap(a, b) == pytest.approx(p.gap(a, b))
