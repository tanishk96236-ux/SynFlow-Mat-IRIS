"""DAO-G: structure generation and the checks that keep it honest."""

import numpy as np
import pytest

pytest.importorskip("pymatgen", reason="pymatgen is an optional MP extra")

from pymatgen.core import Lattice, Structure  # noqa: E402

from synflow import dao_g  # noqa: E402


def test_random_structure_has_the_requested_atoms():
    s = dao_g.random_structure({"Ba": 1, "Ti": 1, "O": 3}, seed=0)
    assert len(s) == 5
    assert sorted(str(site.specie) for site in s) == ["Ba", "O", "O", "O", "Ti"]


def test_random_structure_avoids_overlapping_atoms():
    """Uniform placement in a small cell routinely puts atoms at a quarter of
    a bond length, where forces reach hundreds of eV/Angstrom and no optimiser
    recovers. The first attempt here started at 0.25 and never became valid."""
    for seed in range(6):
        s = dao_g.random_structure({"Ba": 1, "Ti": 1, "O": 3}, seed=seed)
        assert dao_g.min_distance_ratio(s) > 0.5, seed


def test_empty_composition_is_rejected():
    with pytest.raises(ValueError):
        dao_g.random_structure({})


def test_min_distance_ratio_counts_periodic_images():
    """One atom in a small cell is still close to its own image across the
    boundary; a check over the cell contents alone would call it fine."""
    tiny = Structure(Lattice.cubic(1.2), ["Fe"], [[0, 0, 0]])
    roomy = Structure(Lattice.cubic(12.0), ["Fe"], [[0, 0, 0]])
    assert dao_g.min_distance_ratio(tiny) < dao_g.min_distance_ratio(roomy)


def test_volume_is_estimated_from_composition_alone():
    """It must not be able to see the answer, and it must be roughly right.

    Handing a generator the known structure's volume is a leak, and a sharp
    one: relax() holds the cell fixed, so the baseline would be given the
    right cell and left only the positions to solve.
    """
    # a rough sanity band -- most inorganic crystals sit near 20 A^3/atom
    for composition in ({"Na": 1, "Cl": 1}, {"Ba": 1, "Ti": 1, "O": 3},
                        {"Fe": 2, "O": 3}, {"Li": 1, "Fe": 1, "P": 1, "O": 4}):
        volume = dao_g.estimate_volume_per_atom(composition)
        assert 5.0 < volume < 60.0, (composition, volume)

    # bigger atoms, bigger cell -- caesium against lithium
    assert (dao_g.estimate_volume_per_atom({"Cs": 1, "F": 1}) >
            dao_g.estimate_volume_per_atom({"Li": 1, "F": 1}))


def test_estimated_volume_is_independent_of_atom_count():
    """Per atom means per atom. A doubled cell is the same material, so the
    estimate must not change -- otherwise a supercell would be given a
    different prior than the primitive cell of the same crystal."""
    single = dao_g.estimate_volume_per_atom({"Na": 1, "Cl": 1})
    doubled = dao_g.estimate_volume_per_atom({"Na": 2, "Cl": 2})
    assert single == pytest.approx(doubled)


def test_overlapping_atoms_are_invalid():
    overlapping = Structure(Lattice.cubic(8.0), ["Fe", "Fe"],
                            [[0, 0, 0], [0.02, 0, 0]])
    separated = Structure(Lattice.cubic(8.0), ["Fe", "Fe"],
                          [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert not dao_g.is_valid(overlapping)
    assert dao_g.is_valid(separated)


def test_rewrap_folds_atoms_back_into_the_cell():
    """An atom drifting outside means the periodic graph sees a different
    neighbourhood than the coordinates suggest."""
    drifted = Structure(Lattice.cubic(5.0), ["Fe"], [[1.4, -0.3, 2.2]],
                        coords_are_cartesian=False)
    wrapped = dao_g._rewrap(drifted)
    assert ((wrapped.frac_coords >= 0) & (wrapped.frac_coords < 1)).all()


# --------------------------------------------------------------------------
# relaxation, against a potential with a known minimum
# --------------------------------------------------------------------------

class _SpringCritic:
    """Pulls every atom towards the cell centre. Analytic, no model needed,
    and its minimum is obvious -- so a relaxation that fails here is broken
    rather than merely unlucky."""

    name = "spring"

    def __init__(self):
        self.evaluations = 0

    def _centre(self, structure):
        return np.array(structure.lattice.abc) / 2.0

    def score(self, structure):
        d = structure.cart_coords - self._centre(structure)
        return float((d ** 2).sum()) / len(structure)

    def forces(self, structure):
        return -2.0 * (structure.cart_coords - self._centre(structure))

    def energy_and_forces(self, structure):
        """Counted, because how OFTEN this runs is the whole cost of DAO-G."""
        self.evaluations += 1
        return self.score(structure), self.forces(structure)


def test_relaxation_lowers_the_energy():
    critic = _SpringCritic()
    start = dao_g.random_structure({"Fe": 2, "O": 2}, seed=3)
    relaxed = dao_g.relax(start, critic, steps=40)
    assert critic.score(relaxed) < critic.score(start)


def test_relaxation_reduces_the_forces():
    """The measure that caught the broken optimiser: plain gradient descent
    left forces GROWING, 43 to 117 eV/Angstrom."""
    critic = _SpringCritic()
    start = dao_g.random_structure({"Fe": 2, "O": 2}, seed=4)
    relaxed = dao_g.relax(start, critic, steps=40)
    before = np.abs(critic.forces(start)).max()
    after = np.abs(critic.forces(relaxed)).max()
    assert after < before


def test_relaxation_evaluates_the_potential_once_per_geometry():
    """ORB is the entire cost of DAO-G, so the count is worth a test.

    FIRE asks for the energy and the forces as two separate questions about
    one geometry, and re-reads a geometry it has already seen when a step is
    rejected. Answering each of those with a fresh forward pass ran the
    potential four times per step. The bound here is one evaluation per step
    plus the initial one; anything above it means the cache stopped working.
    """
    critic = _SpringCritic()
    start = dao_g.random_structure({"Fe": 2, "O": 2}, seed=7)
    steps = 20
    dao_g.relax(start, critic, steps=steps)
    assert critic.evaluations <= steps + 2, critic.evaluations


def test_relaxation_keeps_the_composition():
    critic = _SpringCritic()
    start = dao_g.random_structure({"Fe": 2, "O": 3}, seed=5)
    relaxed = dao_g.relax(start, critic, steps=20)
    assert (sorted(str(s.specie) for s in relaxed) ==
            sorted(str(s.specie) for s in start))


def test_generate_returns_candidates_sorted_by_energy():
    critic = _SpringCritic()
    results = dao_g.generate({"Fe": 2, "O": 2}, critic, n_candidates=3, steps=15)
    assert len(results) == 3
    energies = [r["energy"] for r in results]
    assert energies == sorted(energies)
    assert all("valid" in r and "start_energy" in r for r in results)


def test_known_structures_are_recognised():
    """A generator scoring well by rediscovering known compounds has found
    nothing, so novelty has to be measured."""
    known = Structure(Lattice.cubic(4.0), ["Na", "Cl"],
                      [[0, 0, 0], [0.5, 0.5, 0.5]])
    same_written_differently = Structure(
        Lattice.cubic(4.0), ["Cl", "Na"], [[0.5, 0.5, 0.5], [0, 0, 0]])
    different = Structure(Lattice.cubic(4.0), ["Na", "Cl"],
                          [[0, 0, 0], [0.25, 0.25, 0.25]])
    assert dao_g.match_known(same_written_differently, [known])
    assert not dao_g.match_known(different, [known])
