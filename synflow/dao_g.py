"""DAO-G: generate crystal structures, steered by DAO-P.

Starts from the simplest thing that actually closes the loop rather than from
the blueprint's flow-matching transformer, for two reasons.

First, DAO-P now returns FORCES, not just scores. ORB-v3 is conservative: it
differentiates its own energy with respect to atomic positions, so it hands
back a direction to move every atom. That is a gradient a generator can
descend directly, and it exists today.

Second, a relaxation baseline is what a flow-matching model will have to beat.
Without it, "the generator produced a valid structure" has no scale -- the
same mistake that cost a week on DAO-P, where a model was built before anyone
knew what a good number looked like.

So: scatter atoms of a requested composition into a cell, then walk downhill
on DAO-P's forces. Crude, physically grounded, and measurable. Flow matching
earns its place by beating it.

WHAT MAKES A GENERATED STRUCTURE GOOD
-------------------------------------
Three things, and they are checked separately because a structure can pass one
while failing another:

  valid     no atoms closer than physically possible -- a cell with two atoms
            on top of each other has a finite energy under any potential and
            means nothing
  stable    DAO-P's energy, comparable only against other structures of the
            same composition
  novel     not a copy of something already in Materials Project

A generator that scores well on stability by reproducing known structures has
discovered nothing, so novelty is measured, not assumed.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from .data.periodic_table import ELEMENTS

_SYMBOL_TO_Z = {v[0]: z for z, v in ELEMENTS.items()}
_COVALENT_RADIUS = {z: v[1] for z, v in ELEMENTS.items()}
_VDW_RADIUS = {z: v[2] for z, v in ELEMENTS.items()}

# Two atoms closer than this fraction of the sum of their covalent radii are
# overlapping, not bonded. 0.5 is deliberately permissive -- real bonds do
# compress, and the check is meant to catch nonsense, not to police chemistry.
MIN_BOND_FRACTION = 0.5


def random_structure(composition: Dict[str, int], volume_per_atom: float = 20.0,
                     seed: Optional[int] = None, min_fraction: float = 0.7,
                     attempts: int = 200):
    """Scatter the requested atoms into a cubic cell.

    Deliberately unstructured: the point is to see how far DAO-P's forces can
    carry a structure that started as noise, so the starting point must not
    smuggle in any crystallography.

    Positions are redrawn until no two atoms overlap. Uniform placement in a
    small cell routinely puts atoms at a quarter of a bond length, where a
    potential reports forces of hundreds of eV/Angstrom and no optimiser
    recovers -- the first attempt here started at a distance ratio of 0.25 and
    oscillated for twenty-four steps without ever becoming valid. Rejecting
    those starts is not cheating: it asks the generator to explore
    arrangements, not to escape from atoms sitting inside one another.

    Args:
        composition: {element symbol: count}
        volume_per_atom: cubic Angstrom. 20 is typical for oxides.
        min_fraction: reject a draw whose closest pair is below this fraction
            of the expected bond length.
    """
    from pymatgen.core import Lattice, Structure

    rng = np.random.default_rng(seed)
    species = [s for s, n in composition.items() for _ in range(int(n))]
    if not species:
        raise ValueError("empty composition")

    side = (volume_per_atom * len(species)) ** (1 / 3)
    lattice = Lattice.cubic(side)

    best, best_ratio = None, -1.0
    for _ in range(attempts):
        candidate = Structure(lattice, species, rng.random((len(species), 3)),
                              coords_are_cartesian=False)
        ratio = min_distance_ratio(candidate)
        if ratio > best_ratio:
            best, best_ratio = candidate, ratio
        if ratio >= min_fraction:
            return candidate
    # nothing clean found -- return the roomiest draw rather than a broken one
    return best


# Calibrated on 30,000 Materials Project structures: the median ratio of true
# volume per atom to the mean sphere volume of its atoms.
_PACKING = 1.0705

# The radius each atom is given is a blend of its covalent and van der Waals
# radii, not the covalent radius alone. Covalent alone is catastrophic for
# light elements -- hydrogen's covalent radius is 0.31 A, so r^3 makes an H
# atom almost weightless and the estimate for H2 came out at 0.20 A^3 per
# atom against a real 16.11, wrong by 82x. Blending in the van der Waals
# radius, which is what actually sets how close atoms sit when they are not
# bonded, drops the average error from 25.7% to 21.0%.
_COVALENT_SHARE = 0.75

# No structure among 30,000 has less than 5.96 A^3 per atom, and anything
# above 60 is a slab or a 2D sheet in a box of vacuum rather than a bulk
# crystal. Clamping to that range costs nothing on real chemistry and stops
# a bad guess producing a cell so small that every atom sees hundreds of its
# own periodic images.
_VOLUME_RANGE = (6.0, 60.0)


def estimate_volume_per_atom(composition: Dict[str, int]) -> float:
    """Guess a cell size from the composition alone, touching no answer.

    This exists so that evaluation does not hand the generator the volume of
    the structure it is being asked to find. `relax` holds the cell FIXED, so
    passing the known volume gives the baseline the right cell for free and
    leaves it only the positions to solve -- which flatters every gap it
    reports.

    Honest about its accuracy: on the 844 cached structures this is off by
    29% on average, against 34% for simply always saying 20 A^3. Composition
    predicts volume only weakly (correlation 0.68), because how atoms pack is
    most of the answer and that is the thing being generated. It is used
    anyway, because a weak guess that knows nothing beats a good one that
    knows the answer.

    It does at least track chemistry rather than missing it, which was worth
    checking before fine-tuning on a single domain:

        general Materials Project    true 17.98   estimated 18.64
        oxides with recipes          true 12.82   estimated ~12

    Oxides are dense, and the estimate follows them down because oxygen is
    small. Residual bias is deliberately left alone -- correcting it per
    domain would be fitting the prior to the answer, which is the habit this
    function exists to break.

    WHAT THE TAIL DID, before the blend and the clamp. The averages above
    looked fine while the tail was catastrophic: H2 came out at 0.20 A^3 per
    atom against a real 16.11. A prior that small puts every atom within
    5 A of hundreds of its own periodic images -- 431 edges per atom was
    measured -- and the velocity head, which sums over edges, produced a
    coordinate loss of 30,506,518. Training survived it only because
    gradients were clipped.
    """
    radii = []
    for symbol, count in composition.items():
        z = _SYMBOL_TO_Z.get(symbol, 6)
        radius = (_COVALENT_SHARE * _COVALENT_RADIUS.get(z, 0.7)
                  + (1.0 - _COVALENT_SHARE) * _VDW_RADIUS.get(z, 1.7))
        radii.extend([radius] * int(count))
    if not radii:
        raise ValueError("empty composition")
    spheres = float(np.mean([4.0 / 3.0 * np.pi * r ** 3 for r in radii]))
    return float(np.clip(_PACKING * spheres, *_VOLUME_RANGE))


def shortest_lattice_translation(lattice) -> float:
    """Length of the shortest non-zero lattice translation.

    This is how far an atom sits from its own nearest periodic image, and it
    caps how small a cell may be regardless of what is inside it. Taking the
    shortest cell edge is not enough: in a skewed cell a diagonal combination
    can be shorter, so the 3x3x3 block of translations is searched, the same
    way the periodic graph builder does.
    """
    matrix = np.asarray(lattice.matrix, dtype=float)
    shifts = np.array([[i, j, k]
                       for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
                       if (i, j, k) != (0, 0, 0)], dtype=float)
    return float(np.linalg.norm(shifts @ matrix, axis=1).min())


def min_distance_ratio(structure) -> float:
    """Closest approach as a fraction of the expected bond length.

    Below 1 means atoms sit closer than their covalent radii suggest; below
    MIN_BOND_FRACTION they are overlapping rather than bonded.

    Two sources of closeness, and an earlier version checked only the first:

    * between different atoms, under the minimum image convention;
    * between an atom and its OWN periodic image. pymatgen's distance matrix
      has nothing on its diagonal, so a single atom in a 1.2 Angstrom cell was
      reported as perfectly fine -- and in a many-atom cell, a cell too small
      in one direction passed the same way.
    """
    radii = [_COVALENT_RADIUS.get(_SYMBOL_TO_Z.get(str(site.specie), 6), 0.7)
             for site in structure]

    # an atom against its own image, which bounds how small the cell may be
    translation = shortest_lattice_translation(structure.lattice)
    worst = min(translation / (2.0 * r) if r > 0 else float("inf")
                for r in radii)

    n = len(structure)
    if n > 1:
        distances = structure.distance_matrix.copy()
        np.fill_diagonal(distances, np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                expected = radii[i] + radii[j]
                if expected > 0:
                    worst = min(worst, distances[i, j] / expected)
    return worst


def is_valid(structure, min_fraction: float = MIN_BOND_FRACTION) -> bool:
    """Physically possible, not merely finite-energy."""
    return min_distance_ratio(structure) >= min_fraction


class _DAOPCalculator:
    """ASE calculator over DAO-P, so real optimisers can drive it.

    ASE wants total energy; DAO-P reports per atom. Forces are already
    per-atom vectors and pass through unchanged.

    Two things keep the potential from being run more often than the physics
    requires, and without them a relaxation costs four forward passes per step
    instead of one:

    * energy and forces come from a single `energy_and_forces` call, because
      a conservative model computes the forces on its way to the energy;
    * results are cached on the exact positions and cell they were computed
      for. FIRE asks for the energy and the forces separately at the same
      geometry, and it also re-reads the last geometry when a step is
      rejected. Keying on the coordinates means a repeat question is free and
      a moved atom always invalidates the answer.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, dao_p):
        self.dao_p = dao_p
        self.results = {}
        self.atoms = None
        self._key = None
        self.calls = 0

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self._compute(atoms)["energy"]

    def get_forces(self, atoms=None):
        return self._compute(atoms)["forces"]

    def get_stress(self, atoms=None):
        raise NotImplementedError("cell relaxation is a separate step")

    def calculate(self, atoms=None, properties=None, system_changes=None):
        self.results = self._compute(atoms)

    def _compute(self, atoms):
        from pymatgen.io.ase import AseAtomsAdaptor

        key = (atoms.get_positions().tobytes(),
               np.asarray(atoms.get_cell()).tobytes())
        if key == self._key:
            return self.results

        structure = AseAtomsAdaptor.get_structure(atoms)
        per_atom, forces = self.dao_p.energy_and_forces(structure)
        self.calls += 1
        self.results = {"energy": per_atom * len(atoms),
                        "forces": np.asarray(forces, dtype=float)}
        self._key = key
        return self.results


def relax(structure, dao_p, steps: int = 60, fmax: float = 0.05,
          trace: bool = False):
    """Relax atomic positions under DAO-P, using FIRE.

    Hand-rolled gradient descent does not work here. A fixed step against a
    stiff potential overshoots the minimum and oscillates: the first attempt
    saw energy swing 32.7 -> 0.8 -> 33.5 -> 0.2 -> 37.8 while forces GREW from
    43 to 117 eV/Angstrom. FIRE adapts its own step and is what the field
    actually uses.

    The cell is held fixed. ORB returns a stress too and relaxing the cell is
    a real improvement, but mixing both makes a failure harder to attribute.

    Args:
        fmax: stop once every force is below this, in eV/Angstrom. 0.05 is the
            usual convergence criterion for a relaxed structure.
    """
    import io
    from contextlib import redirect_stdout

    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = _DAOPCalculator(dao_p)

    history = []
    if trace:
        history.append(_snapshot(0, atoms, dao_p))

    optimiser = FIRE(atoms, logfile=None)
    if trace:
        optimiser.attach(
            lambda: history.append(_snapshot(optimiser.nsteps, atoms, dao_p)))
    with redirect_stdout(io.StringIO()):
        optimiser.run(fmax=fmax, steps=steps)

    relaxed = _rewrap(AseAtomsAdaptor.get_structure(atoms))
    return (relaxed, history) if trace else relaxed


def _snapshot(step, atoms, dao_p):
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = AseAtomsAdaptor.get_structure(atoms)
    energy, forces = dao_p.energy_and_forces(structure)
    forces = np.asarray(forces)
    return {"step": step, "energy": energy,
            "max_force": float(np.linalg.norm(forces, axis=1).max()),
            "min_distance_ratio": min_distance_ratio(structure)}


def _rewrap(structure):
    """Fold atoms back inside the cell.

    Without this an atom drifts out over successive steps and the periodic
    graph, which assumes fractional coordinates in [0, 1), silently sees a
    different neighbourhood than the one on screen.
    """
    from pymatgen.core import Structure

    return Structure(structure.lattice,
                     [site.specie for site in structure],
                     np.mod(structure.frac_coords, 1.0),
                     coords_are_cartesian=False)


def generate(composition: Dict[str, int], dao_p, n_candidates: int = 8,
             steps: int = 60, seed: int = 0, volume_per_atom: float = 20.0):
    """Generate and rank candidates for one composition.

    Several random starts rather than one: relaxation finds the nearest local
    minimum, and a single start reports wherever noise happened to land.

    Returns candidates sorted most stable first, each with its validity and
    the energy it started and ended at.
    """
    results = []
    for k in range(n_candidates):
        start = random_structure(composition, volume_per_atom, seed=seed + k)
        start_energy = dao_p.score(start)
        relaxed = relax(start, dao_p, steps=steps)
        results.append({
            "structure": relaxed,
            "energy": dao_p.score(relaxed),
            "start_energy": start_energy,
            "valid": is_valid(relaxed),
            "min_distance_ratio": min_distance_ratio(relaxed),
            "seed": seed + k,
        })
    return sorted(results, key=lambda r: r["energy"])


def match_known(structure, references: Sequence, tolerance: float = 0.2) -> bool:
    """Is this structure already known?

    Uses pymatgen's StructureMatcher, which compares up to lattice choice,
    origin and atom ordering -- the same crystal written two ways is one
    crystal. A generator scoring well by rediscovering known compounds has
    found nothing, so this is measured rather than assumed.
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher

    matcher = StructureMatcher(ltol=tolerance, stol=tolerance, angle_tol=10)
    return any(matcher.fit(structure, reference) for reference in references)
