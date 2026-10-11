"""DAO-P: the stability critic the generator steers by.

Currently ORB-v3, chosen by measurement on our own polymorph benchmark rather
than from a leaderboard:

    floor (predict no difference)   50.0%    gap MAE 0.0279
    our own EGNN, held out          ~50%     gap MAE 0.047
    M3GNet                          77.8%    gap MAE 0.0238
    ORB-v3                          89.5%    gap MAE 0.0124

    Those are from a set drawn with a different seed to the one the models
    were compared on, so selecting ORB did not also score it. The selected
    figure was 85.8%; the clean one is higher, so the selection bias ran
    against us.

The second column is the one that decides it. The median energy difference
between polymorphs here is 0.0161 eV/atom; a model whose error exceeds that
can only separate obviously different arrangements. ORB's is below it.

This class is deliberately thin. Everything specific to ORB lives in
synflow/potentials.py, so replacing DAO-P is a name in a config, not a rewrite
-- which matters because it will be replaced. The OMat-trained eSEN leading
Matbench Discovery is not yet publicly downloadable, and the field moves.

What Phase III will actually use is `forces`, not `score`. A force says which
atom to move and in which direction; an energy only grades a finished
structure. ORB is conservative, so its forces are the derivative of its own
energy rather than a separately predicted quantity.
"""

from typing import Iterable, List, Optional, Sequence

import numpy as np

from . import potentials


class DAOP:
    """Scores crystal structures for stability, and ranks arrangements.

    Absolute values are not comparable across potentials -- some return
    formation energy, some total energy. Only differences within a composition
    are meaningful, and there the elemental reference cancels. Every method
    here is built on differences for that reason.
    """

    def __init__(self, potential: Optional[potentials.Potential] = None,
                 name: str = potentials.DEFAULT, **kwargs):
        self.potential = potential or potentials.load(name, **kwargs)

    @property
    def name(self) -> str:
        return self.potential.name

    def score(self, structure) -> float:
        """Energy per atom, eV. Comparable only within one composition."""
        return self.potential.energy(structure)

    def forces(self, structure) -> np.ndarray:
        """[N, 3] forces in eV/Angstrom -- the signal a generator descends.

        Raises rather than returning zeros: a generator silently steered by
        nothing is far worse than one that refuses to start.
        """
        if self.potential.forces is None:
            raise NotImplementedError(
                "{} provides no forces; DAO-G needs a potential that does "
                "(orb, uma, mace, chgnet)".format(self.name))
        return self.potential.forces(structure)

    def energy_and_forces(self, structure):
        """Both, from one forward pass where the potential allows it.

        A relaxation step needs the energy and the forces at the same
        geometry. Asking for them separately runs ORB twice per step, which is
        pure waste on a conservative model that computed the forces while
        producing the energy.
        """
        if self.potential.forces is None:
            raise NotImplementedError(
                "{} provides no forces; DAO-G needs a potential that does "
                "(orb, uma, mace, chgnet)".format(self.name))
        return self.potential.both(structure)

    def better(self, structure_a, structure_b) -> bool:
        """Is A more stable than B? Both must share a composition."""
        return self.potential.gap(structure_a, structure_b) < 0

    def rank(self, structures: Sequence) -> List[int]:
        """Indices ordered most stable first."""
        scored = [(self.score(s), i) for i, s in enumerate(structures)]
        return [i for _, i in sorted(scored)]

    def score_all(self, structures: Iterable) -> np.ndarray:
        return np.array([self.score(s) for s in structures])

    def __repr__(self) -> str:
        measured = potentials.MEASURED.get(self.name)
        detail = ("" if not measured else
                  ", benchmark {:.1%} ranking / {:.4f} gap MAE".format(
                      measured["ranking"], measured["gap_mae"]))
        return "DAOP({}{})".format(self.potential.note, detail)


def load(name: str = potentials.DEFAULT, **kwargs) -> DAOP:
    """The single place DAO-P is chosen."""
    return DAOP(name=name, **kwargs)
