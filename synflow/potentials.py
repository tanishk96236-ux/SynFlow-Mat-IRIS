"""Pretrained potentials behind one interface, so DAO-P can be swapped.

DAO-P is ORB-v3 today because it was measured to be the best available on our
own polymorph benchmark -- 89.5% ranking, gap MAE 0.0124 eV/atom against a
median polymorph gap of 0.0161, on compositions it was not selected on. Our
own EGNN sat at chance.

It will not be the best forever. Matbench Discovery moves; the OMat-trained
eSEN that leads it is not yet in a public registry, and UMA, MACE, CHGNet and
SevenNet all sit in the same space. So the choice is made in one place and
named in config, rather than having ORB's particulars spread through the
pipeline where replacing it would mean a rewrite.

Every potential here exposes the same two things:

    energy(structure) -> eV per atom
    forces(structure) -> [N, 3] eV/Angstrom, or None if it cannot

and, optionally, `energy_forces` returning both from one forward pass. That is
not a micro-optimisation: a relaxation step asks for the energy and the forces
at the same geometry, and without it a conservative model is run twice per
step, doubling the cost of every structure DAO-G generates.

Forces matter more than the ranking for Phase III. A force says which atom to
move and in which direction -- a gradient a generator can descend -- where an
energy only scores a finished structure. ORB is conservative, meaning it
obtains forces by differentiating its energy, so they are consistent with it
by construction.

Absolute energies are NOT comparable between potentials: some return formation
energy, others total energy. Only differences within a composition are used
anywhere, and the elemental reference cancels there. `is_total_energy` records
which is which so nothing downstream compares the wrong things.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class Potential:
    """A loaded potential, with what the rest of the code needs to know."""

    name: str
    energy: Callable          # structure -> eV/atom
    forces: Optional[Callable]  # structure -> [N, 3] eV/A, or None
    is_total_energy: bool
    note: str = ""
    # structure -> (eV/atom, [N, 3]) in one pass, when the backend can
    energy_forces: Optional[Callable] = None

    def both(self, structure):
        """Energy and forces, using one forward pass where that is possible."""
        if self.energy_forces is not None:
            return self.energy_forces(structure)
        if self.forces is None:
            raise NotImplementedError("{} provides no forces".format(self.name))
        return self.energy(structure), self.forces(structure)

    def gap(self, structure_a, structure_b) -> float:
        """Energy difference per atom. Valid across potentials even when the
        absolute scale is not, provided both structures share a composition."""
        return self.energy(structure_a) - self.energy(structure_b)


def load_orb(device: str = None, variant: str = "orb_v3_conservative_inf_omat"):
    """ORB-v3 conservative, trained on OMat24. Apache 2.0.

    Three things that are not obvious and cost a debugging cycle each:

    * the loader returns a (model, adapter) tuple, not a model;
    * it calls torch.compile internally, which needs a C++ compiler and fails
      on Windows with "Compiler: cl is not found". Merely suppressing the
      errors is not enough: dynamo still traces every function, fails, and
      logs a stack trace, which cost about two minutes and several hundred
      lines of output on every single run. It is turned off outright instead,
      since it can never succeed on a machine without `cl`;
    * it must NOT be wrapped in torch.no_grad. Being conservative, it gets
      forces by differentiating the energy, so cutting the gradient path fails
      with "element 0 of tensors does not require grad".
    """
    import logging

    import torch
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.disable = True
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

    from orb_models.forcefield import pretrained
    from pymatgen.io.ase import AseAtomsAdaptor

    # Follow the hardware rather than defaulting to CPU. The default used to
    # be "cpu", which was right on the development machine and wrong
    # everywhere a GPU exists: parts of ORB's own batch still land on CUDA,
    # and it fails inside searchsorted with cumsums on cuda:0 against an
    # index tensor on cpu.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, adapter = getattr(pretrained, variant)(
        device=device, precision="float32-high")
    model.eval()

    def _predict(structure):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        graph = adapter.from_ase_atoms(atoms)
        # The adapter builds the graph on the CPU whatever the model's
        # device, so it is moved explicitly. Same rule as everywhere else
        # here: the data follows the model, never the other way round.
        if hasattr(graph, "to"):
            graph = graph.to(device)
        return model.predict(graph), len(atoms)

    def _forces_from(out):
        # A CONSERVATIVE model returns "grad_forces": the forces are the
        # gradient of its own energy, not a separately predicted field. The
        # orb_v3_direct_* variants predict forces outright and use "forces".
        for key in ("grad_forces", "forces"):
            if key in out:
                return torch.as_tensor(out[key]).detach().cpu().numpy()
        raise KeyError("no force field in ORB output; got {}".format(list(out)))

    def _energy_from(out, n):
        # detach first: being conservative, the energy still carries the graph
        # its forces were differentiated from, and reading a scalar off a
        # tensor that requires grad warns for good reason.
        return float(torch.as_tensor(out["energy"]).detach().reshape(-1)[0]) / n

    def energy(structure):
        out, n = _predict(structure)
        return _energy_from(out, n)

    def forces(structure):
        return _forces_from(_predict(structure)[0])

    def energy_forces(structure):
        out, n = _predict(structure)
        return _energy_from(out, n), _forces_from(out)

    return Potential(name="orb", energy=energy, forces=forces,
                     is_total_energy=True,
                     note="ORB-v3 conservative, OMat24, Apache 2.0",
                     energy_forces=energy_forces)


def load_m3gnet(device: str = "cpu"):
    """M3GNet formation-energy head. Weakest of the common potentials on our
    benchmark (79.2%) -- kept as the reference point the others improved on."""
    import matgl
    model = matgl.load_model("M3GNet-Eform-MP-2019.4.1")
    return Potential(name="m3gnet",
                     energy=lambda s: float(model.predict_structure(s)),
                     forces=None, is_total_energy=False,
                     note="M3GNet-Eform-MP-2019.4.1, formation energy, no forces")


def load_uma(device: str = "cpu", checkpoint: str = "uma-s-1p2p1"):
    """FAIR's UMA on its materials task.

    Note the eSEN checkpoints fairchem publishes are trained on molecules
    (OMol25), catalysis (OC25) and MOFs (ODAC25) -- none on inorganic crystals.
    Using one of those here would score badly for the wrong reason. UMA is
    multi-domain and `task_name="omat"` selects materials.

    fairchem pulls hydra and omegaconf, which break torch_geometric. Keep it
    out of any environment that needs PyG, or use the portable benchmark.
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from pymatgen.io.ase import AseAtomsAdaptor

    calc = FAIRChemCalculator(
        pretrained_mlip.get_predict_unit(checkpoint, device=device),
        task_name="omat")

    def _atoms(structure):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = calc
        return atoms

    return Potential(
        name="uma",
        energy=lambda s: float(_atoms(s).get_potential_energy()) / len(s),
        forces=lambda s: np.asarray(_atoms(s).get_forces()),
        is_total_energy=True,
        note="UMA {} on task omat".format(checkpoint))


def load_mace(device: str = "cpu", model: str = "medium"):
    from mace.calculators import mace_mp
    from pymatgen.io.ase import AseAtomsAdaptor

    calc = mace_mp(model=model, default_dtype="float64", device=device)

    def _atoms(structure):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = calc
        return atoms

    return Potential(
        name="mace",
        energy=lambda s: float(_atoms(s).get_potential_energy()) / len(s),
        forces=lambda s: np.asarray(_atoms(s).get_forces()),
        is_total_energy=True, note="MACE-MP {}".format(model))


def load_chgnet(device: str = "cpu"):
    from chgnet.model.model import CHGNet
    model = CHGNet.load()

    return Potential(
        name="chgnet",
        energy=lambda s: float(model.predict_structure(s)["e"]),
        forces=lambda s: np.asarray(model.predict_structure(s)["f"]),
        is_total_energy=True, note="CHGNet pretrained")


REGISTRY = {
    "orb": load_orb,
    "m3gnet": load_m3gnet,
    "uma": load_uma,
    "mace": load_mace,
    "chgnet": load_chgnet,
}

# Measured on data/benchmark/polymorph_benchmark.json -- our own 408 pairs, not
# a published leaderboard. Update when a model is actually run, never from a
# paper's abstract.
MEASURED = {
    # held-out set: 200 compositions drawn with a different seed from the
    # ones these models were compared on, so the choice of ORB is not also
    # scored on the pairs that made it
    "orb":    {"ranking": 0.895, "gap_mae": 0.0124},
    "m3gnet": {"ranking": 0.778, "gap_mae": 0.0238},
}

DEFAULT = "orb"


def load(name: str = DEFAULT, **kwargs) -> Potential:
    """Load a potential by name. This is the only place DAO-P is chosen."""
    if name not in REGISTRY:
        raise ValueError("unknown potential {!r}; have {}".format(
            name, sorted(REGISTRY)))
    return REGISTRY[name](**kwargs)
