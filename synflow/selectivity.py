"""Will the TARGET form, or something else first? Thermodynamic selectivity.

WHY "DOWNHILL" WAS NOT ENOUGH
-----------------------------
On the A-Lab's 57 targets the simulator called every recipe feasible -- the
33 the robot made and the 13 unseen ones it could not. That is no accident:
the A-Lab chose targets that are stable, so "is the target reaction
downhill" is yes for all of them and separates nothing. The A-Lab's own
analysis of its failures points elsewhere: the precursors reacted first
into OTHER phases, which used up the driving force the target needed.

THE METRIC
----------
Primary competition, after McDermott et al. (arXiv 2308.11816):

    C1 = dG(target) - min(0, min_i dG(competing_i))      eV per atom

Every dG is per atom of reactants, at the firing temperature, with the gas
term. A competing reaction is any balanced reaction of the same precursors
-- all of them or a subset -- to one other phase of the chemical system.
C1 < 0: making the target releases more energy per atom than anything else
the precursors could do. C1 > 0: some other product is more favourable, and
that is where impurities and failures are expected. Lower is better.

WHAT IS SIMPLER HERE THAN IN THE PAPER
--------------------------------------
* Solids carry no entropy; the paper estimates it with a machine-learned
  Gibbs descriptor. Only the gases' entropy is counted, as in reaction.py.
* Competing products are every formula Materials Project holds in the
  system (lowest-energy polymorph each), not only those on the hull. A
  metastable product can only make C1 look worse, never better, and only
  when its energy is close to the hull anyway.
* Secondary competition (C2, the target reacting on with leftover
  precursors) is not computed.

All energies come from one fetch of the whole chemical system -- the same
table reaction.py uses -- so every reaction is on one energy scale.
"""

from itertools import combinations
from typing import Dict, List, Optional, Sequence

from . import reaction

_LIGHT = frozenset({"O", "C", "H", "N"})


def _metals(formula: str) -> frozenset:
    from pymatgen.core import Composition
    return frozenset(str(e) for e in Composition(formula)) - _LIGHT


def reaction_free_energy(reactants: Sequence[str], product: str,
                         temperature_C: float, table: Dict) -> Optional[dict]:
    """dG per atom of reactants for reactants -> product (+ gases), or None.

    None when it cannot be balanced, when an entry is missing, or when the
    balance would make a "reactant" come out as a product -- a reaction of
    these precursors must consume them.
    """
    from pymatgen.analysis.reaction_calculator import ComputedReaction
    from pymatgen.core import Composition

    gases = sorted(set(reaction.released_gases(reactants)) | {"O2"})
    names = [Composition(f).reduced_formula for f in reactants]
    if any(n not in table for n in names) \
            or Composition(product).reduced_formula not in table:
        return None
    gas_entries = [table[Composition(g).reduced_formula] for g in gases
                   if Composition(g).reduced_formula in table]
    try:
        rxn = ComputedReaction([table[n] for n in names],
                               [table[Composition(product).reduced_formula]]
                               + gas_entries)
    except Exception:                                           # noqa: BLE001
        return None
    # A reactant the balance does not need gets a coefficient of about
    # -3e-16, not 0, and a test of "< 0" counted it as consumed -- the same
    # reaction then appeared again under a larger set of precursors. The
    # energies were unaffected; the duplicates were not real reactions.
    for name in names:
        if rxn.get_coeff(Composition(name).reduced_composition) > -1e-8:
            return None                      # not consumed: not a reaction
    target_comp = Composition(product).reduced_composition
    if rxn.get_coeff(target_comp) < 1e-8:
        return None
    gas_moles = reaction.gas_coefficients(rxn)
    dG = (rxn.calculated_reaction_energy
          + reaction.entropy_term(gas_moles, temperature_C))
    atoms = sum(-rxn.get_coeff(c) * c.num_atoms for c in rxn.all_comp
                if rxn.get_coeff(c) < 0)
    if atoms <= 0:
        return None
    return {"reaction": str(rxn), "dG_eV_per_atom": dG / atoms,
            "gases": gas_moles}


def primary_competition(precursors: Sequence[str], target: str,
                        temperature_C: float, api_key: Optional[str] = None,
                        table: Optional[Dict] = None,
                        max_subset: int = 3) -> dict:
    """C1 for one recipe, and the competing reaction that sets it.

    `table` maps reduced formula -> lowest-energy ComputedEntry for the whole
    chemical system; fetched once from Materials Project if not given.
    Subsets of up to `max_subset` precursors are tried as competitors, since
    solid-state reactions start between pairs.
    """
    from pymatgen.core import Composition

    elements = set()
    for formula in list(precursors) + [target]:
        elements |= {str(e) for e in Composition(formula)}
    for gas in set(reaction.released_gases(precursors)) | {"O2"}:
        elements |= {str(e) for e in Composition(gas)}
    if table is None:
        table = reaction._system_ground_states("-".join(sorted(elements)),
                                               api_key)

    target_rxn = reaction_free_energy(precursors, target, temperature_C,
                                      table)
    if target_rxn is None:
        return {"C1_eV_per_atom": None,
                "error": "target reaction not available"}

    excluded = {Composition(f).reduced_formula
                for f in list(precursors) + [target]}
    excluded |= {Composition(g).reduced_formula
                 for g in reaction.GAS_ENTROPY}
    all_metals = frozenset().union(*(_metals(p) for p in precursors))
    # A competing product may hold only the target's elements and oxygen.
    # The first Colab run named MnPNO5 as the rival of Mn2VPO7 -- nitrogen
    # from NH4H2PO4 kept in a solid, when at 900 C in air it has long since
    # left as ammonia. Carbon, hydrogen and nitrogen in a precursor are
    # there to leave; the impurities a chemist finds are in the target's
    # own system. This also cuts the search several-fold.
    allowed = {str(e) for e in Composition(target)} | {"O"}

    # PAIRS AND UP, NOT ONE PRECURSOR ON ITS OWN. The first full run let a
    # single precursor's decomposition compete: CaCO3 -> CaO + CO2 was the
    # "rival" of five targets, Ag2O -> Ag of another. That is calcination,
    # a step every carbonate route takes on the way, and the CaO it leaves
    # still reacts. Competition is the precursors reacting with EACH OTHER
    # into a phase that is not the target.
    best: Optional[dict] = None
    tried = 0
    for size in range(2, min(max_subset, len(precursors)) + 1):
        for subset in combinations(sorted(precursors), size):
            subset_metals = frozenset().union(*(_metals(p) for p in subset))
            for formula in table:
                if formula in excluded:
                    continue
                if not {str(e) for e in Composition(formula)} <= allowed:
                    continue
                metals = _metals(formula)
                # every precursor in the subset must contribute a metal, and
                # the product may hold only metals the subset supplies
                if not metals or not metals <= subset_metals \
                        or not metals <= all_metals:
                    continue
                if any(not (_metals(p) & metals) for p in subset):
                    continue
                competing = reaction_free_energy(list(subset), formula,
                                                 temperature_C, table)
                tried += 1
                if competing is None:
                    continue
                if best is None or competing["dG_eV_per_atom"] \
                        < best["dG_eV_per_atom"]:
                    best = dict(competing, product=formula,
                                precursors=list(subset))

    floor = min(0.0, best["dG_eV_per_atom"]) if best else 0.0
    remaining = (remaining_driving_force(best, precursors, target,
                                         temperature_C, table)
                 if best and best["dG_eV_per_atom"] < 0 else None)
    return {
        "C1_eV_per_atom": target_rxn["dG_eV_per_atom"] - floor,
        "target_dG_eV_per_atom": target_rxn["dG_eV_per_atom"],
        "target_reaction": target_rxn["reaction"],
        "strongest_competitor": best,
        "remaining": remaining,
        "competitors_tried": tried,
        "temperature_C": temperature_C,
    }


def remaining_driving_force(rival: dict, precursors: Sequence[str],
                            target: str, temperature_C: float,
                            table: Dict) -> Optional[dict]:
    """What is left to drive the target once the strongest rival has formed.

    The A-Lab's own account of its failures (Szymanski et al. 2023): the
    precursors first react into intermediates, and when the step from those
    intermediates to the target has under ~50 meV/atom left, it stalls --
    the energy went into making the intermediate. C1 asks whether a rival is
    favoured; this asks whether the target can still be reached from it.

    The rival plus the fewest further precursors that balance to the
    target; dG per atom of those reactants, at the firing temperature.
    """
    product = rival["product"]
    pool = sorted(set(precursors))
    for size in range(0, len(pool) + 1):
        for extra in combinations(pool, size):
            step = reaction_free_energy([product] + list(extra), target,
                                        temperature_C, table)
            if step is not None:
                return dict(step, from_=[product] + list(extra))
    return None


def explain(result: dict) -> str:
    """Plain words for the protocol."""
    c1 = result.get("C1_eV_per_atom")
    if c1 is None:
        return "selectivity not computed: {}".format(result.get("error"))
    rival = result.get("strongest_competitor")
    if c1 < 0:
        return ("selective: making the target releases {:.0f} meV/atom more "
                "than any competing reaction".format(-1000 * c1))
    return ("NOT selective: {} -> {} releases {:.0f} meV/atom more than the "
            "target reaction; expect it as an impurity or an intermediate "
            "that stalls the synthesis".format(
                " + ".join(rival["precursors"]), rival["product"],
                1000 * c1))
