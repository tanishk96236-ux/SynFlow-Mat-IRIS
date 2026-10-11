"""Will the recipe actually work? Thermodynamics of a solid-state reaction.

WHY THIS IS THE PIECE THAT MAKES DAO-S CHECKABLE
------------------------------------------------
DAO-S predicts precursors and a firing temperature. Until now nothing tested
them. It says "make KRhO2 from K2CO3 and Rh2O3 at 772 °C" and the only
evidence that this is sensible is that the network learned it from 19,488
recipes where chemists did similar things.

This computes whether the reaction it proposes is downhill, whether something
else would form instead, and whether the temperature it chose is the
temperature the reaction actually needs. Three questions DAO-S cannot answer
about its own output.

It is also the layer MatterGen and GNoME do not have. GNoME published 380,000
structures predicted stable and nobody can make most of them; a recipe plus a
feasibility check is a different kind of output from a structure.

THE TERM THAT DECIDES EVERYTHING, AND OMITTING IT WOULD LOOK FINE
-----------------------------------------------------------------
    K2CO3 + Rh2O3  ->  2 KRhO2 + CO2(gas)

At 0 K this is uphill. Every energy in this repository is a 0 K enthalpy, so a
naive calculation would call it impossible -- and would call every carbonate
route impossible, which is most of solid-state synthesis.

It works because the CO2 leaves. S°(CO2) is 213.8 J/mol/K, so at 1045 K the
-T*dS term is about 2.3 eV per CO2 released. That is enormous next to typical
reaction enthalpies, and it is precisely why carbonates are the precursor of
choice: the entropy of the escaping gas pays for the reaction.

So the gas term is not a refinement here. Leave it out and the answer is
confidently wrong for the majority of real recipes.

WHAT THIS IS NOT
----------------
Not a virtual furnace. Time, particle size, diffusion and nucleation are
absent, and they decide real outcomes -- a reaction can be downhill and still
not happen in twelve hours. Simulating that is an open research area
(cellular-automaton phase evolution, diffusion-limited selectivity) and is not
attempted here.

What this gives is the thermodynamic gate: a reaction that fails it will not
work, and one that passes it might.
"""

import functools
import os
import re
from typing import Dict, List, Optional, Sequence

# Standard molar entropies at 298 K, J/(mol K), for the gases a solid-state
# route actually releases. The dominant term in any reaction that evolves one.
GAS_ENTROPY = {
    "CO2": 213.8,
    "O2": 205.2,
    "H2O": 188.8,
    "NH3": 192.8,
    "N2": 191.6,
    "CO": 197.7,
    "SO2": 248.2,
    "NO2": 240.0,
    "Cl2": 223.1,
    "HCl": 186.9,
}

# J/mol -> eV per formula unit
J_PER_MOL_TO_EV = 1.0 / 96485.0

# Which precursors release which gas when they decompose. Carbonates give CO2,
# nitrates NO2 and O2, hydroxides water, ammonium salts ammonia and water.
# This is what makes the entropy term computable without a full gas-phase
# treatment.
DECOMPOSITION_GAS = {
    "CO3": "CO2",
    "NO3": "NO2",
    "OH": "H2O",
    "NH4": "NH3",
    "C2O4": "CO2",       # oxalate
}


def released_gases(precursors: Sequence[str]) -> Dict[str, float]:
    """Which gases leave, and how many per formula unit of each precursor.

    Counted from the precursor formulas rather than balanced properly, because
    the point is the entropy term and it scales with the number of gas
    molecules. K2CO3 gives one CO2; Ca(NO3)2 gives two NO2 and an oxygen.
    """
    from pymatgen.core import Composition

    gases: Dict[str, float] = {}
    for formula in precursors:
        try:
            composition = Composition(formula)
        except Exception:                                   # noqa: BLE001
            continue
        counts = {str(el): n for el, n in composition.items()}
        # carbonate and oxalate: every carbon leaves as CO2
        if "C" in counts and "O" in counts:
            gases["CO2"] = gases.get("CO2", 0.0) + counts["C"]
        # Nitrogen leaves as ammonia from an ammonium salt and as NO2 from a
        # nitrate. Which one is decided by whether there is enough hydrogen
        # for NH4.
        hydrogen = counts.get("H", 0.0)
        if "N" in counts:
            if hydrogen >= 3 * counts["N"]:
                gases["NH3"] = gases.get("NH3", 0.0) + counts["N"]
                hydrogen -= 3 * counts["N"]
            else:
                gases["NO2"] = gases.get("NO2", 0.0) + counts["N"]
        # Whatever hydrogen is left leaves as water. NH4H2PO4 has six
        # hydrogens and one nitrogen: three go to ammonia and three to water,
        # and an earlier version dropped the water entirely whenever nitrogen
        # was present -- which understated the entropy term for exactly the
        # phosphate precursors the olivine recipes use.
        if hydrogen > 0:
            gases["H2O"] = gases.get("H2O", 0.0) + hydrogen / 2.0
    return gases


@functools.lru_cache(maxsize=1)
def _gas_names() -> Dict[str, str]:
    """pymatgen reduced formula -> the name GAS_ENTROPY uses ("H3N" -> "NH3")."""
    from pymatgen.core import Composition
    return {Composition(name).reduced_formula: name for name in GAS_ENTROPY}


def gas_coefficients(reaction) -> Dict[str, float]:
    """Moles of each gas in a BALANCED reaction: positive released, negative
    consumed.

    WHY THE BALANCED EQUATION AND NOT THE PRECURSORS
    released_gases() reads gases off the precursor formulas -- carbonates
    give CO2, hydroxides water -- and the entropy term used to be built from
    it alone. Oxygen released or taken up by a change of oxidation state
    appears only in the balanced equation, so it was never counted:

        2 MnO2 -> Mn2O3 + 0.5 O2      gas term 0.000 eV, should be -0.928

    The A-Lab's robots decomposed MnO2 by 600 C; the simulator rejected it
    with dG = +0.866 eV. That +0.866 was the whole enthalpy -- and it matches
    the calorimetric value, +0.867 eV, to the meV. Materials Project was
    right; the simulator was missing the term that pays for it. The same
    omission hit Sb2O5 -> SbO2 and both BaO2 routes, and left reactions that
    CONSUME oxygen (Co3O4 -> LiCoO2) without the penalty they should carry.
    """
    names = _gas_names()
    out: Dict[str, float] = {}
    for composition in reaction.all_comp:
        name = names.get(composition.reduced_formula)
        if name is None:
            continue
        coefficient = float(reaction.get_coeff(composition))
        if abs(coefficient) < 1e-12:
            continue
        # moles of the molecule, whatever multiple the composition was given in
        molecules = coefficient * (composition.num_atoms
                                   / composition.reduced_composition.num_atoms)
        out[name] = out.get(name, 0.0) + molecules
    return out


def entropy_term(gases: Dict[str, float], temperature_C: float,
                 pressures: Optional[Dict[str, float]] = None) -> float:
    """-T*dS for the gases in eV: negative for gas released, positive for gas
    consumed. Each gas at 1 bar unless `pressures` says otherwise.

    At 772 °C one CO2 released contributes about -2.3 eV. A reaction enthalpy
    of +1 eV is comfortably paid for by a single carbonate. Oxygen consumed
    from the furnace atmosphere costs entropy the same way, with the sign
    reversed.
    """
    return -(temperature_C + 273.15) * gas_entropy_per_kelvin(gases, pressures)


# THE PRESSURE A GAS LEAVES INTO
#
# The S° values above are for each gas at 1 bar. A gas that leaves into a
# furnace full of something else is at its own partial pressure, and its
# entropy is higher by -R ln(p / 1 bar):
#
#     CO2 into air, 4.2e-4 bar:   +64.6 J/mol/K    (213.8 -> 278.4)
#     O2 into air,  0.21 bar:     +13.0 J/mol/K    (205.2 -> 218.2)
#
# So 1 bar is the conservative case -- it understates the help a released
# gas gives -- and air is the realistic one for a box furnace. Only the
# gases air actually fixes are given here. Water depends on the weather and
# there is no ammonia in air at all; either would need a number chosen to
# fit, so they stay at 1 bar in both cases.
AIR_BAR = {"N2": 0.7808, "O2": 0.2095, "CO2": 4.2e-4}

GAS_CONSTANT = 8.314462618       # J/(mol K)


def entropy_at(gas: str, pressure_bar: float = 1.0) -> Optional[float]:
    """S of one mole of `gas` at `pressure_bar`, J/(mol K); ideal gas."""
    import math

    standard = GAS_ENTROPY.get(gas)
    if standard is None:
        return None
    return standard - GAS_CONSTANT * math.log(pressure_bar)


def gas_entropy_per_kelvin(gases: Dict[str, float],
                           pressures: Optional[Dict[str, float]] = None
                           ) -> float:
    """dS of the gases, eV/K: positive when there is net gas released.

    `pressures` maps a gas to its partial pressure in bar; a gas not named is
    at 1 bar. An unknown gas contributes nothing rather than a guess.
    """
    total = 0.0
    for gas, moles in gases.items():
        entropy = entropy_at(gas, (pressures or {}).get(gas, 1.0))
        if entropy is not None:
            total += moles * entropy
    return total * J_PER_MOL_TO_EV


def crossover(enthalpy_eV: float, gases: Dict[str, float],
              pressures: Optional[Dict[str, float]] = None) -> dict:
    """Where dG = dH - T*dS changes sign, and which side of it is downhill.

        kind       dH    dS    downhill
        floor      > 0   > 0   above T   (a carbonate paying for its reaction)
        ceiling    < 0   < 0   below T   (an oxide taking up oxygen)
        always    <= 0  >= 0   at every temperature
        never     >= 0  <= 0   at no temperature

    A CEILING IS NEW, AND IT IS REAL
    While the gases came from the precursors, every count was positive and
    there could only be a floor. With oxygen read off the balanced equation
    it can be consumed, and then heating works against the reaction: above
    the ceiling the product gives its oxygen back. That is why a cobaltite
    or a nickelate goes oxygen-deficient if it is fired too hot.

    dS is the gases' alone -- the solids' entropy change is small next to a
    mole of gas and is not counted -- and both dH and S are the 298 K values.
    """
    per_kelvin = gas_entropy_per_kelvin(gases, pressures)
    if per_kelvin > 0 and enthalpy_eV > 0:
        return {"kind": "floor",
                "temperature_C": float(enthalpy_eV / per_kelvin - 273.15)}
    if per_kelvin < 0 and enthalpy_eV < 0:
        return {"kind": "ceiling",
                "temperature_C": float(enthalpy_eV / per_kelvin - 273.15)}
    if enthalpy_eV <= 0 and per_kelvin >= 0:
        return {"kind": "always", "temperature_C": None}
    return {"kind": "never", "temperature_C": None}


@functools.lru_cache(maxsize=32)
def _ground_states(chemsys: str, wanted: tuple,
                   api_key: Optional[str] = None) -> dict:
    """Lowest-energy entry for each wanted formula, cached.

    WHY THIS CACHES THE ANSWER AND NOT THE QUERY
    --------------------------------------------
    A reaction needs one entry per formula it mentions -- five to eight of
    them. The query that supplies them returns the whole chemical system, and
    that is thousands of entries: one Ba-Ti-O-C call came back with 1,215.

    Two earlier versions cached the whole system. Both killed the process with
    no traceback and exit code 4, once partway through the scrambled set and
    once partway through the real one. The first also asked for structures,
    via hull._entries_for, which made it worse; removing them was not enough,
    because the problem is the count and not the attachments. Holding dozens
    of full chemical systems is simply more than this machine has.

    The fix is to cache what is used rather than what is fetched. The full
    list stays local and is freed when this returns; what survives is a dict
    of at most a handful of entries.

    The whole system still has to be FETCHED, even though almost none of it is
    kept. get_entries_in_chemsys runs the GGA/r2SCAN mixing scheme across the
    system to put every entry on one energy scale, and asking formula by
    formula would hand back energies that are individually right and not
    comparable to each other -- which is the kind of error that produces a
    confident wrong reaction energy rather than a crash.
    """
    from pymatgen.core import Composition

    table = _system_ground_states(chemsys, api_key)
    targets = {Composition(f).reduced_formula for f in wanted}
    return {f: e for f, e in table.items() if f in targets}


@functools.lru_cache(maxsize=2)
def _system_ground_states(chemsys: str,
                          api_key: Optional[str] = None) -> dict:
    """The lowest-energy entry of EVERY formula in one chemical system.

    Fetching the system is the slow step: eight elements (Na-Ca-Fe-P-O-C-H-N
    for an A-Lab phosphate) took minutes, and a target is assessed twice --
    with our precursors and with the robot's -- usually in the same system.
    Keyed on the wanted formulas, the second call fetched it all again.

    What is kept is one entry per formula, not the polymorphs, and only for
    the last two systems -- the memory lesson in _ground_states still holds;
    two small tables are not dozens of full systems.
    """
    import pickle

    from mp_api.client import MPRester

    # ON DISK TOO. An eight-element system takes minutes to fetch, and the
    # selectivity evaluation fetched the same 57 systems three times in two
    # days. The table is one entry per formula, small; it lives under runs/
    # (not committed) and is keyed by the system. Delete the folder to force
    # a fresh fetch after a Materials Project database release.
    path = os.path.join(SYSTEM_CACHE, chemsys + ".pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as handle:
                return pickle.load(handle)
        except Exception:                                       # noqa: BLE001
            pass                    # unreadable: fetch again and overwrite

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise SystemExit("MP_API_KEY is not set")

    best: dict = {}
    with MPRester(key) as mpr:
        for entry in mpr.get_entries_in_chemsys(chemsys.split("-")):
            formula = entry.composition.reduced_formula
            if (formula not in best
                    or entry.energy_per_atom < best[formula].energy_per_atom):
                best[formula] = entry
    # Written to a file named for this process, then renamed into place:
    # two workers fetching the same system must not write one file at once.
    os.makedirs(SYSTEM_CACHE, exist_ok=True)
    temporary = "{}.{}.tmp".format(path, os.getpid())
    with open(temporary, "wb") as handle:
        pickle.dump(best, handle)
    os.replace(temporary, path)
    return best


SYSTEM_CACHE = os.path.join("runs", "cache", "systems")


def _novel_entry(structure, critic, api_key: Optional[str] = None):
    """A ComputedEntry for a compound Materials Project does not have.

    Built the same way hull.py places a candidate: the potential scores the
    structure, that becomes a formation energy against the pure elements, and
    MP's own elemental energies put it back on MP's absolute scale. The
    calibration -- and its measured uncertainty -- is hull's, not a second
    one invented here.
    """
    from pymatgen.entries.computed_entries import ComputedEntry

    from synflow import hull

    elements = sorted({str(site.specie) for site in structure})
    diagram, entries = hull.phase_diagram(elements, api_key)
    mp_refs, orb_refs = hull._references(diagram, entries, critic)

    absent = [el for el in elements if el not in orb_refs]
    if absent:
        raise ValueError("no scorable elemental reference for {}".format(
            ", ".join(absent)))

    correction = hull.calibrate(diagram, entries, critic, orb_refs, sample=30)
    fitted = correction["calibration_per_element"]
    fractions = structure.composition.fractional_composition
    bias = sum(fractions.get(el, 0.0) * c for el, c in fitted.items())

    formation = hull.formation_energy(structure, critic, orb_refs) - bias
    on_mp_scale = formation + sum(fractions.get(el, 0.0) * mp_refs[el]
                                  for el in elements)
    # The calibration's own scatter travels with the entry. It used to be
    # dropped here, so a novel target's reaction came out as a bare number --
    # KRhO2 at dG -0.09 eV with no hint that the potential alone is uncertain
    # by more than that.
    return ComputedEntry(structure.composition,
                         on_mp_scale * len(structure),
                         data={"calibration_error_eV_per_atom":
                               float(correction["calibration_error"]),
                               "n_calibration": correction["n_calibration"]})


# The anchor's error, eV/atom, measured by scripts/validate_anchoring.py
# (2026-10-09) on 41 A-Lab targets Materials Project has, each treated as
# unknown -- its own entries removed, its structure relaxed by ORB:
#
#                       median |err|   MAE (40)   RMS (40)   closer on
#     anchored             0.0068      0.0109     0.0220      30 / 40
#     calibrated           0.0352      0.0480        --       10 / 40
#
# The RMS, not the MAE, is used, so the worst cases (Mo(PO3)5 at +0.124) set
# the band rather than averaging away. One compound, BaGdCrFeO6, is off by
# ~3 eV/atom under BOTH methods -- a defect of its MP entry relative to its
# neighbours, not of either method -- and is left out of these numbers and
# named here instead. The calibration's own STATED error averaged 0.117,
# against a real 0.048: the old bands were pessimistic as well as wide.
ANCHOR_ERROR_EV: Optional[float] = 0.0220


def _anchored_entry(structure, critic, api_key: Optional[str] = None,
                    chemsys: Optional[str] = None):
    """A novel target anchored to its decomposition phases (hull.anchor).

    The anchor phases' energies are taken from the reaction's own table
    when it has them, so the target and the precursors sit on one MP
    scale; hull's fetch of the target's subsystem supplies the rest.
    Returns None when anchoring is not validated, or not possible.
    """
    if ANCHOR_ERROR_EV is None:
        return None
    from pymatgen.entries.computed_entries import ComputedEntry

    from synflow import hull

    elements = sorted({str(site.specie) for site in structure})
    diagram, _ = hull.phase_diagram(elements, api_key)
    anchored = hull.anchor(structure, critic, diagram)
    if anchored is None:
        return None
    table = _system_ground_states(chemsys, api_key) if chemsys else {}
    energy = anchored["gap_eV_per_atom"]
    for formula, fraction in anchored["anchors"]:
        source = table.get(formula)
        if source is None:
            energy = anchored["energy_per_atom"]       # hull's scale whole
            break
        energy += fraction * source.energy_per_atom
    return ComputedEntry(structure.composition,
                         energy * structure.composition.num_atoms,
                         data={"calibration_error_eV_per_atom":
                               ANCHOR_ERROR_EV,
                               "anchors": anchored["anchors"],
                               "hull_gap_eV_per_atom":
                                   anchored["gap_eV_per_atom"]})


def polymorph_gap(structure, formula: str, critic,
                  api_key: Optional[str] = None,
                  relax_steps: int = 80) -> Optional[dict]:
    """How far the given structure sits above the known phase of its formula.

    BOTH ARE SCORED BY THE POTENTIAL. Comparing the potential's energy for
    the new structure with Materials Project's DFT energy for the known one
    would carry the calibration scatter between the two methods -- 0.11 to
    0.16 eV/atom in hull.py, larger than most polymorph gaps. Scored by the
    same potential, that error cancels: on our polymorph benchmark ORB-v3
    ranks 89.5% of pairs correctly with a gap MAE of 0.0124 eV/atom
    (synflow/dao_p.py). The known structure is relaxed the same way as the
    generated one before it is scored.
    """
    from mp_api.client import MPRester

    from synflow import dao_g

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        return None
    try:
        with MPRester(key) as mpr:
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=["material_id", "structure", "energy_above_hull"])
    except Exception:                                           # noqa: BLE001
        return None
    docs = [d for d in docs if d.structure is not None]
    if not docs:
        return None
    ground = min(docs, key=lambda d: d.energy_above_hull or 0.0)
    known = dao_g.relax(ground.structure, critic, steps=relax_steps)
    gap = float(critic.score(structure)) - float(critic.score(known))
    return {"known_material_id": str(ground.material_id),
            "gap_eV_per_atom": gap,
            "method": "ORB-v3 on both structures, both relaxed"}


class NotAllMeasured(ValueError):
    """A species in the reaction has no measured formation enthalpy."""

    def __init__(self, formulas):
        self.formulas = list(formulas)
        super().__init__("not measured: {}".format(", ".join(self.formulas)))


def reaction_enthalpy(precursors: Sequence[str], target: str) -> dict:
    """Hess's law on MEASURED formation enthalpies -- every species, or none.

        dH = sum over products of n dHf  -  sum over reactants of n dHf

    Gases from NIST-JANAF, solids from calorimetry (Wang et al. 2021). Raises
    NotAllMeasured if any species has no measurement; assess then takes the
    whole reaction from Materials Project instead.

    ONE SOURCE PER REACTION, BECAUSE MIXING WAS MEASURED TO FAIL
    The first version filled each gap species by species -- measured where
    it could, MP's per-compound formation energy elsewhere -- on the claim
    that both sit on one scale. Checked on 2026-10-08, they do not:

        TiO2   measured -3.261 eV/atom   MP summary -3.508   (0.25 off)
        Ta2O5  measured -3.030           MP summary -3.347   (0.32 off)

    and a reaction is a small difference of large numbers, so offsets that
    size do not cancel when the two sides come from different sources:

        CuO + Ta2O5 -> CuTa2O6    mixed  -2.147 eV    all-MP  +0.338 eV

    -2.1 eV for an oxide + oxide reaction is not chemistry. Within one
    source the offsets do cancel -- MP's entries are mixed onto one scale by
    get_entries_in_chemsys, and where every species is measured the two
    agree: MnO2 -> Mn2O3 +0.867 against +0.866, BaTiO3 +1.247 against +1.171.

    WHAT THIS MEANS FOR A NOVEL TARGET
    A generated material has no calorimetry, so its reaction is never all
    measured and always goes to the MP path, where reaction_energy places it
    with the potential. Measured values serve the known compounds: the
    A-Lab rows and the controls, as a check on MP.

    An earlier docstring here said MP's enthalpy for 2 MnO2 -> Mn2O3 + 0.5 O2
    was 0.93 eV too high. It was not; the 0.93 eV was the oxygen's entropy,
    which the simulator had never counted -- see gas_coefficients.
    """
    import math

    from pymatgen.analysis.reaction_calculator import Reaction
    from pymatgen.core import Composition

    from synflow import thermo_data

    gas_products = sorted(set(released_gases(precursors)) | {"O2"})
    reactants = [Composition(f) for f in precursors]
    # The target as its REDUCED formula, the way ComputedReaction stores it.
    # Given as written, "Mn2P4O12" sat in the reaction and normalize_to was
    # asked for MnP2O6, which is not in it -- every A-Lab Mn2P4O12 row came
    # back "could not balance".
    reduced_target = Composition(target).reduced_composition
    products = [reduced_target] + [Composition(g) for g in gas_products]
    try:
        reaction = Reaction(reactants, products)
        reaction.normalize_to(reduced_target, 1.0)
    except Exception as error:                              # noqa: BLE001
        raise ValueError("could not balance {} -> {}: {}".format(
            " + ".join(precursors), target, error))

    present = [(c, float(reaction.get_coeff(c))) for c in reaction.all_comp
               if abs(float(reaction.get_coeff(c))) >= 1e-9]
    records = {c.reduced_formula: thermo_data.formation_enthalpy(
        c.reduced_formula) for c, _ in present}
    missing = sorted(f for f, record in records.items() if record is None)
    if missing:
        raise NotAllMeasured(missing)

    sources, enthalpy, variance, unstated = {}, 0.0, 0.0, []
    for composition, coefficient in present:
        formula = composition.reduced_formula
        record = records[formula]
        per_formula = record["per_atom"] * composition.num_atoms
        enthalpy += coefficient * per_formula
        # An error bar that was not stated is unknown, not zero. 301 of the
        # 2,135 measured entries have none -- Fe2O3 among them -- and the
        # dataset stores that as NaN, which is truthy: the first version
        # tested `if uncertainty:` and printed YFeO3's error as "nan".
        error = record.get("uncertainty")
        if error is None or not math.isfinite(error):
            unstated.append(formula)
        elif error > 0:
            variance += (coefficient * error * composition.num_atoms) ** 2
        sources[formula] = record["source"]

    return {
        "reaction": str(reaction),
        "enthalpy_eV": float(enthalpy),
        "enthalpy_eV_per_atom": float(
            enthalpy / max(reduced_target.num_atoms, 1)),
        "target_coefficient": 1.0,
        "gases": gas_coefficients(reaction),
        # from the species that state one; those that do not are listed, so
        # the number is read as a lower bound when the list is not empty
        "enthalpy_uncertainty_eV": float(variance ** 0.5) if variance else None,
        "uncertainty_not_stated_for": unstated,
        "enthalpy_sources": sources,
        "enthalpy_source": "measured, every species",
    }


def reaction_energy(precursors: Sequence[str], target: str,
                    critic=None, api_key: Optional[str] = None,
                    structure=None) -> dict:
    """Balance the reaction and compute its energy, per atom of target.

    Enthalpies come from Materials Project, not from the potential. That is
    deliberate: MP's formation energies carry its anion corrections, and the
    carbonates and nitrates a recipe uses are exactly the compounds where an
    uncorrected GGA energy is worst. The potential is used elsewhere, on the
    candidate itself, where it was measured.
    """
    from pymatgen.analysis.reaction_calculator import ComputedReaction
    from pymatgen.core import Composition

    # THE GASES ARE PRODUCTS, AND LEAVING THEM OUT MEANS NOTHING BALANCES
    #
    # A reaction written as
    #
    #     Li2CO3 + Fe2O3 + NH4H2PO4  ->  LiFePO4
    #
    # cannot balance: the carbon, nitrogen and hydrogen have nowhere to go.
    # The first version of this passed only the target as a product and every
    # carbonate or ammonium recipe failed with "could not balance", which
    # reads as a bad recipe rather than as a bad reaction equation.
    #
    # Oxygen is included as a possible product whether or not a gas is
    # detected, because the precursors and the target often differ in
    # oxidation state. Fe2O3 is Fe(3+) and LiFePO4 is Fe(2+), so making one
    # from the other releases oxygen -- which is also why LiFePO4 is fired
    # under inert or reducing atmosphere in practice.
    gas_products = sorted(set(released_gases(precursors)) | {"O2"})

    wanted = list(precursors) + [target] + gas_products
    elements = set()
    for formula in wanted:
        elements |= {str(el) for el in Composition(formula)}

    # Ground states only, and only for the formulas this reaction mentions.
    # The filtering happens inside the cache rather than here, so the rest of
    # the chemical system is never held -- see _ground_states.
    best = _ground_states("-".join(sorted(elements)), tuple(wanted), api_key)
    if not best:
        raise ValueError("no Materials Project entries for {}".format(
            "-".join(sorted(elements))))

    missing, chosen = [], {}
    for formula in wanted:
        reduced = Composition(formula).reduced_formula
        if reduced not in best:
            missing.append(formula)
        else:
            chosen[formula] = best[reduced]

    # THE TARGET IS NOVEL, SO IT IS NOT IN THE DATABASE -- THAT IS THE POINT
    #
    # Taking every energy from Materials Project works for precursors, which
    # are ordinary reagents, and fails for exactly the compounds this tool
    # exists to judge. The first run raised "Materials Project has no entry
    # for KRhO2" on our own best candidate.
    #
    # The structure is what we have instead, and hull.py already solves this
    # problem: score it with the potential, convert to a formation energy
    # against the pure elements -- where the potential is good -- and add MP's
    # own elemental energies back to land on MP's scale. The same calibration
    # and the same measured uncertainty, reused rather than reinvented.
    #
    # AND WHEN THE FORMULA IS KNOWN, THE STRUCTURE STILL HAS TO BE SCORED
    #
    # The first version scored the structure only when MP had no entry for
    # the formula. A generated NEW POLYMORPH of a known formula -- a BaTiO3
    # that is not perovskite -- was then judged as MP's BaTiO3: the verdict
    # described a material the input was not. With a structure given, its own
    # energy is used, and its height above the known ground state is reported:
    # a synthesis that can reach a lower polymorph usually does.
    #
    # ANCHORED WHERE IT CAN BE. When the formula is known, the structure's
    # energy is MP's DFT energy for the known phase PLUS the potential's gap
    # between the two structures. The potential is then trusted only for a
    # difference between two structures of one composition -- 0.0124 eV/atom
    # MAE on our benchmark -- instead of for an absolute energy through
    # hull's calibration, whose scatter is 0.11-0.16 eV/atom. Ten times
    # tighter, for every generated polymorph of a known formula.
    from_potential = False
    polymorph = None
    if structure is not None and critic is not None:
        from pymatgen.entries.computed_entries import ComputedEntry

        known = best.get(Composition(target).reduced_formula)
        if known is not None:
            polymorph = polymorph_gap(structure,
                                      Composition(target).reduced_formula,
                                      critic, api_key)
        if polymorph is not None:
            chosen[target] = ComputedEntry(
                structure.composition,
                (known.energy_per_atom + polymorph["gap_eV_per_atom"])
                * structure.composition.num_atoms,
                data={"calibration_error_eV_per_atom": ORB_GAP_MAE_EV})
            polymorph["energy_from"] = "MP known phase + potential's gap"
        else:
            chosen[target] = (_anchored_entry(structure, critic, api_key,
                                              chemsys="-".join(
                                                  sorted(elements)))
                              or _novel_entry(structure, critic, api_key))
        if target in missing:
            missing.remove(target)
        from_potential = True

    essential = [f for f in missing if f in list(precursors) + [target]]
    if essential:
        raise ValueError(
            "Materials Project has no entry for {}{}".format(
                ", ".join(essential),
                "" if structure is not None else
                " -- pass the structure and the critic to score it instead"))

    reactants = [chosen[f] for f in precursors]
    products = [chosen[target]] + [chosen[g] for g in gas_products
                                   if g in chosen]
    try:
        reaction = ComputedReaction(reactants, products)
    except Exception as error:                              # noqa: BLE001
        raise ValueError("could not balance {} -> {}: {}".format(
            " + ".join(precursors), target, error))

    # ComputedReaction stores REDUCED compositions, so asking it for the
    # coefficient of an entry's own composition fails whenever Materials
    # Project holds a multi-formula-unit cell. Its LiFePO4 entry is 28 atoms,
    # Li4Fe4P4O16, and the lookup raised "Composition('Li4 Fe4 P4 O16') is not
    # in list" -- which reads as a missing entry rather than a units mismatch.
    n_target = abs(reaction.get_coeff(
        Composition(target).reduced_composition))
    atoms = Composition(target).reduced_composition.num_atoms
    per_atom = reaction.calculated_reaction_energy / max(n_target * atoms, 1)

    # The potential's error on the target, carried into the reaction: the
    # calibration's scatter per atom, times the target atoms the reaction
    # makes. Precursors come from MP and share its error, which
    # error_band_meV_per_atom adds separately.
    target_error = None
    if from_potential:
        per = chosen[target].data.get("calibration_error_eV_per_atom")
        if per is not None:
            target_error = float(per * atoms * n_target)

    return {
        "reaction": str(reaction),
        "enthalpy_eV": float(reaction.calculated_reaction_energy),
        "enthalpy_eV_per_atom": float(per_atom),
        "target_coefficient": float(n_target),
        # on the same basis as the enthalpy: this reaction, as balanced
        "gases": gas_coefficients(reaction),
        "target_from_potential": from_potential,
        "target_energy_error_eV": target_error,
        "polymorph": polymorph,
        # For an anchored novel target: its height above the known phases
        # it would fall apart into -- its own stability, which "is the
        # reaction downhill" never asks. MatterGen's Mn3NiO6 came out
        # downhill from Mn2O3 + Ni2O3 and the lab got Mn2NiO4 instead.
        "target_stability": (
            {"above_hull_eV_per_atom":
                chosen[target].data["hull_gap_eV_per_atom"],
             "decomposes_to": chosen[target].data.get("anchors")}
            if from_potential
            and "hull_gap_eV_per_atom" in chosen[target].data else None),
    }


_HYDRATE = re.compile(r"^(.+?)[·.*](\d*\.?\d*)H2O$")


def readable(formula: str) -> str:
    """A corpus precursor name as a formula pymatgen can read.

    The corpus writes hydrates with a middle dot and phases with a Greek
    prefix -- "FeC2O4·2H2O", "α-Al2O3" -- and pymatgen reads neither: five
    of the first 39 real recipes in validate_simulator came back "invalid
    formula" for that alone. A hydrate becomes FeC2O4(H2O)2, so its water is
    weighed and released like any other. The phase prefix is dropped for the
    THERMODYNAMICS only: the energy is the ground state's either way, and
    recipes.py keeps the prefix wherever a phase is a real choice.
    """
    from .recipes import normalise_precursor

    text = normalise_precursor(formula, strip_phase=True)
    match = _HYDRATE.match(text)
    if match:
        base, count = match.groups()
        text = "{}(H2O){}".format(base, count or "1")
    return text


def assess(precursors: Sequence[str], target: str, temperature_C: float,
           critic=None, api_key: Optional[str] = None, structure=None,
           enthalpies: str = "measured") -> dict:
    """The whole check: is the reaction downhill at the temperature proposed?

    Returns the enthalpy, the gas-entropy term at that temperature, their
    sum, and the temperature at which the sum crosses zero -- which is the
    number to compare against what DAO-S chose.

    enthalpies="measured" (the default) uses calorimetry when EVERY species
    in the reaction has been measured, and otherwise takes the whole
    reaction from Materials Project -- never a mixture; reaction_enthalpy
    says why. enthalpies="mp" always uses Materials Project, so the two can
    be compared on the same reactions.
    """
    result = {"target": target, "precursors": list(precursors),
              "temperature_C": float(temperature_C),
              "enthalpies": enthalpies}
    precursors = [readable(p) for p in precursors]
    target = readable(target)
    try:
        energy = None
        # A given structure is the material being judged, and calorimetry
        # describes the KNOWN phase of that formula -- so with a structure the
        # reaction is always taken from MP plus the structure's own energy.
        scored = structure is not None and critic is not None
        if enthalpies != "mp" and not scored:
            try:
                energy = reaction_enthalpy(precursors, target)
            except NotAllMeasured as gap:
                result["not_measured"] = gap.formulas
        if energy is None:
            energy = reaction_energy(precursors, target, critic, api_key,
                                     structure=structure)
            energy["enthalpy_source"] = "Materials Project, every species"
    except Exception as error:                              # noqa: BLE001
        result["feasible"] = None
        result["error"] = "{}: {}".format(type(error).__name__,
                                          str(error)[:120])
        return result
    result.update(energy)

    # Gases from the balanced equation, signed: + released, - consumed.
    # Not from released_gases(precursors), which misses oxygen exchanged by a
    # change of oxidation state and assumes one of each precursor -- see
    # gas_coefficients for the five A-Lab rejections that caused.
    gases = result["gases"]
    result["entropy_term_eV"] = entropy_term(gases, temperature_C)
    result["free_energy_eV"] = (result["enthalpy_eV"]
                                + result["entropy_term_eV"])
    result["free_energy_eV_per_atom"] = (
        result["free_energy_eV"]
        / max(result["target_coefficient"]
              * __import__("pymatgen.core", fromlist=["core"])
              .Composition(target).reduced_composition.num_atoms, 1))

    crossing = crossover(result["enthalpy_eV"], gases)
    result["onset_temperature_C"] = (
        crossing["temperature_C"] if crossing["kind"] == "floor"
        else float("inf") if crossing["kind"] == "never"
        else None)                          # "always" and "ceiling"
    result["ceiling_temperature_C"] = (
        crossing["temperature_C"] if crossing["kind"] == "ceiling" else None)

    result["feasible"] = result["free_energy_eV"] < 0

    # The same reaction in air, where CO2 leaves at 4.2e-4 bar and O2 is at
    # 0.21 -- see AIR_BAR. Reported beside the 1 bar verdict, not instead of
    # it: 1 bar is the conservative bound and air the realistic one for a box
    # furnace, and the pair brackets the answer without a fitted number.
    in_air = entropy_term(gases, temperature_C, AIR_BAR)
    crossing_air = crossover(result["enthalpy_eV"], gases, AIR_BAR)
    result["in_air"] = {
        "pressures_bar": dict(AIR_BAR),
        "entropy_term_eV": in_air,
        "free_energy_eV": result["enthalpy_eV"] + in_air,
        "feasible": result["enthalpy_eV"] + in_air < 0,
        "onset_temperature_C": (
            crossing_air["temperature_C"] if crossing_air["kind"] == "floor"
            else float("inf") if crossing_air["kind"] == "never" else None),
        "ceiling_temperature_C": (
            crossing_air["temperature_C"]
            if crossing_air["kind"] == "ceiling" else None),
    }

    # DRIVING FORCE PER ATOM, AND THE KINETIC-RISK FLAG
    #
    # A reaction can be downhill and still not happen in any useful time. The
    # A-Lab measured where that line falls: of its targets that failed for
    # "sluggish kinetics", every driving force listed in its Supplementary
    # Table 5 is between 7 and 43 meV/atom, and the paper puts the threshold
    # at 50. Ca2Ti3O8 at 7 meV/atom did not form after a day of reheating.
    #
    # Normalised per atom of REACTANTS, which is pymatgen's convention and
    # the one the A-Lab used. The A-Lab defined its threshold on 0 K DFT
    # reaction energies; this applies it to dG at the firing temperature,
    # which includes the gas term. For gas-free reactions -- exactly where
    # its failures concentrated -- the two are the same number.
    reactant_atoms = _reactant_atoms(result.get("reaction"))
    if reactant_atoms:
        force = -1000.0 * result["free_energy_eV"] / reactant_atoms
        result["driving_force_meV_per_atom"] = force
        result["kinetic_risk"] = bool(result["feasible"]
                                      and force < KINETIC_RISK_MEV_PER_ATOM)
        air = result["in_air"]
        air["driving_force_meV_per_atom"] = (
            -1000.0 * air["free_energy_eV"] / reactant_atoms)
        air["kinetic_risk"] = bool(
            air["feasible"]
            and air["driving_force_meV_per_atom"] < KINETIC_RISK_MEV_PER_ATOM)
        band = error_band_meV_per_atom(result, reactant_atoms)
        result["error_band_meV_per_atom"] = band
        result["within_error"] = (None if band is None
                                  else abs(force) < band)
    else:
        result["driving_force_meV_per_atom"] = None
        result["kinetic_risk"] = None
        result["error_band_meV_per_atom"] = None
        result["within_error"] = None
    result["verdict_class"] = (
        "unchecked" if result["feasible"] is None
        else "uncertain" if result.get("within_error")
        else "feasible" if result["feasible"] else "rejected")
    return result


# How closely DFT reaction energies match experiment when every species is a
# compound -- the case in solid-state synthesis. Bartel et al., npj Comput.
# Mater. 5, 4 (2019): PBE, SCAN and experiment agree within about
# 35 meV/atom on 231 compound-only reactions. Taken from that paper, not fitted
# to any result here.
DFT_REACTION_ERROR_MEV_PER_ATOM = 35.0


def error_band_meV_per_atom(result: dict, reactant_atoms: float
                            ) -> Optional[float]:
    """How large |dG| must be, per atom of reactants, to mean anything.

        measured, every species    2 sigma of the calorimetric errors
                                   (None if any species states none)
        Materials Project          35 meV/atom, Bartel et al. 2019
        + a structure scored by    the potential's calibration scatter on
          the potential            the target, 2 sigma, added in quadrature

    Inside the band the verdict is "uncertain": BiFeO3 from its oxides
    (+0.008 eV over 5 atoms, under 2 meV/atom) was reported "rejected" as
    firmly as a reaction 300 meV/atom uphill.
    """
    import math

    if not reactant_atoms:
        return None
    if (result.get("enthalpy_source") or "").startswith("measured"):
        sigma = result.get("enthalpy_uncertainty_eV")
        if not sigma or result.get("uncertainty_not_stated_for"):
            return None
        return 2000.0 * sigma / reactant_atoms
    band = DFT_REACTION_ERROR_MEV_PER_ATOM
    target_error = result.get("target_energy_error_eV")
    if target_error:
        band = math.hypot(band, 2000.0 * target_error / reactant_atoms)
    return band


# Below this driving force a downhill reaction tends not to finish in a
# laboratory timescale. From the A-Lab (Szymanski et al., Nature 2023),
# Supplementary Table 5: every target it failed on for sluggish kinetics sat
# between 7 and 43 meV/atom.
KINETIC_RISK_MEV_PER_ATOM = 50.0

# A generated structure this far above the known phase of its formula is
# flagged. Four times ORB-v3's polymorph-gap MAE on our benchmark (0.0124
# eV/atom, synflow/dao_p.py), so a flag is a difference it can resolve. Both
# energies are the potential's -- see polymorph_gap.
POLYMORPH_FLAG_MEV_PER_ATOM = 50.0

# A novel target this far above the known phases it would decompose into is
# flagged as unstable: twice the anchor's measured RMS error (22 meV/atom),
# so the flag marks a height the method can actually resolve.
HULL_FLAG_MEV_PER_ATOM = 44.0

# ORB-v3's error on the energy difference between two polymorphs, MAE on our
# held-out benchmark (synflow/dao_p.py). Used as the target's error when its
# energy is anchored to a known phase.
ORB_GAP_MAE_EV = 0.0124


def _reactant_atoms(equation: Optional[str]) -> Optional[float]:
    """Atoms on the reactant side of a balanced equation, coefficients
    included. Reads pymatgen's `->` and the corpus's `==` alike."""
    if not equation:
        return None
    from pymatgen.core import Composition

    text = equation.replace("==", "->")
    if "->" not in text:
        return None
    total = 0.0
    for term in text.split("->", 1)[0].split("+"):
        parts = term.strip().split()
        if not parts:
            continue
        if len(parts) == 2:
            coefficient, formula = float(parts[0]), parts[1]
        else:
            coefficient, formula = 1.0, parts[0]
        try:
            total += coefficient * Composition(formula).num_atoms
        except Exception:                                       # noqa: BLE001
            return None
    return total or None


def verdict(result: dict) -> str:
    """Plain words, with the kinetic warning appended where it applies."""
    text = _verdict(result)
    if result.get("within_error"):
        text += (" -- WITHIN ERROR: {:.0f} meV/atom either way is inside the "
                 "+/-{:.0f} meV/atom these energies can resolve, so "
                 "thermodynamics cannot decide this one".format(
                     abs(result["driving_force_meV_per_atom"]),
                     result["error_band_meV_per_atom"]))
    stability = result.get("target_stability")
    if stability:
        above = 1000.0 * stability["above_hull_eV_per_atom"]
        into = " + ".join(f for f, _ in stability.get("decomposes_to") or [])
        if above > HULL_FLAG_MEV_PER_ATOM:
            text += (" -- TARGET UNSTABLE: {:.0f} meV/atom above the known "
                     "phases it would fall apart into ({}); expect those "
                     "instead, or a mixture".format(above, into))
        else:
            text += (" -- target {:+.0f} meV/atom against the known phases "
                     "({})".format(above, into))
    polymorph = result.get("polymorph")
    if polymorph:
        gap = 1000.0 * polymorph["gap_eV_per_atom"]
        if gap > POLYMORPH_FLAG_MEV_PER_ATOM:
            text += (" -- POLYMORPH: this structure is {:.0f} meV/atom above "
                     "the known phase of the same formula; a synthesis that "
                     "can reach the known phase will usually make it "
                     "instead".format(gap))
        else:
            text += (" -- polymorph: {:+.0f} meV/atom against the known phase "
                     "of the same formula".format(gap))
    air = result.get("in_air")
    if air and result.get("feasible") is not None \
            and air["feasible"] != result["feasible"]:
        text += (" -- in AIR (CO2 at 4.2e-4 bar, O2 at 0.21) dG is {:+.2f} eV, "
                 "so it is {}".format(air["free_energy_eV"],
                                      "downhill there" if air["feasible"]
                                      else "uphill there"))
    if result.get("kinetic_risk"):
        text += (" -- KINETIC RISK: driving force {:.0f} meV/atom is below "
                 "{:.0f}; the A-Lab found reactions this weak often do not "
                 "finish, and regrinding or a higher temperature may be "
                 "needed".format(result["driving_force_meV_per_atom"],
                                 KINETIC_RISK_MEV_PER_ATOM))
    return text


def _verdict(result: dict) -> str:
    """Plain words, and the comparison DAO-S cannot make about itself."""
    if result.get("feasible") is None:
        return "could not check: {}".format(result.get("error", "unknown"))

    onset = result.get("onset_temperature_C")
    ceiling = result.get("ceiling_temperature_C")
    chosen = result["temperature_C"]
    text = "dH {:+.2f} eV, gas term {:+.2f} eV at {:.0f} C, dG {:+.2f} eV".format(
        result["enthalpy_eV"], result["entropy_term_eV"], chosen,
        result["free_energy_eV"])
    taken_up = " + ".join(sorted(
        gas for gas, moles in (result.get("gases") or {}).items() if moles < 0))

    if not result["feasible"]:
        if onset == float("inf"):
            if taken_up:
                return text + (" -- UPHILL at every temperature: it takes up "
                               "{} rather than giving gas off, so heating "
                               "only makes it worse".format(taken_up))
            return text + " -- UPHILL and no gas is released to pay for it"
        if ceiling is not None:
            return text + (" -- uphill here: above about {:.0f} C the target "
                           "gives back the {} it took up; fire below "
                           "that".format(ceiling, taken_up))
        if onset is not None:
            return text + " -- uphill here; needs about {:.0f} C".format(onset)
        return text + " -- uphill"

    if ceiling is not None:
        return text + (" -- downhill below a ceiling of about {:.0f} C, above "
                       "which the target gives back the {} it took up; the recipe "
                       "fires at {:.0f} C".format(ceiling, taken_up, chosen))
    if onset is None:
        return text + " -- downhill even without the gas"

    # THE ONSET IS A FLOOR, NOT A PREDICTION
    #
    # These two temperatures measure different things and an earlier version
    # of this compared them as if they should agree.
    #
    #   onset        thermodynamic: below it the reaction CANNOT go
    #   DAO-S        practical: where it goes at a useful rate
    #
    # Measured on the BaTiO3 control: onset 255 C, DAO-S 1100 C, and BaTiO3 is
    # genuinely fired at 1100-1300 C. The gap between them is kinetics --
    # diffusion, nucleation, time -- which this module does not model. So the
    # check is one-sided: the chosen temperature must be ABOVE the onset, and
    # being far above it is normal rather than suspicious.
    if onset < 25:
        # LiFePO4 comes out at -150 C: three gases leave, so the driving
        # force is enormous and the floor is below room temperature. Quoting
        # a sub-zero floor invites the wrong question; the honest statement is
        # that thermodynamics never blocks it and only kinetics decides.
        return text + (" -- downhill at any practical temperature; only "
                       "kinetics decides, at the recipe's {:.0f} C".format(
                           chosen))
    if chosen < onset:
        return text + (" -- INCONSISTENT: the reaction cannot go below about "
                       "{:.0f} C and the recipe fires at {:.0f} C".format(
                           onset, chosen))
    return text + (" -- consistent: thermodynamic floor about {:.0f} C, the recipe "
                   "fires at {:.0f} C, and the gap is kinetics".format(
                       onset, chosen))
