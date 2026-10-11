"""Is a generated compound actually new, and actually stable?

This is the piece the blueprint does not have, and without it the phrase
"novel material" has no meaning here.

WHY THE EXISTING MEASURE CANNOT ANSWER IT
-----------------------------------------
Everything measured so far compares a generated structure against the known
Materials Project structure of the SAME composition. That works only for
compositions MP already has -- which is precisely the set that contains
nothing new. For a composition nobody has made there is no reference
structure to compare to, and the question changes from "did we find the
right arrangement" to "is this compound stable at all".

WHAT STABILITY MEANS FOR A NEW COMPOSITION
------------------------------------------
A compound is stable if nothing it could decompose into is cheaper. Take
Ba-Ti-O: a candidate BaTi2O5 competes against BaO + TiO2, against BaTiO3 +
TiO2, against every other mixture of known phases with the same overall
composition. The convex hull is the lowest-energy combination available at
each composition, and `energy above hull` is how far our candidate sits
above it.

    e_above_hull = 0          on the hull; stable
    0 to ~0.1 eV/atom         metastable, and routinely synthesised --
                              Sun et al. 2016 found most known oxides sit
                              in this band
    much above that           it would decompose

Below zero would mean the candidate is more stable than everything known
for that composition, which is the actual claim a discovery makes.

ONE HONESTY CONSTRAINT, AND IT DECIDES THE DESIGN
-------------------------------------------------
Materials Project's hull is built from DFT energies. ORB predicts DFT
energies but is not DFT, and the two sit on different absolute scales --
DAO-P's own docstring says absolute energies are not comparable across
potentials. Dropping an ORB number straight onto a DFT hull compares two
different quantities and produces a confident, meaningless answer.

So the correction is measured rather than assumed: for compounds where MP
has both a structure and an energy, the offset between what ORB says and
what DFT says is computed per chemical system, and applied to the
candidate. It is crude, and `calibration_error` reports how crude, so a
result can be read with the right amount of doubt.
"""

import collections
import functools
from typing import Dict, List, Optional

import numpy as np
from pymatgen.core import Composition


@functools.lru_cache(maxsize=256)
def _entries_for(chemsys: str, api_key: Optional[str] = None):
    """Every known phase in a chemical system, cached.

    Cached because the same system comes up repeatedly -- substituting
    across a perovskite family asks for Ba-Ti-O once per candidate
    otherwise, and the query is the slow part.
    """
    import os

    from mp_api.client import MPRester

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise SystemExit("MP_API_KEY is not set")
    with MPRester(key) as mpr:
        # inc_structure, or most entries come back without one and
        # cannot be scored at all -- a 40-point fit silently became 8.
        return tuple(mpr.get_entries_in_chemsys(
            chemsys.split("-"), inc_structure=True))


def phase_diagram(elements, api_key: Optional[str] = None):
    """The convex hull for a set of elements, from Materials Project."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram

    chemsys = "-".join(sorted(set(elements)))
    entries = _entries_for(chemsys, api_key)
    if not entries:
        raise ValueError("no known phases for {}".format(chemsys))
    return PhaseDiagram(list(entries)), entries


def _references(diagram, entries, critic):
    """The same pure elements priced two ways: by MP, and by ORB.

    This pairing is the whole mechanism. ORB is good at formation energies
    -- differences from the pure elements -- and useless at absolute ones,
    because its zero is not DFT's zero. MP's hull lives on MP's absolute
    scale. So the candidate's formation energy is computed with ORB, and
    then MP's own elemental energies are added back to put it on MP's
    scale. Each method is used only where it is valid.
    """
    mp_energy, orb_energy = {}, {}
    for element, entry in diagram.el_refs.items():
        symbol = str(element)
        mp_energy[symbol] = entry.energy_per_atom
        structure = getattr(entry, "structure", None)
        if structure is None or not structure.is_ordered:
            continue
        try:
            orb_energy[symbol] = critic.score(structure)
        except Exception:                                   # noqa: BLE001
            continue
    return mp_energy, orb_energy


def formation_energy(structure, critic, orb_refs: Dict[str, float]) -> float:
    """Energy per atom relative to the pure elements, entirely under ORB."""
    composition = structure.composition
    missing = [str(el) for el in composition if str(el) not in orb_refs]
    if missing:
        raise ValueError(
            "no elemental reference for {}".format(", ".join(missing)))
    fractions = composition.fractional_composition
    reference = sum(fractions.get(el, 0.0) * e for el, e in orb_refs.items())
    return critic.score(structure) - reference


def calibrate(diagram, entries, critic, orb_refs, sample: int = 20,
              seed: int = 0, max_atoms: int = 40) -> dict:
    """How closely ORB's formation energies track Materials Project's.

    Not a correction -- a statement of how much to trust what follows. MP's
    formation energies come from the phase diagram itself, so both sides
    are measured from the same pure elements and the comparison is fair.

    Two earlier attempts tried to correct between TOTAL energies, first
    with one offset per chemical system and then one per element. Both were
    the wrong problem, and the symptom was unmissable: MgO, a textbook
    stable compound, landed 4.16 eV above its own hull. Total energies from
    two different methods share no zero, and no fitted shift creates one.
    """
    import random

    usable = [e for e in entries
              if getattr(e, "structure", None) is not None
              and e.structure.is_ordered and len(e.composition) > 1]
    if not usable:
        raise ValueError("no multi-element phases with structures")

    # SAMPLE ACROSS THE PHASES WE CAN AFFORD, NOT THE SMALLEST ONES
    #
    # This used to sort by cell size and take the 2*sample smallest, then
    # sample within that. Which means changing `sample` changed the
    # POPULATION and not just how much of it was drawn: at sample=10 the fit
    # saw the 20 smallest phases in the chemical system, at sample=30 the 60
    # smallest. The 20 smallest are mostly simple binaries of a few atoms,
    # where ORB's error is both smaller and more uniform than it is on the
    # complex phases a candidate actually competes against.
    #
    # So the uncertainty at ten phases was not merely noisy, it was measured
    # on a biased subsample and understated the real scatter. Measured on
    # perovskites, ten phases against thirty:
    #
    #     YTaO3    +0.562 +/- 0.006   ->   +0.594 +/- 0.115
    #     LaHfO3   +0.258 +/- 0.010   ->   +0.269 +/- 0.067
    #     AgMoO3   +0.214 +/- 0.036   ->   +0.255 +/- 0.130
    #
    # An uncertainty that grows nineteenfold when given more data is not
    # converging on anything; it was wrong.
    #
    # The size cap stays, because scoring a 200-atom cell with ORB is what
    # the truncation was protecting against. But it is now a cap on cost
    # rather than a ranking, so everything affordable is eligible and the
    # sample is representative of it.
    affordable = [e for e in usable if len(e.structure) <= max_atoms]
    if len(affordable) < min(sample, 8):
        # Rather than silently fitting on three phases, widen the cap and
        # say so in the output via n_calibration.
        affordable = sorted(usable, key=lambda e: len(e.structure))[:sample * 2]
    chosen = random.Random(seed).sample(affordable,
                                        min(sample, len(affordable)))

    errors = []
    for entry in chosen:
        try:
            theirs = diagram.get_form_energy_per_atom(entry)
            ours = formation_energy(entry.structure, critic, orb_refs)
        except Exception:                                   # noqa: BLE001
            continue
        errors.append((ours - theirs, entry.composition))

    if not errors:
        raise ValueError("could not compare any phase against MP")

    # Fit the error PER ELEMENT, not as one number, because it is not one
    # number. Measured on Na-Cl the error grows with chlorine content --
    # +0.13 at 25% Cl, +0.29 at 50%, +0.53 at 87% -- and on Mg-O it grows
    # with oxygen the same way. That is the signature of a bad elemental
    # reference: O2 and Cl2 are molecular solids, Materials Project applies
    # a specific correction to them because GGA describes the O2 molecule
    # badly, and ORB carries no such correction.
    #
    # An error that is linear in composition needs a correction that is
    # linear in composition. A median would leave most of it behind.
    elements = sorted({str(el) for _, comp in errors for el in comp})
    matrix = np.array([[comp.fractional_composition.get(el, 0.0)
                        for el in elements] for _, comp in errors])
    y = np.array([e for e, _ in errors])
    coefficients, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    residuals = y - matrix @ coefficients

    # TWO THINGS ABOUT calibration_error, AND THE SECOND IS WORSE
    #
    # It is the standard deviation of the residuals -- how far individual
    # known phases sit from the fitted correction -- so it is the right
    # uncertainty to quote for a new candidate, which is one more draw from
    # that same scatter. It is NOT a standard error, and more phases do not
    # average it down. KRhO2 on 40 phases instead of 10: +0.0311 +/- 0.131
    # became +0.0387 +/- 0.112. Four times the data, 15% less spread.
    #
    # The second thing is the one that matters. THE CENTRAL ESTIMATE IS ALSO
    # UNSTABLE at small samples, and by more than the quoted spread. KCrPO4,
    # same change from 10 phases to 40:
    #
    #     +0.1204 +/- 0.143   ->   +0.3021 +/- 0.165
    #
    # The distance moved 0.18 eV/atom, further than the uncertainty it came
    # with, and the uncertainty grew. A ten-phase fit is not a coarse
    # version of a forty-phase fit; it is a different and wrong plane
    # through the data. Most of the nine free coefficients in a four-element
    # system are being fitted on almost nothing.
    #
    # So the screening sample is 30 rather than 10, and a result whose
    # distance is near its spread stays unresolved no matter the sample --
    # the scatter is what a composition-linear ORB-to-MP correction leaves
    # behind, and past it lies a better correction or DFT.
    return {"calibration_per_element": dict(zip(
                elements, (float(c) for c in coefficients))),
            "calibration_bias": float(np.median(y)),
            "calibration_error": float(np.std(residuals)),
            "calibration_max_error": float(np.abs(residuals).max()),
            "n_calibration": len(y),
            # How many phases were eligible, so a thin chemical system is
            # visible as a thin one. When these are equal the fit used
            # everything available and asking for more cannot help.
            "calibration_pool": len(affordable),
            "calibration_atoms_max": max_atoms}


# Calibration is ~10 ORB scorings per chemical system, and screening a
# perovskite family asks about Ba-Ti-O once per candidate. Cached by system
# and by the critic's identity, so swapping the potential invalidates it
# rather than silently reusing another model's numbers.
_CALIBRATION_CACHE = {}


def _cached_calibration(diagram, entries, critic, orb_refs, chemsys, sample):
    key = (chemsys, id(critic), sample)
    if key not in _CALIBRATION_CACHE:
        _CALIBRATION_CACHE[key] = calibrate(diagram, entries, critic,
                                            orb_refs, sample)
    return _CALIBRATION_CACHE[key]


def energy_above_hull(structure, critic, api_key: Optional[str] = None,
                      calibration_sample: int = 20) -> dict:
    """How far a candidate sits above the lowest-energy mixture available."""
    from pymatgen.analysis.phase_diagram import PDEntry

    elements = sorted({str(site.specie) for site in structure})
    diagram, entries = phase_diagram(elements, api_key)
    reference_key = ("refs", "-".join(elements), id(critic))
    if reference_key not in _CALIBRATION_CACHE:
        _CALIBRATION_CACHE[reference_key] = _references(
            diagram, entries, critic)
    mp_refs, orb_refs = _CALIBRATION_CACHE[reference_key]

    missing = [el for el in elements if el not in orb_refs]
    if missing:
        raise ValueError(
            "no scorable elemental structure for {}".format(", ".join(missing)))

    correction = _cached_calibration(diagram, entries, critic, orb_refs,
                                     "-".join(elements), calibration_sample)

    # Subtract the measured bias. ORB's formation energies sit systematically
    # off MP's within a chemical system -- the same functional, the same
    # corrections, applied to neither -- and that offset is what `calibrate`
    # measures on phases MP already has. Measuring it and then not applying
    # it, which is what this did at first, leaves a stable compound sitting
    # 0.34 eV/atom above its own hull for no reason.
    #
    # The residual spread stays in the output as calibration_error, and
    # `verdict` refuses to judge anything smaller than it.
    fitted = correction["calibration_per_element"]
    missing_fit = [str(el) for el in structure.composition
                   if str(el) not in fitted]
    if missing_fit:
        raise ValueError("no calibration fitted for {}".format(
            ", ".join(missing_fit)))
    bias = sum(structure.composition.fractional_composition.get(el, 0.0) * c
               for el, c in fitted.items())
    ours = formation_energy(structure, critic, orb_refs) - bias
    fractions = structure.composition.fractional_composition
    on_mp_scale = ours + sum(fractions.get(el, 0.0) * mp_refs[el]
                             for el in elements)

    entry = PDEntry(structure.composition, on_mp_scale * len(structure))
    # allow_negative, because a candidate BELOW the hull is the only kind
    # that is a discovery -- and the default refuses to report it, raising
    # "No valid decomposition found" instead. Correct in its own terms: an
    # entry under the hull has nothing to decompose into, it redefines the
    # hull. For screening, that exception IS the answer.
    above = float(diagram.get_e_above_hull(entry, allow_negative=True))
    return {
        "formula": structure.composition.reduced_formula,
        "chemsys": "-".join(elements),
        "e_above_hull": above,
        "formation_energy_per_atom": ours,
        "known_phases": len(entries),
        "stable_phases": len(diagram.stable_entries),
        **correction,
        "is_known": any(e.composition.reduced_formula ==
                        structure.composition.reduced_formula
                        for e in entries),
    }


def verdict(result: dict) -> str:
    """Plain words for a hull distance, with its uncertainty respected.

    A candidate 0.02 eV/atom above the hull with a calibration spread of
    0.15 has not been shown to be anything, and saying "metastable" there
    would be inventing precision the measurement does not have.
    """
    above, error = result["e_above_hull"], result["calibration_error"]

    if error > 0.1:
        return ("calibration too loose to judge (+/- {:.2f} eV/atom)"
                .format(error))
    if above < -error:
        return "BELOW the known hull -- more stable than anything recorded"
    if above <= error:
        return "on the hull within calibration error -- plausibly stable"
    if above <= 0.1:
        return "metastable, in the band where most known oxides sit"
    if above <= 0.3:
        return "well above the hull; unlikely without special conditions"
    return "far above the hull; would decompose"


def screen(structures, critic, api_key: Optional[str] = None) -> List[dict]:
    """Hull distance for many candidates, most stable first.

    Failures are kept in the output rather than dropped. A composition
    whose chemical system MP does not cover is a real outcome -- it means
    the candidate cannot be judged, which is different from judging it
    badly.
    """
    results = []
    for structure in structures:
        try:
            row = energy_above_hull(structure, critic, api_key)
            row["verdict"] = verdict(row)
        except Exception as error:                          # noqa: BLE001
            row = {"formula": structure.composition.reduced_formula,
                   "e_above_hull": None,
                   "verdict": "could not judge: {}".format(
                       type(error).__name__)}
        results.append(row)
    return sorted(results, key=lambda r: (r["e_above_hull"] is None,
                                          r["e_above_hull"] or 0.0))


def anchor(structure, critic, diagram, relax_steps: int = 80
           ) -> Optional[dict]:
    """The structure's energy, anchored to the known phases it would become.

    WHY NOT THE CALIBRATION
    _novel_entry puts a new structure on MP's scale through a correction
    that is linear in composition, fitted per chemical system, and its
    scatter is 0.11-0.16 eV/atom. Carried into a reaction, that made every
    novel target in the MatterGen showcase "uncertain" -- bands of +/-111
    to +/-643 meV/atom, against +/-38 for a known target.

    THE ANCHOR
    MP's hull already says what the structure's composition would fall
    apart into, and MP has those phases' energies and structures. So the
    potential is asked only for a DIFFERENCE between structures of nearly
    the same chemistry -- the candidate against that mixture, both relaxed
    and scored the same way -- and MP supplies the rest:

        E(candidate) = sum x_i E_MP(phase_i) + [E_pot(candidate)
                                                - sum x_i E_pot(phase_i)]

    The same trick as the polymorph gap (0.0124 eV/atom MAE), stretched
    across compositions; how far it stretches is measured by
    scripts/validate_anchoring.py before it is used.

    Returns None when the decomposition includes a pure element (O2, a
    metal) or a phase with no ordered structure -- the potential is not
    trusted on those, and the calibration path is used instead.
    """
    from synflow import dao_g

    try:
        decomposition = diagram.get_decomposition(structure.composition)
    except Exception:                                           # noqa: BLE001
        return None
    pot_mix = mp_mix = 0.0
    anchors = []
    for entry, fraction in decomposition.items():
        phase = getattr(entry, "structure", None)
        if entry.composition.is_element or phase is None \
                or not phase.is_ordered:
            return None
        relaxed = dao_g.relax(phase, critic, steps=relax_steps)
        pot_mix += fraction * float(critic.score(relaxed))
        mp_mix += fraction * float(entry.energy_per_atom)
        anchors.append((entry.composition.reduced_formula, float(fraction)))
    gap = float(critic.score(structure)) - pot_mix
    return {"energy_per_atom": mp_mix + gap, "gap_eV_per_atom": gap,
            "anchors": anchors}
