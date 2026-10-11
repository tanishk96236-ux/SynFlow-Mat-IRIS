"""What kind of compound a target is, and the oxidation state of one metal in it.

Both exist for one consumer: deciding which precursor supplies a metal. The
metal-lookup rule picks one precursor per metal without looking at the target,
and its misses are predictable from exactly these two facts.

    Fe(3+) targets   n=1,541   Fe2O3 used 85%   -> the rule is right 85%
    Fe(2+) targets   n=418     Fe2O3 used 28%   -> the rule is right 28%

and for a target with no oxygen at all, iron is usually supplied as the bare
metal, which the rule never proposes.

WHY THE OXIDATION STATE IS SOLVED HERE INSTEAD OF ASKED OF PYMATGEN
------------------------------------------------------------------
Composition.oxi_state_guesses enumerates charge-balanced assignments over an
INTEGER number of sites. Measured on 400 random targets:

    oxide, integer formula      80% get a guess
    oxide, decimal formula       0% -- all 118 failed

and decimal (doped) formulas are about a third of the corpus.
Ba0.85Ca0.15Zr0.1Ti0.9O3 needs a hundred sites to become integer, which is
past what the enumeration will do.

But the question here is narrower: the state of ONE metal. That is the
calculation a chemist does by hand -- put every other element at its usual
state, require the total charge to be zero, solve for the one you want:

    Ba0.85 Ca0.15 Zr0.1 Ti0.9 O3, solving for Ti
      Ba +2 x 0.85 = +1.70
      Ca +2 x 0.15 = +0.30
      Zr +4 x 0.10 = +0.40
      O  -2 x 3    = -6.00
      sum          = -3.60    so 0.9 x Ti = +3.60, Ti = +4

"Usual state" is not chosen by hand: it is the most frequent state of that
element in the ICSD, from the occurrence table pymatgen itself uses to rank
its guesses. The solve is used only where the enumeration fails, and its
agreement with the enumeration was measured where both work.
"""

import functools
import os
from typing import Dict, Optional

# Classes in priority order. A target with oxygen is an oxide whatever else it
# contains -- phosphates, borates and silicates are oxides here, because the
# precursor question ("supply the metal as an oxide or a carbonate?") is the
# same for them.
ANION_CLASSES = ("oxide", "chalcogenide", "nitride", "halide", "carbide",
                 "boride", "pnictide", "hydride", "intermetallic")

_ANION_STATE = {"O": -2, "S": -2, "Se": -2, "Te": -2, "F": -1, "Cl": -1,
                "Br": -1, "I": -1, "N": -3}


def anion_class(formula: str) -> Optional[str]:
    """oxide, chalcogenide, nitride, halide, carbide, boride, pnictide,
    hydride or intermetallic -- or None if the formula cannot be read."""
    from .recipes import composition

    elements = set(composition(formula))
    if not elements:
        return None
    if "O" in elements:
        return "oxide"
    if elements & {"S", "Se", "Te"}:
        return "chalcogenide"
    if "N" in elements:
        return "nitride"
    if elements & {"F", "Cl", "Br", "I"}:
        return "halide"
    if "C" in elements:
        return "carbide"
    if "B" in elements:
        return "boride"
    if elements & {"P", "As", "Sb"}:
        return "pnictide"
    if "H" in elements:
        return "hydride"
    return "intermetallic"


@functools.lru_cache(maxsize=1)
def _icsd_most_common() -> Dict[str, int]:
    """Each element's most frequent POSITIVE oxidation state in the ICSD.

    Positive, because in an oxide everything except oxygen is the cation
    side: phosphorus is +5 in a phosphate, sulfur +6 in a sulfate, carbon +4
    in a carbonate. Their most frequent state overall can be negative --
    phosphides, sulfides -- which would be wrong in exactly the compounds
    this is used on.
    """
    from monty.serialization import loadfn
    from pymatgen.core import composition as composition_module

    path = os.path.join(os.path.dirname(composition_module.__file__),
                        "icsd_bv.yaml")
    occurrence = loadfn(path)["occurrence"]

    best: Dict[str, tuple] = {}
    for species, count in occurrence.items():
        element = species.rstrip("+-0123456789.")
        sign = -1 if species.endswith("-") else 1
        digits = species[len(element):].rstrip("+-") or "1"
        try:
            state = sign * int(float(digits))
        except ValueError:
            continue
        if state <= 0:
            continue
        if element not in best or count > best[element][1]:
            best[element] = (state, count)
    return {element: state for element, (state, _) in best.items()}


def _solved_state(formula: str, element: str) -> Optional[int]:
    """The element's state by charge balance, everything else held at its
    usual state. None when there is no anion to balance against."""
    from .recipes import composition

    counts = composition(formula)
    if element not in counts or not counts[element]:
        return None

    cls = anion_class(formula)
    if cls == "oxide":
        anions = {"O"}
    elif cls in ("chalcogenide", "nitride", "halide"):
        anions = {e for e in counts if e in _ANION_STATE and e != "O"}
    else:
        return None                     # intermetallics have no balance

    usual = _icsd_most_common()
    charge = 0.0
    for other, amount in counts.items():
        if other == element:
            continue
        if other in anions:
            charge += _ANION_STATE[other] * amount
        elif other in usual:
            charge += usual[other] * amount
        else:
            return None                 # an element with no known state
    state = -charge / counts[element]
    rounded = round(state)
    # A fractional answer means the "usual state" assumption failed for one
    # of the others -- mixed valence somewhere -- so the integer would be a
    # guess dressed as a result. Only near-integers are kept.
    if abs(state - rounded) > 0.25 or rounded <= 0:
        return None

    # AND ONLY STATES THE ELEMENT CAN ACTUALLY TAKE
    #
    # Dividing by the element's amount amplifies any error in the others,
    # and a minor dopant amplifies it most. Without this check the solve
    # returned Nb +29 in Li0.75La3Zr1.75Nb0.25O12 and Zn +10 in
    # Mn2Zn0.25Ni0.75O4; on integer oxides it returned Sr +3, Zn +6 and Ag +4.
    # None of those exist.
    #
    # The first of those is worth a second look: it is a lithium garnet, and
    # the real formula is Li6.75La3Zr1.75Nb0.25O12 -- the extraction dropped
    # the "6". The charge balance failing is what exposed it. Refusing here
    # means the precursor rule backs off to a coarser key rather than acting
    # on a composition that cannot exist.
    from pymatgen.core import Element
    try:
        allowed = set(Element(element).oxidation_states)
    except Exception:                                           # noqa: BLE001
        return None
    if rounded not in allowed:
        return None
    return int(rounded)


@functools.lru_cache(maxsize=50000)
def _enumerated(formula: str) -> Optional[dict]:
    """pymatgen's best charge-balanced assignment for the whole formula.

    Cached per FORMULA, not per element: the enumeration answers for every
    element at once, and asking it once per element repeated the slow part
    three or four times for every target.
    """
    from pymatgen.core import Composition
    try:
        guesses = Composition(formula).oxi_state_guesses(max_sites=-20)
    except Exception:                                           # noqa: BLE001
        return None
    return dict(guesses[0]) if guesses else None


@functools.lru_cache(maxsize=200000)
def oxidation_state(formula: str, element: str) -> Optional[int]:
    """The oxidation state of `element` in `formula`, or None.

    pymatgen's ICSD-ranked enumeration where it succeeds -- integer formulas
    -- and the charge-balance solve where it does not. Measured where both
    answer: the solve agrees with the enumeration 95.8% of the time on
    integer oxides, and refuses rather than guesses where it cannot.
    """
    cls = anion_class(formula)
    if cls is None:
        return None
    if cls == "intermetallic":
        return 0
    enumerated = _enumerated(formula)
    if enumerated and element in enumerated:
        return int(round(enumerated[element]))
    return _solved_state(formula, element)
