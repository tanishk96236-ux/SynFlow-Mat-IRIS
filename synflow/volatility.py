"""Will part of the recipe evaporate before it reacts? A volatility warning.

WHY
---
Two of the three failures among MatterGen's lab-tested candidates (Nature
2025, SI D.10.5) were volatility, not thermodynamics: Cr2MoO6 and LaMoO4 "lost
Mo" when heated to 900 C. The simulator called the lab's LaMoO4 recipe
downhill by 4.6 eV -- correctly, and uselessly, because the molybdenum had
left as MoO3 vapour. Nothing in this repository modelled that.

THE RULE, AND ITS SOURCES
-------------------------
A handful of elements form oxides that melt low and evaporate readily, and
solid-state practice works around them -- a lid, a sealed tube, an excess of
the volatile precursor, or a lower temperature. For each, the warning fires
when the firing temperature reaches the melting point of that volatile
oxide or carbonate -- the point past which its vapour pressure climbs
steeply. The melting points are the Precursor Genome's own table (with CAS
numbers and supplier product numbers), read through kinetics.melting_point,
not values written here:

    Mo -> MoO3 (795 C)     Sb -> Sb2O3 (655 C)     Bi -> Bi2O3 (817 C)
    Pb -> PbO (886 C)      V  -> V2O5 (690 C)
    Li -> Li2CO3 (720 C)   Na -> Na2CO3 (856 C)    K  -> K2CO3 (891 C)

It keys on the ELEMENT in any precursor, because a metal or a lower oxide
oxidises in air to the volatile one: the lab's LaMoO4 used Mo metal, which
becomes MoO3. Which elements are listed is solid-state convention, labelled
as such; it is a warning to act on, not a prediction of how much is lost.
"""

from typing import Dict, List, Sequence

# element -> the volatile compound whose melting point sets the threshold
VOLATILE = {
    "Mo": "MoO3",
    "Sb": "Sb2O3",
    "Bi": "Bi2O3",
    "Pb": "PbO",
    "V": "V2O5",
    "Li": "Li2CO3",
    "Na": "Na2CO3",
    "K": "K2CO3",
}


def thresholds() -> Dict[str, float]:
    """element -> temperature (C) above which loss is warned of."""
    from . import kinetics

    out = {}
    for element, compound in VOLATILE.items():
        value = kinetics.melting_point(compound)
        if value is not None:
            out[element] = value
    return out


def check(precursors: Sequence[str], temperature_C: float) -> List[dict]:
    """One entry per volatile element present above its threshold."""
    from pymatgen.core import Composition

    limits = thresholds()
    present = set()
    for formula in precursors:
        try:
            present |= {str(e) for e in Composition(formula)}
        except Exception:                                       # noqa: BLE001
            continue
    warnings = []
    for element in sorted(present & set(limits)):
        if temperature_C >= limits[element]:
            warnings.append({
                "element": element,
                "as": VOLATILE[element],
                "threshold_C": limits[element],
                "temperature_C": temperature_C,
                "advice": ("{} can be lost as {} vapour above {:.0f} C: use a "
                           "lid or a sealed tube, a 5-10% excess of the {} "
                           "precursor, or fire lower".format(
                               element, VOLATILE[element], limits[element],
                               element)),
            })
    return warnings
