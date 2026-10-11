"""Measured formation enthalpies: calorimetry for solids, JANAF for gases.

WHAT THEY ARE FOR
-----------------
The precursors a recipe uses -- carbonates, oxides, nitrates of common
metals -- are among the most studied substances there are, and their
formation enthalpies have been MEASURED, most with a stated uncertainty.
When every species in a reaction has been measured, the reaction enthalpy
can be taken from measurement alone, with an error bar, and set against
Materials Project's. reaction.reaction_enthalpy does that.

When any species has not been measured -- which includes every generated
material -- the whole reaction comes from Materials Project instead. Never a
mixture of the two:

MIXING THEM FAILS, MEASURED 2026-10-08
--------------------------------------
This module first filled the gaps with MP's per-compound formation energy,
on the argument that MP's corrections are fitted to measured data, so both
sit on one scale. Compound by compound they do not:

    TiO2    measured -3.261 eV/atom    MP -3.508    (0.25 eV/atom)
    Ta2O5   measured -3.030            MP -3.347    (0.32)
    Co3O4   measured -1.347            MP -1.290    (0.06, other sign)

Within MP those offsets cancel across a reaction; across sources they do
not, and a reaction is a small difference of large numbers:

    CuO + Ta2O5 -> CuTa2O6    mixed -2.147 eV    all-MP +0.338 eV

MP's per-compound values with JANAF's gases fail the same way: BaCO3 + TiO2
-> BaTiO3 came out +2.40 eV, against +1.17 all-MP and +1.25 all-measured.
So that fallback was removed, and with it a second wrong claim: that
mixing was "safe here".

A FIRST CLAIM, ALSO WITHDRAWN
This docstring originally said MP's enthalpy for 2 MnO2 -> Mn2O3 + 0.5 O2
was 0.93 eV too high. MP gives +0.866 eV and calorimetry +0.867; the 0.93 eV
was the entropy of the O2 released, which the simulator had not counted --
see reaction.gas_coefficients.

SOURCES
-------
gases   NIST-JANAF gas-phase values, below. Gas phase matters: the CO2
        leaving a carbonate leaves as gas.
solids  Kingsbury et al., Phys. Rev. Materials 6, 013801 (2022): 2,135
        calorimetric formation enthalpies from Kim et al., Kubaschewski and
        NIST-JANAF, 1,834 of them with a stated uncertainty.
"""

import functools
import gzip
import json
import os
from typing import Dict, Optional, Tuple

DATASET = "data/thermo/expt_formation_enthalpy_kingsbury.json.gz"
_FIGSHARE = "https://ndownloader.figshare.com/files/28232538"
EXPECTED_MD5 = "961033dbfa21626850ee6ee8d8e14d7c"

KJ_PER_MOL_TO_EV = 1.0 / 96.4853

# Standard formation enthalpies of gases at 298.15 K, kJ/mol, NIST-JANAF.
# Kingsbury's dataset excludes gases, so these few are stated here, the same
# way the gas entropies in reaction.py are.
GAS_FORMATION_KJ_PER_MOL = {
    "CO2": -393.51,
    "CO": -110.53,
    "H2O": -241.83,
    "NH3": -45.90,
    "NO2": 33.10,
    "SO2": -296.81,
    "HCl": -92.31,
    "O2": 0.0, "N2": 0.0, "H2": 0.0, "Cl2": 0.0, "F2": 0.0,
}


@functools.lru_cache(maxsize=1)
def _gas_table() -> Dict[str, float]:
    """The gas table keyed by pymatgen's reduced formula.

    pymatgen orders elements by electronegativity, so ammonia is "H3N", not
    "NH3". Keyed by the common spelling, the lookup would have missed it and
    sent ammonia -- released by every NH4H2PO4 recipe -- to the solid-phase
    database instead.
    """
    from pymatgen.core import Composition
    return {Composition(k).reduced_formula: v
            for k, v in GAS_FORMATION_KJ_PER_MOL.items()}


def download_dataset(path: str = DATASET) -> str:
    import hashlib
    import urllib.request

    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urllib.request.urlretrieve(_FIGSHARE, path)
    with open(path, "rb") as handle:
        digest = hashlib.md5(handle.read()).hexdigest()
    if digest != EXPECTED_MD5:
        os.remove(path)
        raise RuntimeError("md5 {} != {}; file removed".format(
            digest, EXPECTED_MD5))
    return path


@functools.lru_cache(maxsize=1)
def measured_table(path: str = DATASET) -> Dict[str, Tuple[float, Optional[float]]]:
    """reduced formula -> (formation enthalpy eV/atom, uncertainty eV/atom).

    The uncertainty is None where the source states none -- 301 of the
    2,135 entries, Fe2O3, FeO, CuO and MgO among them. The file stores those
    as NaN; reading NaN as "no error" would understate the error bar, and
    `NaN or 0.0` does not catch it, because NaN is truthy.
    """
    import math

    from pymatgen.core import Composition

    if not os.path.exists(path):
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        blob = json.load(handle)
    columns = blob["columns"]
    out = {}
    for row in blob["data"]:
        record = dict(zip(columns, row))
        try:
            key = Composition(record["formula"]).reduced_formula
        except Exception:                                       # noqa: BLE001
            continue
        try:
            error = float(record.get("uncertainty"))
        except (TypeError, ValueError):
            error = None
        if error is not None and not math.isfinite(error):
            error = None
        out[key] = (float(record["expt_form_e"]), error)
    return out


def formation_enthalpy(formula: str) -> Optional[dict]:
    """eV per atom from JANAF (gases) or calorimetry (solids), else None.

    Returns {"per_atom", "uncertainty", "source"}. There is deliberately no
    computed fallback -- see the module docstring for what mixing did.
    """
    from pymatgen.core import Composition

    try:
        composition = Composition(formula)
    except Exception:                                           # noqa: BLE001
        return None
    reduced = composition.reduced_formula
    atoms = composition.reduced_composition.num_atoms

    gases = _gas_table()
    if reduced in gases:
        per_molecule = gases[reduced] * KJ_PER_MOL_TO_EV
        return {"per_atom": per_molecule / atoms, "uncertainty": 0.0,
                "source": "NIST-JANAF gas"}

    table = measured_table()
    if reduced in table:
        value, error = table[reduced]
        return {"per_atom": value, "uncertainty": error,
                "source": "measured (Kingsbury 2022)"}
    return None
