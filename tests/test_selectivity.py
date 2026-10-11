"""Selectivity on a made-up Ba-Ti-O system, checkable by hand.

Energies are invented, per formula unit, so the arithmetic is visible:

    BaO      -6.0      TiO2    -9.0      O2  -1.0 (per O2)
    BaTiO3  -15.5      ->  BaO + TiO2 -> BaTiO3   dE = -0.5 eV over 5 atoms
    Ba2TiO4 -22.0      ->  2 BaO + TiO2 -> Ba2TiO4 dE = -1.0 eV over 7 atoms

Per atom of reactants: target -0.100, competitor -0.143. The competitor is
more favourable, so C1 = -0.100 - (-0.143) = +0.043 eV/atom: not selective.
No network: the table is passed in.
"""

import pytest

from synflow import selectivity


def _table(ba2tio4):
    from pymatgen.entries.computed_entries import ComputedEntry
    energies = {"BaO": -6.0, "TiO2": -9.0, "O2": -1.0, "BaTiO3": -15.5,
                "Ba2TiO4": ba2tio4}
    return {formula: ComputedEntry(formula, energy)
            for formula, energy in energies.items()}


def test_target_reaction_per_atom_by_hand():
    result = selectivity.reaction_free_energy(["BaO", "TiO2"], "BaTiO3", 800,
                                              _table(-22.0))
    assert result["dG_eV_per_atom"] == pytest.approx(-0.5 / 5, abs=1e-6)


def test_a_more_favourable_competitor_makes_c1_positive():
    result = selectivity.primary_competition(["BaO", "TiO2"], "BaTiO3", 800,
                                             table=_table(-22.0))
    assert result["strongest_competitor"]["product"] == "Ba2TiO4"
    assert result["C1_eV_per_atom"] == pytest.approx(-0.1 + 1.0 / 7,
                                                     abs=1e-6)
    assert "NOT selective" in selectivity.explain(result)


def test_a_product_keeping_a_precursors_volatile_element_is_not_a_rival():
    """BaCO3 + TiO2: a carbon-bearing "competitor" would be carbon kept in a
    solid at the firing temperature. Only the target's elements and oxygen
    may appear in a rival product."""
    from pymatgen.entries.computed_entries import ComputedEntry
    table = _table(-21.2)
    table.update({"BaCO3": ComputedEntry("BaCO3", -14.0),
                  "CO2": ComputedEntry("CO2", -2.0),
                  # an absurdly stable carbon-bearing phase
                  "Ba2TiC2O7": ComputedEntry("Ba2TiC2O7", -60.0)})
    result = selectivity.primary_competition(["BaCO3", "TiO2"], "BaTiO3",
                                             800, table=table)
    assert result["strongest_competitor"]["product"] != "Ba2TiC2O7"


def test_a_weak_competitor_leaves_the_target_selective():
    # Ba2TiO4 at -21.2: 2 BaO + TiO2 -> Ba2TiO4 releases only 0.2 eV / 7 atoms
    result = selectivity.primary_competition(["BaO", "TiO2"], "BaTiO3", 800,
                                             table=_table(-21.2))
    assert result["C1_eV_per_atom"] == pytest.approx(-0.1 + 0.2 / 7,
                                                     abs=1e-6)
    assert result["C1_eV_per_atom"] < 0
    assert selectivity.explain(result).startswith("selective")


def test_what_is_left_after_the_rival_forms_by_hand():
    """Once Ba2TiO4 has formed, the step to the target is

        Ba2TiO4 + TiO2 -> 2 BaTiO3

    With Ba2TiO4 at -22.0: 2(-15.5) - (-22.0 - 9.0) = 0 -- nothing left, the
    stall the A-Lab described. With Ba2TiO4 at -21.2: -0.8 eV over 10 atoms
    of reactants = -0.080 eV/atom left."""
    stalled = selectivity.primary_competition(["BaO", "TiO2"], "BaTiO3", 800,
                                              table=_table(-22.0))
    assert stalled["remaining"]["dG_eV_per_atom"] == pytest.approx(0.0,
                                                                   abs=1e-6)
    alive = selectivity.primary_competition(["BaO", "TiO2"], "BaTiO3", 800,
                                            table=_table(-21.2))
    assert alive["remaining"]["dG_eV_per_atom"] == pytest.approx(-0.080,
                                                                 abs=1e-6)
    assert alive["remaining"]["from_"][0] == "Ba2TiO4"


def test_one_precursor_decomposing_is_not_a_rival():
    """BaO2 -> BaO + 1/2 O2 is calcination, not competition."""
    from pymatgen.entries.computed_entries import ComputedEntry
    table = _table(-21.2)
    table["BaO2"] = ComputedEntry("BaO2", -5.0)      # wildly unstable
    result = selectivity.primary_competition(["BaO2", "TiO2"], "BaTiO3", 800,
                                             table=table)
    rival = result["strongest_competitor"]
    assert rival is None or len(rival["precursors"]) >= 2


def test_a_reactant_the_balance_does_not_need_is_not_consumed():
    """SnO2 is not needed to make BaTiO3; its coefficient comes out about
    -3e-16. That must read as zero, not as consumed."""
    from pymatgen.entries.computed_entries import ComputedEntry
    table = _table(-21.2)
    table["SnO2"] = ComputedEntry("SnO2", -8.0)
    assert selectivity.reaction_free_energy(["Ba2TiO4", "TiO2", "SnO2"],
                                            "BaTiO3", 800, table) is None
    assert selectivity.reaction_free_energy(["Ba2TiO4", "TiO2"], "BaTiO3",
                                            800, table) is not None
