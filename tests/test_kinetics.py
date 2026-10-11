"""The kinetic layer: what the robots saw, read through monotonicity.

The one certain fact about solid-state kinetics used here: more heat and
more time never undo a reaction. So a pair that stayed unreacted at 1100 C
will stay unreacted at 900 C, and a pair that reacted at 400 C will react at
900 C. Everything else -- the time-temperature exchange rate, which outcome
counts as "nothing happened" -- is checked below by hand.

Needs the Precursor Genome ledger in data/precursor_genome; skipped without.
"""

import math
import os

import pytest

from synflow import kinetics

present = pytest.mark.skipif(not os.path.exists(kinetics.LEDGER),
                             reason="Precursor Genome not downloaded")


@present
def test_the_garnet_pair_the_simulator_got_wrong_is_blocked():
    """Y2O3 + Ga2O3 is downhill to the garnet, and the A-Lab saw nothing
    form at 700 or 900 C. The Precursor Genome saw nothing at 1100 C either.
    That settles both Table 2 rows."""
    for temperature in (700, 900):
        found = kinetics.evidence("Y2O3", "Ga2O3", temperature, 240)
        assert found["verdict"] == "blocked"
        assert found["seen"][0] == 1100.0


@present
def test_a_shorter_hold_only_speaks_for_a_lower_temperature():
    """Seen unreacted at 1100 C for 1 h. A 4 h hold gets a 50 K margin, from
    1/T1 - 1/T2 = R ln 4 / Q with Q ~ 300 kJ/mol: 1100 C settles 1040 C, not
    1080 C."""
    assert kinetics.evidence("Y2O3", "Ga2O3", 1040, 240)["verdict"] \
        == "blocked"
    assert kinetics.evidence("Y2O3", "Ga2O3", 1080, 240) is None
    # with the same one-hour hold, no margin is needed
    assert kinetics.evidence("Y2O3", "Ga2O3", 1080, 60)["verdict"] \
        == "blocked"


def test_the_margin_is_the_arrhenius_number():
    """R ln 4 / Q at Q = 300 kJ/mol, around 1150 K, is about 50 K."""
    T = 1150.0
    shift = T ** 2 * 8.314 * math.log(4) / 300e3
    assert 40 < shift < 65
    assert kinetics.TIME_MARGIN_K == 50.0


@present
def test_transformed_is_not_evidence_of_no_reaction():
    """MgO + NiO at 1100 C was classified "transformed" with metallic Mg --
    in air, impossible; it is the (Mg,Ni)O solid solution. The A-Lab made
    MgNiO2 at 900 C, so this pair must not be blocked."""
    assert kinetics.evidence("MgO", "NiO", 900, 240) is None


@present
def test_a_pair_seen_reacting_cooler_is_allowed():
    found = kinetics.evidence("MgO", "V2O5", 900, 240)
    assert found["verdict"] == "observed"
    assert found["seen"][0] <= 900


@present
def test_the_tammann_ratio_by_hand():
    """Y2O3 melts at 2440 C, Ga2O3 at 1900 C. At 700 C:
    h_min = 973.15 / 2173.15 = 0.448 -- far below Tammann's 2/3."""
    x = kinetics.features("Y2O3", "Ga2O3", 700, 240)
    assert x["h_min"] == pytest.approx(973.15 / 2173.15, abs=1e-4)
    assert x["h_max"] == pytest.approx(973.15 / 2713.15, abs=1e-4)


@present
def test_an_unknown_melting_point_abstains():
    assert kinetics.melting_point("CaGd3O6") is None
    assert kinetics.features("CaGd3O6", "TiO2", 900, 240) is None
    assert kinetics.evidence("CaGd3O6", "TiO2", 900, 240) is None


@present
def test_a_recipe_is_checked_pair_by_pair():
    report = kinetics.check_recipe(["Y2O3", "Ga2O3", "In2O3"], 900, 4)
    assert len(report["pairs"]) == 3
    assert ("Ga2O3", "Y2O3") in report["blocked_pairs"]


def test_decomposing_precursors_are_recognised():
    for formula in ("BaCO3", "Ca(OH)2", "NH4H2PO4", "FeC2O4", "BaO2"):
        assert kinetics.decomposes(formula), formula
    for formula in ("TiO2", "Y2O3", "NiO"):
        assert not kinetics.decomposes(formula), formula
