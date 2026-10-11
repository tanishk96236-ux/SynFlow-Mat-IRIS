"""Will the recipe work? The thermodynamics, checked by hand.

DAO-S predicts precursors and a firing temperature and nothing tested either.
This module answers three questions DAO-S cannot ask about its own output: is
the reaction downhill, at what temperature does it become downhill, and does
that agree with the temperature it chose.

The term that decides almost every real recipe is the entropy of the gas that
leaves:

    K2CO3 + Rh2O3  ->  2 KRhO2 + CO2(gas)

Every energy in this repository is a 0 K enthalpy, so without the gas term this
reaction is uphill -- and so is every carbonate route, which is most of
solid-state synthesis. S°(CO2) is 213.8 J/mol/K, so at 1045 K the -T*dS term is
about -2.3 eV per CO2. That is why carbonates are the precursor of choice.

Omitting it would not error. It would report that the standard route to
LiFePO4 cannot work.

These tests need no network: the entropy arithmetic and the gas accounting are
where the physics is, and both are checkable by hand.
"""

import pytest

from synflow import reaction as R


# --------------------------------------------------------------------------
# the entropy term
# --------------------------------------------------------------------------

def test_one_carbon_dioxide_at_firing_temperature():
    """-T*S with S = 213.8 J/mol/K at 772 C, converted to eV.

        -213.8 * 1045.15 / 96485 = -2.316 eV
    """
    assert R.entropy_term({"CO2": 1.0}, 772) == pytest.approx(-2.316, abs=0.01)


def test_the_term_grows_with_temperature_and_is_always_negative():
    """It pays for the reaction, so it can only ever help."""
    values = [R.entropy_term({"CO2": 1.0}, t) for t in (25, 500, 772, 1000)]
    assert all(v < 0 for v in values)
    assert values == sorted(values, reverse=True)
    # and it is large: one carbonate at 1000 C is nearly 3 eV
    assert values[-1] < -2.5


def test_no_gas_means_no_entropy_help():
    """MgO + NiO releases nothing, so that reaction must be downhill on its
    own enthalpy or it does not happen. A screen that gave it a free 2 eV
    would accept routes that cannot work."""
    assert R.entropy_term({}, 997) == 0.0
    assert R.entropy_term(R.released_gases(["MgO", "NiO"]), 997) == 0.0


def test_an_unknown_gas_contributes_nothing_rather_than_guessing():
    assert R.entropy_term({"XeF6": 1.0}, 800) == 0.0


# --------------------------------------------------------------------------
# which gases leave
# --------------------------------------------------------------------------

def test_carbonates_release_carbon_dioxide():
    assert R.released_gases(["K2CO3", "Rh2O3"]) == {"CO2": 1.0}
    assert R.released_gases(["Li2CO3", "MoO3"]) == {"CO2": 1.0}
    assert R.released_gases(["BaCO3", "TiO2"]) == {"CO2": 1.0}


def test_an_ammonium_phosphate_releases_both_ammonia_and_water():
    """NH4H2PO4 has six hydrogens and one nitrogen: three go to ammonia and
    three to water. An earlier version dropped the water whenever nitrogen was
    present, which understated the entropy term for exactly the phosphate
    precursors every olivine recipe uses."""
    gases = R.released_gases(["NH4H2PO4"])
    assert gases["NH3"] == pytest.approx(1.0)
    assert gases["H2O"] == pytest.approx(1.5), gases


def test_a_nitrate_gives_NO2_and_an_ammonium_salt_gives_NH3():
    """Decided by whether there is enough hydrogen for NH4."""
    assert "NO2" in R.released_gases(["Ca(NO3)2"])
    assert "NH3" in R.released_gases(["NH4NO3"]) or \
           "NO2" in R.released_gases(["NH4NO3"])


def test_plain_oxides_release_nothing():
    for precursors in (["MgO", "NiO"], ["Ag2O", "WO3"], ["CuO", "Sc2O3"],
                       ["Nb2O5", "PbO"]):
        assert R.released_gases(precursors) == {}, precursors


def test_a_hydroxide_releases_water():
    gases = R.released_gases(["Ca(OH)2"])
    assert gases.get("H2O") == pytest.approx(1.0)


def test_an_unparseable_formula_is_skipped_not_fatal():
    """A recipe may name something the parser cannot read; that must not stop
    the check on the rest."""
    assert R.released_gases(["not a formula", "K2CO3"]) == {"CO2": 1.0}


# --------------------------------------------------------------------------
# what the verdict says
# --------------------------------------------------------------------------

def test_an_uphill_reaction_with_no_gas_is_named_as_hopeless():
    """No gas is released, so nothing can pay for a positive enthalpy at any
    temperature. Saying "needs a higher temperature" there would be wrong."""
    text = R.verdict({"feasible": False, "enthalpy_eV": 1.2,
                      "entropy_term_eV": 0.0, "free_energy_eV": 1.2,
                      "temperature_C": 900,
                      "onset_temperature_C": float("inf")})
    assert "no gas" in text


def test_an_uphill_reaction_with_gas_names_the_temperature_it_needs():
    text = R.verdict({"feasible": False, "enthalpy_eV": 2.8,
                      "entropy_term_eV": -2.3, "free_energy_eV": 0.5,
                      "temperature_C": 700, "onset_temperature_C": 1150})
    assert "1150" in text


def test_the_onset_is_a_floor_and_not_a_prediction():
    """These two temperatures measure different things, and an earlier version
    of the verdict compared them as if they should agree.

        onset   thermodynamic: below it the reaction CANNOT go
        DAO-S   practical: where it goes at a useful rate

    Measured on the BaTiO3 control: onset 255 C, DAO-S 1100 C, and BaTiO3 is
    genuinely fired at 1100-1300 C. The gap is kinetics -- diffusion,
    nucleation, time -- which this module does not model.

    So the check is ONE-SIDED. Far above the floor is normal. Below it is the
    only inconsistency there is.
    """
    normal = R.verdict({"feasible": True, "enthalpy_eV": 1.17,
                        "entropy_term_eV": -3.04, "free_energy_eV": -1.87,
                        "temperature_C": 1100, "onset_temperature_C": 255})
    assert "consistent" in normal
    assert "kinetics" in normal
    assert "INCONSISTENT" not in normal

    below = R.verdict({"feasible": True, "enthalpy_eV": 2.2,
                       "entropy_term_eV": -2.3, "free_energy_eV": -0.1,
                       "temperature_C": 400, "onset_temperature_C": 900})
    assert "INCONSISTENT" in below
    assert "900" in below and "400" in below


def test_the_batio3_control_numbers():
    """The control that validated the whole approach, kept as a regression.

        BaCO3 + TiO2 -> BaTiO3 + CO2
        dH  +1.171 eV   uphill at 0 K -- without the gas term this reads as
                        impossible, and BaTiO3 is in every capacitor
        gas -3.043 eV   at 1100 C
        dG  -1.872 eV
    """
    gases = R.released_gases(["BaCO3", "TiO2"])
    assert gases == {"CO2": 1.0}
    assert R.entropy_term(gases, 1100) == pytest.approx(-3.043, abs=0.02)
    # and the enthalpy alone would have rejected it
    assert 1.171 - 3.043 < 0 < 1.171


# --------------------------------------------------------------------------
# gases from the balanced equation, not from the precursors
# --------------------------------------------------------------------------
#
# The entropy term used to be built from released_gases(precursors): one CO2
# per carbonate formula unit, and no oxygen ever. On the A-Lab's reactions
# that was wrong in both directions at once:
#
#     2 MnO2 -> Mn2O3 + 0.5 O2           counted 0 O2     rejected; it formed
#     2 BaCO3 + SnO2 -> Ba2SnO4 + 2 CO2  counted 1 CO2    rejected; it formed
#     0.5 K2CO3 + ... -> KRhO2 + 0.5 CO2 counted 1 CO2    twice the help
#
# Five of the eight robot-made products the gate rejected were this.

def _balanced(reactants, products, target):
    from pymatgen.analysis.reaction_calculator import Reaction
    from pymatgen.core import Composition

    reaction = Reaction([Composition(f) for f in reactants],
                        [Composition(f) for f in products])
    reaction.normalize_to(Composition(target).reduced_composition, 1.0)
    return reaction


def test_oxygen_released_by_a_change_of_oxidation_state_is_counted():
    """Mn(4+) -> Mn(3+): no precursor 'releases' a gas, and yet half an O2
    leaves per Mn2O3. That half mole is the whole reason MnO2 decomposes."""
    reaction = _balanced(["MnO2"], ["Mn2O3", "O2"], "Mn2O3")
    assert R.gas_coefficients(reaction) == pytest.approx({"O2": 0.5})
    assert R.released_gases(["MnO2"]) == {}          # what used to be used


def test_two_carbonates_release_two_carbon_dioxides():
    reaction = _balanced(["BaCO3", "SnO2"], ["Ba2SnO4", "CO2", "O2"],
                         "Ba2SnO4")
    assert R.gas_coefficients(reaction) == pytest.approx({"CO2": 2.0})
    assert R.released_gases(["BaCO3", "SnO2"]) == {"CO2": 1.0}   # half


def test_per_formula_unit_of_target_half_a_carbonate_is_half_a_gas():
    reaction = _balanced(["K2CO3", "Rh2O3"], ["KRhO2", "CO2", "O2"], "KRhO2")
    assert R.gas_coefficients(reaction) == pytest.approx({"CO2": 0.5})


def test_oxygen_taken_up_is_counted_negative():
    """Co3O4 is Co(+2.67) and LiCoO2 is Co(+3): the furnace supplies oxygen,
    and taking a gas in costs entropy rather than paying it."""
    reaction = _balanced(["Li2CO3", "Co3O4"], ["LiCoO2", "CO2", "O2"],
                         "LiCoO2")
    gases = R.gas_coefficients(reaction)
    assert gases["CO2"] == pytest.approx(0.5)
    assert gases["O2"] == pytest.approx(-1 / 12)


def test_ammonia_is_found_under_pymatgens_spelling():
    """pymatgen writes ammonia "H3N"; looked up as "NH3" it would vanish."""
    reaction = _balanced(["NH4H2PO4", "Li2CO3", "FeC2O4"],
                         ["LiFePO4", "NH3", "H2O", "CO2", "O2"], "LiFePO4")
    assert "NH3" in R.gas_coefficients(reaction)


def test_gas_taken_up_works_against_the_reaction():
    assert R.entropy_term({"O2": -0.25}, 800) > 0
    assert R.entropy_term({"O2": 0.25}, 800) < 0


# --------------------------------------------------------------------------
# the pressure a gas leaves into
# --------------------------------------------------------------------------

def test_a_gas_at_a_lower_pressure_has_more_entropy():
    """S = S° - R ln(p / 1 bar). CO2 leaving into air is at 4.2e-4 bar:
    213.8 + 8.314 * ln(1 / 4.2e-4) = 213.8 + 64.6 = 278.4 J/mol/K."""
    assert R.entropy_at("CO2", 4.2e-4) == pytest.approx(278.4, abs=0.2)
    assert R.entropy_at("O2", 0.2095) == pytest.approx(218.2, abs=0.2)
    assert R.entropy_at("CO2") == R.GAS_ENTROPY["CO2"]


def test_air_is_never_worse_than_one_bar_for_a_released_gas():
    """1 bar is the conservative bound: every pressure in AIR_BAR is below
    it, so a released gas can only help more in air."""
    for gas, pressure in R.AIR_BAR.items():
        assert pressure < 1.0, gas
        assert R.entropy_term({gas: 1.0}, 800, R.AIR_BAR) \
            < R.entropy_term({gas: 1.0}, 800)


def test_one_carbonate_in_air_at_1100_C():
    """-278.4 * 1373.15 / 96485 = -3.96 eV, against -3.04 eV at 1 bar."""
    assert R.entropy_term({"CO2": 1.0}, 1100, R.AIR_BAR) == \
        pytest.approx(-3.962, abs=0.01)


def test_oxygen_taken_from_air_costs_more_than_from_one_bar():
    """Air holds O2 at 0.21 bar, so each mole taken from it gives up more
    entropy than a mole at 1 bar: the ceiling comes down in air."""
    assert R.entropy_term({"O2": -0.25}, 900, R.AIR_BAR) \
        > R.entropy_term({"O2": -0.25}, 900)
    one_bar = R.crossover(-0.5, {"O2": -0.25})["temperature_C"]
    in_air = R.crossover(-0.5, {"O2": -0.25}, R.AIR_BAR)["temperature_C"]
    assert in_air < one_bar


def test_gases_air_does_not_fix_stay_at_one_bar():
    """Water depends on the weather and air has no ammonia; giving either a
    pressure would mean choosing a number to fit."""
    assert "H2O" not in R.AIR_BAR and "NH3" not in R.AIR_BAR
    assert R.entropy_term({"H2O": 1.0}, 800, R.AIR_BAR) == \
        R.entropy_term({"H2O": 1.0}, 800)


# --------------------------------------------------------------------------
# floor, ceiling, always, never
# --------------------------------------------------------------------------

def test_mno2_floor_from_the_measured_enthalpy():
    """dH +0.867 eV, 0.5 O2 at 205.2 J/mol/K:

        T = 0.867 * 96485 / (0.5 * 205.2) = 815 K = 542 C

    The A-Lab saw Mn2O3 by 600 C. The old simulator, with no O2 counted,
    had no floor at all and rejected it outright."""
    crossing = R.crossover(0.867, {"O2": 0.5})
    assert crossing["kind"] == "floor"
    assert crossing["temperature_C"] == pytest.approx(542, abs=2)


def test_a_reaction_that_takes_up_oxygen_has_a_ceiling():
    """dH -0.5 eV, 0.25 O2 consumed: downhill below
    0.5 * 96485 / (0.25 * 205.2) = 940 K = 667 C, uphill above it."""
    crossing = R.crossover(-0.5, {"O2": -0.25})
    assert crossing["kind"] == "ceiling"
    assert crossing["temperature_C"] == pytest.approx(667, abs=2)


def test_the_four_kinds():
    assert R.crossover(-1.0, {"CO2": 1.0})["kind"] == "always"
    assert R.crossover(-1.0, {})["kind"] == "always"
    assert R.crossover(1.0, {})["kind"] == "never"
    assert R.crossover(1.0, {"O2": -0.5})["kind"] == "never"


def test_a_ceiling_is_named_in_the_verdict():
    text = R.verdict({"feasible": True, "enthalpy_eV": -0.5,
                      "entropy_term_eV": 0.4, "free_energy_eV": -0.1,
                      "temperature_C": 600, "onset_temperature_C": None,
                      "ceiling_temperature_C": 667, "gases": {"O2": -0.25}})
    assert "ceiling" in text and "667" in text and "O2" in text

    above = R.verdict({"feasible": False, "enthalpy_eV": -0.5,
                       "entropy_term_eV": 0.6, "free_energy_eV": 0.1,
                       "temperature_C": 900, "onset_temperature_C": None,
                       "ceiling_temperature_C": 667, "gases": {"O2": -0.25}})
    assert "uphill here" in above and "667" in above


def test_taking_up_gas_with_a_positive_enthalpy_is_hopeless():
    text = R.verdict({"feasible": False, "enthalpy_eV": 0.3,
                      "entropy_term_eV": 0.4, "free_energy_eV": 0.7,
                      "temperature_C": 800,
                      "onset_temperature_C": float("inf"),
                      "gases": {"O2": -0.25}})
    assert "every temperature" in text and "O2" in text


def _measured_data_present():
    import os

    from synflow import thermo_data
    return os.path.exists(thermo_data.DATASET)


@pytest.mark.skipif(not _measured_data_present(),
                    reason="data/thermo not downloaded")
def test_assess_counts_the_oxygen_mno2_releases():
    """End to end, offline: MnO2 and Mn2O3 are both calorimetric and O2 is
    in the gas table, so nothing here touches the network. The A-Lab made
    Mn2O3 from MnO2 by 600 C; this is the row the old gate rejected."""
    result = R.assess(["MnO2"], "Mn2O3", 600)
    assert result["gases"] == pytest.approx({"O2": 0.5})
    assert result["entropy_term_eV"] == pytest.approx(-0.928, abs=0.01)
    assert result["feasible"] is True
    assert result["onset_temperature_C"] == pytest.approx(542, abs=5)
    assert result["enthalpy_sources"]["MnO2"].startswith("measured")


@pytest.mark.skipif(not _measured_data_present(),
                    reason="data/thermo not downloaded")
def test_an_unstated_error_bar_is_listed_not_printed_as_nan():
    """Fe2O3's measured enthalpy has no stated uncertainty; the dataset
    stores NaN. It must be named as missing, not summed into the error.
    ZnO, Fe2O3 and ZnFe2O4 are all calorimetric, so this is offline."""
    import math

    result = R.reaction_enthalpy(["ZnO", "Fe2O3"], "ZnFe2O4")
    assert result["uncertainty_not_stated_for"] == ["Fe2O3"]
    assert math.isfinite(result["enthalpy_uncertainty_eV"])


@pytest.mark.skipif(not _measured_data_present(),
                    reason="data/thermo not downloaded")
def test_one_unmeasured_species_means_no_measured_reaction():
    """Never fill a gap from another source. Measured CuO and Ta2O5 with
    MP's CuTa2O6 gave -2.147 eV for an oxide + oxide reaction; all-MP gives
    +0.338. ZnNiO2 has no calorimetry, so this reaction must refuse."""
    with pytest.raises(R.NotAllMeasured) as gap:
        R.reaction_enthalpy(["ZnO", "NiO"], "ZnNiO2")
    assert gap.value.formulas == ["ZnNiO2"]


@pytest.mark.skipif(not _measured_data_present(),
                    reason="data/thermo not downloaded")
def test_the_measured_table_has_no_computed_fallback():
    from synflow import thermo_data
    assert thermo_data.formation_enthalpy("ZnNiO2") is None
    assert thermo_data.formation_enthalpy("TiO2")["source"].startswith(
        "measured")
    assert thermo_data.formation_enthalpy("NH3")["source"] == "NIST-JANAF gas"


@pytest.mark.skipif(not _measured_data_present(),
                    reason="data/thermo not downloaded")
def test_a_target_written_unreduced_still_balances():
    """The A-Lab writes Mn2P4O12, whose reduced formula is MnP2O6. The
    measured path balanced the formula as written and then normalised to
    the reduced one, which was not in the reaction: seven rows came back
    "could not balance". It must reach the measured-data check instead."""
    with pytest.raises(R.NotAllMeasured):
        R.reaction_enthalpy(["MnO", "NH4H2PO4"], "Mn2P4O12")


def test_corpus_names_become_readable_formulas():
    """Five of the first 39 real recipes failed on notation alone."""
    from pymatgen.core import Composition

    assert R.readable("FeC2O4·2H2O") == "FeC2O4(H2O)2"
    assert R.readable("Bi(NO3)3·5H2O") == "Bi(NO3)3(H2O)5"
    assert R.readable("α-Al2O3") == "Al2O3"
    assert R.readable("BaCO3") == "BaCO3"
    # and pymatgen can now read every one of them
    for name in ("FeC2O4·2H2O", "Fe(NO3)3·9H2O", "CH3COOLi·2H2O",
                 "α-Al2O3"):
        Composition(R.readable(name))
    # the hydrate's water is released like any other
    assert R.released_gases([R.readable("FeC2O4·2H2O")])["H2O"] \
        == pytest.approx(2.0)


# --------------------------------------------------------------------------
# the error band, and the verdict it allows
# --------------------------------------------------------------------------

def test_the_band_for_an_all_mp_reaction_is_bartels_35():
    """Bartel et al. 2019: compound-only reactions, DFT vs experiment within
    about 35 meV/atom. Taken from the paper, fitted to nothing here."""
    result = {"enthalpy_source": "Materials Project, every species"}
    assert R.error_band_meV_per_atom(result, 10) == pytest.approx(35.0)


def test_a_scored_structure_adds_its_calibration_error_in_quadrature():
    """Target error 0.5 eV over 10 reactant atoms: 2 sigma = 100 meV/atom,
    so the band is sqrt(35^2 + 100^2) = 105.9 meV/atom."""
    result = {"enthalpy_source": "Materials Project, every species",
              "target_energy_error_eV": 0.5}
    assert R.error_band_meV_per_atom(result, 10) == pytest.approx(105.95,
                                                                  abs=0.05)


def test_a_measured_band_is_two_sigma_and_none_when_a_sigma_is_missing():
    """MnO2 -> Mn2O3: sigma 0.035 eV over 2 MnO2 = 6 atoms -> 11.7."""
    measured = {"enthalpy_source": "measured, every species",
                "enthalpy_uncertainty_eV": 0.035,
                "uncertainty_not_stated_for": []}
    assert R.error_band_meV_per_atom(measured, 6) == pytest.approx(
        2000 * 0.035 / 6)
    measured["uncertainty_not_stated_for"] = ["Fe2O3"]
    assert R.error_band_meV_per_atom(measured, 6) is None


def test_a_reaction_inside_the_band_is_called_uncertain():
    """BiFeO3 from its oxides: +0.008 eV over 5 atoms, 1.6 meV/atom uphill.
    It was reported "rejected" as firmly as a reaction 300 meV/atom uphill."""
    text = R.verdict({"feasible": False, "enthalpy_eV": 0.008,
                      "entropy_term_eV": 0.0, "free_energy_eV": 0.008,
                      "temperature_C": 800,
                      "onset_temperature_C": float("inf"),
                      "driving_force_meV_per_atom": -1.6,
                      "error_band_meV_per_atom": 35.0,
                      "within_error": True})
    assert "WITHIN ERROR" in text and "35" in text


def test_a_polymorph_far_above_the_known_phase_is_flagged():
    base = {"feasible": True, "enthalpy_eV": -0.5, "entropy_term_eV": 0.0,
            "free_energy_eV": -0.5, "temperature_C": 900,
            "onset_temperature_C": None}
    high = R.verdict(dict(base, polymorph={"gap_eV_per_atom": 0.120}))
    assert "POLYMORPH" in high and "120" in high
    close = R.verdict(dict(base, polymorph={"gap_eV_per_atom": 0.010}))
    assert "POLYMORPH" not in close and "+10" in close


def test_a_failed_check_says_so_rather_than_guessing():
    text = R.verdict({"feasible": None, "error": "ValueError: no entry for X"})
    assert "could not check" in text
    assert "no entry" in text


# --------------------------------------------------------------------------
# the constants
# --------------------------------------------------------------------------

def test_the_entropies_are_the_tabulated_ones():
    """Standard molar entropies at 298 K, J/(mol K). Wrong by a factor and
    every carbonate route would be wrong by the same factor."""
    assert R.GAS_ENTROPY["CO2"] == pytest.approx(213.8, abs=0.5)
    assert R.GAS_ENTROPY["O2"] == pytest.approx(205.2, abs=0.5)
    assert R.GAS_ENTROPY["H2O"] == pytest.approx(188.8, abs=0.5)
    assert R.GAS_ENTROPY["NH3"] == pytest.approx(192.8, abs=0.5)


def test_the_unit_conversion_is_per_electronvolt():
    """J/mol to eV is a division by Faraday's number in coulombs."""
    assert R.J_PER_MOL_TO_EV == pytest.approx(1 / 96485.0, rel=1e-6)


def test_an_unstable_novel_target_is_named():
    """Anchored at 120 meV/atom above Mn2NiO4 + MnO2: the lab got Mn2NiO4."""
    base = {"feasible": True, "enthalpy_eV": -0.98, "entropy_term_eV": 0.0,
            "free_energy_eV": -0.98, "temperature_C": 1100,
            "onset_temperature_C": None}
    text = R.verdict(dict(base, target_stability={
        "above_hull_eV_per_atom": 0.120,
        "decomposes_to": [("Mn2NiO4", 0.7), ("MnO2", 0.3)]}))
    assert "TARGET UNSTABLE" in text and "Mn2NiO4" in text
    stable = R.verdict(dict(base, target_stability={
        "above_hull_eV_per_atom": 0.010,
        "decomposes_to": [("Mn2NiO4", 1.0)]}))
    assert "UNSTABLE" not in stable and "+10" in stable
