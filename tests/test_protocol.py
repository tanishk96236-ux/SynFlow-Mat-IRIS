"""The lab protocol -- the output this project is FOR.

Locked on 2026-10-06 in docs/target_output.md. Everything here is derived or
conventional rather than predicted, so these tests check arithmetic and
rules, and every number in them can be reproduced by hand.
"""

import pytest

from synflow import protocol as P


# --------------------------------------------------------------------------
# grams: the mass balance must close
# --------------------------------------------------------------------------

def test_the_masses_close_against_the_gas_that_leaves():
    """BaCO3 + TiO2 -> BaTiO3 + CO2, for 2 g of BaTiO3.

        BaTiO3  233.19 g/mol -> 2 g is 0.008577 mol
        BaCO3   0.008577 x 197.34 = 1.6925 g
        TiO2    0.008577 x  79.87 = 0.6850 g
        in      2.3775 g,  out 2.0000 g,  difference 0.3775 g
        CO2     0.008577 x  44.01 = 0.3775 g    <- exactly the difference
    """
    rows = P.weigh_out("BaCO3 + TiO2 -> BaTiO3 + CO2", "BaTiO3", batch_g=2.0)
    grams = {r["precursor"]: r["grams"] for r in rows}
    assert grams["BaCO3"] == pytest.approx(1.6925, abs=1e-3)
    assert grams["TiO2"] == pytest.approx(0.6850, abs=1e-3)

    co2 = P.molar_mass("CO2") * 2.0 / P.molar_mass("BaTiO3")
    assert sum(grams.values()) - 2.0 == pytest.approx(co2, abs=1e-3)


def test_fractional_coefficients_scale_correctly():
    """KRhO2 needs half a formula unit of each precursor."""
    rows = P.weigh_out("0.5 K2CO3 + 0.5 Rh2O3 -> KRhO2 + 0.5 CO2", "KRhO2")
    grams = {r["precursor"]: r["grams"] for r in rows}
    assert grams["K2CO3"] == pytest.approx(0.7943, abs=1e-3)
    assert grams["Rh2O3"] == pytest.approx(1.4587, abs=1e-3)


def test_oxygen_is_never_put_on_the_balance():
    """O2 consumed by the reaction comes from the furnace atmosphere. Listing
    it as something to weigh would be an instruction no one can follow."""
    rows = P.weigh_out(
        "0.3333 Co3O4 + 0.5 Li2CO3 + 0.08333 O2 -> LiCoO2 + 0.5 CO2", "LiCoO2")
    assert "O2" not in {r["precursor"] for r in rows}
    assert {r["precursor"] for r in rows} == {"Co3O4", "Li2CO3"}


def test_the_corpus_equation_format_is_read_too():
    """The corpus writes `==` and an explicit coefficient of one."""
    rows = P.weigh_out("1 CuO + 1 WO3 == 1 CuWO4", "CuWO4")
    assert {r["precursor"] for r in rows} == {"CuO", "WO3"}


def test_an_unreadable_equation_gives_no_masses_rather_than_wrong_ones():
    assert P.weigh_out("", "BaTiO3") is None
    assert P.weigh_out("not an equation", "BaTiO3") is None


# --------------------------------------------------------------------------
# atmosphere: which side oxygen is on decides it
# --------------------------------------------------------------------------

def test_oxygen_consumed_means_the_furnace_must_supply_it():
    """Co averages +2.67 in Co3O4 and must reach +3 in LiCoO2."""
    gas = P.atmosphere(
        "0.3333 Co3O4 + 0.5 Li2CO3 + 0.08333 O2 -> LiCoO2 + 0.5 CO2")
    assert "O2" in gas["atmosphere"]
    assert gas["oxygen_coefficient"] == pytest.approx(0.08333)


def test_oxygen_released_means_inert_not_air():
    """Fe is +3 in Fe2O3 and +2 in LiFePO4. Air would oxidise it back, which
    is why LiFePO4 is fired under argon. The first mock of the locked output
    said "air (O2 on the product side)" -- this is the rule that says no."""
    gas = P.atmosphere("0.5 Li2CO3 + 0.5 Fe2O3 + NH4H2PO4 -> LiFePO4 "
                       "+ 0.5 CO2 + 1.5 H2O + NH3 + 0.25 O2")
    assert "inert" in gas["atmosphere"]
    assert "air" not in gas["atmosphere"].split()[0]


def test_no_oxygen_exchange_means_plain_air():
    gas = P.atmosphere("BaCO3 + TiO2 -> BaTiO3 + CO2")
    assert gas["atmosphere"] == "air"
    assert gas["oxygen_coefficient"] == 0.0


# --------------------------------------------------------------------------
# what the printed protocol says
# --------------------------------------------------------------------------

def _rendered(onset, force, risk):
    lab = P.build("LiCoO2", ["Co3O4", "Li2CO3"], 707,
                  equation="0.3333 Co3O4 + 0.5 Li2CO3 + 0.08333 O2 "
                           "-> LiCoO2 + 0.5 CO2",
                  assessment={"feasible": True, "enthalpy_eV": 0.118,
                              "entropy_term_eV": -2.172,
                              "free_energy_eV": -2.054,
                              "onset_temperature_C": onset,
                              "driving_force_meV_per_atom": force,
                              "kinetic_risk": risk})
    return P.render(lab)


def test_a_sub_zero_floor_is_not_printed_as_a_temperature():
    """LiCoO2 came out at "floor about -220 C" two lines above a verdict
    saying "downhill at any practical temperature" -- the same fact, stated so
    that one line looked like an error."""
    text = _rendered(onset=-220.0, force=373.0, risk=False)
    assert "-220" not in text
    assert "any practical temperature" in text


def test_the_driving_force_is_always_shown_so_the_check_is_visible():
    """A flag that only appears when it fires leaves the reader unable to
    tell "checked and fine" from "never checked"."""
    text = _rendered(onset=-220.0, force=373.0, risk=False)
    assert "373 meV/atom" in text
    assert "above the 50 meV/atom" in text


def test_a_weak_reaction_is_flagged():
    text = _rendered(onset=-220.0, force=12.0, risk=True)
    assert "KINETIC RISK" in text


def test_every_field_states_where_it_came_from():
    """Only predicted fields carry a measured error. A protocol that mixes
    them with arithmetic invites the reader to doubt all of it equally."""
    lab = P.build("BaTiO3", ["BaCO3", "TiO2"], 1176,
                  equation="BaCO3 + TiO2 -> BaTiO3 + CO2")
    kinds = lab["provenance"]
    assert kinds["precursors"].startswith("predicted")
    assert kinds["firing_temperature_C"].startswith("predicted")
    assert kinds["masses_g"].startswith("derived")
    assert kinds["atmosphere"].startswith("derived")
    assert kinds["ramp_rate_C_per_min"].startswith("convention")


def test_firing_cycles_come_from_the_record_when_there_is_one():
    assert P.firing_steps(("Mixing", "Heating", "Mixing", "Heating"))[
        "firing_cycles"] == 2
    assert P.firing_steps(("Heating",))["firing_cycles"] == 1
    # not recorded: the standard two, and it says so
    default = P.firing_steps(())
    assert default["firing_cycles"] == 2
    assert not default["from_record"]


def test_purity_is_stated_and_labelled_a_convention():
    """No corpus records purity per recipe, so it must not read as derived."""
    lab = P.build("BaTiO3", ["BaCO3", "TiO2"], 1176,
                  equation="BaCO3 + TiO2 -> BaTiO3 + CO2")
    assert lab["provenance"]["purity"].startswith("convention")
    assert "99.9%" in P.render(lab)


# --------------------------------------------------------------------------
# the predicted temperature against the simulator's floor and ceiling
# --------------------------------------------------------------------------

def test_below_the_floor_everywhere_says_how_hot_to_go():
    notes = P.temperature_checks(800, {"onset_temperature_C": 886,
                                       "in_air": {"onset_temperature_C": 850}})
    assert len(notes) == 1 and "BELOW" in notes[0] and "900" in notes[0]


def test_below_the_one_bar_floor_but_above_air_needs_an_open_crucible():
    """KRhO2's case: 1 bar floor 886 C, in air 617 C."""
    notes = P.temperature_checks(772, {"onset_temperature_C": 886,
                                       "in_air": {"onset_temperature_C": 617}})
    assert len(notes) == 1 and "open crucible" in notes[0]


def test_above_a_ceiling_is_named():
    notes = P.temperature_checks(1000, {"onset_temperature_C": None,
                                        "ceiling_temperature_C": 900})
    assert any("ABOVE the ceiling" in n for n in notes)


def test_a_temperature_beyond_a_box_furnace_is_flagged():
    assert any("box furnace" in n for n in P.temperature_checks(1445, {}))
    assert P.temperature_checks(1100, {"onset_temperature_C": 300}) == []


def test_the_checks_are_printed_in_the_firing_step():
    lab = P.build("KRhO2", ["K2CO3", "Rh2O3"], 772,
                  equation="0.5 K2CO3 + 0.5 Rh2O3 -> KRhO2 + 0.5 CO2",
                  assessment={"onset_temperature_C": 886,
                              "in_air": {"onset_temperature_C": 617}})
    assert "CHECK:" in P.render(lab)


# --------------------------------------------------------------------------
# volatility -- the failure MatterGen's lab hit twice
# --------------------------------------------------------------------------

def _genome_present():
    import os

    from synflow import kinetics
    return os.path.exists(kinetics.LEDGER)


@pytest.mark.skipif(not _genome_present(), reason="Precursor Genome absent")
def test_molybdenum_is_flagged_where_the_lab_lost_it():
    """MatterGen SI D.10.5: Cr2MoO6 and LaMoO4 lost Mo at 900 C; at 750 C
    nothing evaporated. MoO3 melts at 795 C (Precursor Genome table)."""
    from synflow import volatility
    assert [w["element"] for w in volatility.check(["Cr2O3", "MoO3"], 900)] \
        == ["Mo"]
    assert volatility.check(["Cr2O3", "MoO3"], 750) == []
    # the lab's LaMoO4 used Mo METAL, which oxidises to MoO3 in air
    assert [w["element"] for w in volatility.check(["La2O3", "Mo"], 900)] \
        == ["Mo"]
    # TaCr2O6 (made) and Mn3NiO6 (failed otherwise) are not flagged
    assert volatility.check(["Cr", "Ta2O5"], 1500) == []
    assert volatility.check(["Mn2O3", "Ni2O3"], 1100) == []


@pytest.mark.skipif(not _genome_present(), reason="Precursor Genome absent")
def test_the_warning_reaches_the_printed_protocol():
    lab = P.build("Cr2MoO6", ["Cr2O3", "MoO3"], 900,
                  equation="Cr2O3 + MoO3 -> Cr2MoO6")
    assert "VOLATILITY" in P.render(lab)
    assert lab["provenance"]["volatility"].startswith("convention")
