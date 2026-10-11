"""Anion class and oxidation state -- the context the precursor rule ignored.

Every case here can be checked by hand, and most of them are the textbook
calculation: hold everything else at its usual charge, require the total to
be zero, solve for the one element wanted.
"""

from synflow import valence as V
from synflow.recipe_metrics import ConditionalLookup, MetalLookupBaseline
from synflow.recipes import Recipe


# --------------------------------------------------------------------------
# anion class
# --------------------------------------------------------------------------

def test_anion_class_follows_the_priority_order():
    assert V.anion_class("BaTiO3") == "oxide"
    assert V.anion_class("LiFePO4") == "oxide"          # phosphates are oxides
    assert V.anion_class("CdS") == "chalcogenide"
    assert V.anion_class("GaN") == "nitride"
    assert V.anion_class("LiF") == "halide"
    assert V.anion_class("NbC") == "carbide"
    assert V.anion_class("MgB2") == "boride"
    assert V.anion_class("TaAs") == "pnictide"
    assert V.anion_class("NiTi2") == "intermetallic"


def test_oxygen_wins_over_everything_else():
    """An oxysulfide or oxyhalide is supplied the way an oxide is."""
    assert V.anion_class("La2O2S") == "oxide"
    assert V.anion_class("YOCl") == "oxide"


def test_an_unreadable_formula_has_no_class():
    assert V.anion_class("") is None
    assert V.anion_class("(Ba1-xCax)TiO3") is None


# --------------------------------------------------------------------------
# the usual states come from the ICSD, and are positive
# --------------------------------------------------------------------------

def test_the_usual_states_are_the_oxoanion_ones():
    """Phosphorus is -3 in a phosphide and +5 in a phosphate. In an oxide it
    is always the second, so the table keeps the most frequent POSITIVE
    state -- the most frequent overall would be wrong in exactly the
    compounds this is used on."""
    usual = V._icsd_most_common()
    assert usual["P"] == 5
    assert usual["S"] == 6
    assert usual["C"] == 4
    assert usual["Fe"] == 3
    assert usual["Ti"] == 4


# --------------------------------------------------------------------------
# oxidation state
# --------------------------------------------------------------------------

def test_the_iron_states_that_decide_the_precursor():
    """The Fe(2+)/Fe(3+) split is the result this module exists for."""
    assert V.oxidation_state("LiFePO4", "Fe") == 2
    assert V.oxidation_state("Fe2O3", "Fe") == 3
    assert V.oxidation_state("BaFe12O19", "Fe") == 3


def test_a_doped_formula_is_solved_by_charge_balance():
    """pymatgen's enumeration returns nothing for every decimal oxide --
    0 of 118 in a measured sample -- because it needs an integer number of
    sites. By hand:

        Ba +2, Zr +4 x 0.05 = +0.2, O -2 x 3 = -6
        so 0.95 x Ti = +3.8 and Ti = +4
    """
    assert V.oxidation_state("BaZr0.05Ti0.95O3", "Ti") == 4
    assert V.oxidation_state("LiNi0.5Mn1.5O4", "Mn") == 4


def test_a_state_the_element_cannot_take_is_refused():
    """Li0.75La3Zr1.75Nb0.25O12 is a lithium garnet whose real formula is
    Li6.75...: the extraction dropped the 6. Charge balance then demanded
    Nb +29. Refusing is what lets the precursor rule back off instead of
    acting on a composition that cannot exist."""
    for element in ("Li", "La", "Zr", "Nb"):
        assert V.oxidation_state("Li0.75La3Zr1.75Nb0.25O12", element) is None


def test_an_intermetallic_is_zero_valent():
    assert V.oxidation_state("NiTi2", "Ni") == 0


# --------------------------------------------------------------------------
# the conditional rule
# --------------------------------------------------------------------------

def _corpus():
    """Iron supplied two ways, depending on the target's iron state."""
    recipes = []
    for i in range(6):
        recipes.append(Recipe("LiFePO4", ["Li2CO3", "FeC2O4", "NH4H2PO4"],
                              (), (700,)))
        recipes.append(Recipe("Fe2O3", ["Fe2O3"], (), (900,)))
    for i in range(20):
        recipes.append(Recipe("BaFe12O19", ["BaCO3", "Fe2O3"], (), (1200,)))
    return recipes


def test_it_picks_by_the_iron_state():
    """The plain rule sends Fe2O3 to every iron target, because Fe2O3 is the
    commonest iron source overall. Looking at the target, an Fe(2+) phosphate
    gets the Fe(2+) oxalate it was actually made from."""
    plain = MetalLookupBaseline(_corpus())
    smart = ConditionalLookup(_corpus(), min_count=5)

    assert "Fe2O3" in plain.predict("LiFePO4")
    assert "FeC2O4" in smart.predict("LiFePO4")
    assert "Fe2O3" in smart.predict("BaFe12O19")


def test_an_untrusted_key_falls_back_to_the_plain_rule():
    """Below min_count the specific level is not believed, so on any key it
    has too little evidence for, this IS the plain rule."""
    plain = MetalLookupBaseline(_corpus())
    smart = ConditionalLookup(_corpus(), min_count=1000)
    assert smart.predict("LiFePO4") == plain.predict("LiFePO4")


def test_explain_says_which_level_answered():
    smart = ConditionalLookup(_corpus(), min_count=5)
    rows = {r["metal"]: r for r in smart.explain("LiFePO4")}
    assert rows["Fe"]["level"] == "state"
    assert rows["Fe"]["state"] == 2
    assert rows["Fe"]["precursor"] == "FeC2O4"
