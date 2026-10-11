"""A recipe a chemist can weigh out, from a balanced equation.

DAO-S returns three fields -- target, precursors, firing temperature -- and
that is not a protocol. Nobody can take {K2CO3, Rh2O3} at 772 C into a lab,
because the first thing they need is how much of each to put on the balance.

WHAT IS CALCULATED HERE AND WHAT IS PREDICTED ELSEWHERE
-------------------------------------------------------
This module predicts nothing. Everything in it follows from the balanced
equation and the periodic table, so there is no accuracy to report for it --
only arithmetic to get right. Keeping that boundary visible matters, because
a protocol that mixes measured predictions with derived quantities invites
the reader to doubt all of it equally.

    predicted   which precursors        DAO-S, rule, 0.662 Jaccard unseen
    predicted   firing temperature      DAO-S, network, 192 C MAE unseen
    DERIVED     masses                  stoichiometry x molar mass
    DERIVED     atmosphere              which side of the equation O2 is on
    DERIVED     feasibility             reaction.assess
    CONVENTION  ramp rate, grinding     not in any corpus; stated as such

THE ATMOSPHERE IS THE INTERESTING DERIVATION
--------------------------------------------
It is not guessed. The balancer puts oxygen wherever the oxidation states
require, and which side it lands on is the answer:

    Co3O4 + Li2CO3 + O2 -> LiCoO2 + CO2       O2 consumed  -> fire in air/O2
    Li2CO3 + Fe2O3 + NH4H2PO4 -> LiFePO4 + O2 O2 released  -> inert or reducing

Co averages +2.67 in Co3O4 and must reach +3, so oxygen is taken in. Fe is
+3 in Fe2O3 and must fall to +2, so oxygen comes out -- and LiFePO4 is
genuinely fired under argon for that reason. Neither direction is hardcoded.
"""

import re
from typing import Dict, List, Optional

# Grinding and ramp rate are not recorded in either corpus, so they are
# conventions rather than predictions and are labelled that way in the
# output. These are the ordinary values for a solid-state oxide route.
RAMP_RATE_C_PER_MIN = 5.0
GRIND_MINUTES = 45

# Purity is not recorded per recipe in either corpus, so this too is a
# convention, and labelled as one. 99.9% is the ordinary grade for a
# solid-state oxide synthesis: impurities at the percent level are enough to
# stabilise a competing phase, and a carbonate that has absorbed water weighs
# out wrong -- the reason K2CO3 and Li2CO3 are dried before weighing.
PURITY = "99.9% or better; dry hygroscopic precursors (carbonates, " \
         "hydroxides) at 120-200 C before weighing"
DEFAULT_BATCH_G = 2.0

_TERM = re.compile(r"^\s*([0-9.]*)\s*([A-Za-z0-9()]+)\s*$")


def molar_mass(formula: str) -> Optional[float]:
    """g/mol, or None if the formula cannot be read."""
    from pymatgen.core import Composition
    try:
        return float(Composition(formula).weight)
    except Exception:                                           # noqa: BLE001
        return None


def parse_equation(equation: str) -> Optional[dict]:
    """A balanced equation string into coefficients per side.

    Accepts both the arrow this project writes and the `==` the corpus uses.
    """
    if not equation:
        return None
    text = equation.replace("==", "->")
    if "->" not in text:
        return None
    left, right = text.split("->", 1)

    def side(part):
        out = {}
        for term in part.split("+"):
            match = _TERM.match(term)
            if not match:
                return None
            coefficient, formula = match.groups()
            out[formula] = float(coefficient) if coefficient else 1.0
        return out

    reactants, products = side(left), side(right)
    if reactants is None or products is None:
        return None
    return {"reactants": reactants, "products": products}


def weigh_out(equation: str, target: str,
              batch_g: float = DEFAULT_BATCH_G) -> Optional[List[dict]]:
    """How many grams of each precursor, for a batch of this much target.

    Scaled so the target mass comes out at `batch_g`. A gram or two is the
    ordinary scale for a first attempt: enough for XRD and a pressed pellet,
    little enough that a wrong guess costs nothing.
    """
    parsed = parse_equation(equation)
    if parsed is None:
        return None

    from pymatgen.core import Composition
    try:
        wanted = Composition(target).reduced_formula
    except Exception:                                           # noqa: BLE001
        return None

    target_coefficient = None
    for formula, coefficient in parsed["products"].items():
        try:
            if Composition(formula).reduced_formula == wanted:
                target_coefficient = coefficient
        except Exception:                                       # noqa: BLE001
            continue
    target_mass = molar_mass(target)
    if not target_coefficient or not target_mass:
        return None

    # moles of target in one unit of the equation -> scale to the batch
    moles = batch_g / (target_coefficient * target_mass)

    rows = []
    for formula, coefficient in sorted(parsed["reactants"].items()):
        mass = molar_mass(formula)
        if mass is None or formula == "O2":
            continue        # oxygen comes from the atmosphere, not the balance
        rows.append({
            "precursor": formula,
            "coefficient": coefficient,
            "molar_mass_g_per_mol": round(mass, 3),
            "grams": round(coefficient * moles * mass, 4),
        })
    return rows


def atmosphere(equation: str) -> dict:
    """Which side oxygen is on, and what furnace gas that implies."""
    parsed = parse_equation(equation)
    if parsed is None:
        return {"atmosphere": None, "reason": "equation could not be read"}

    as_reactant = parsed["reactants"].get("O2", 0.0)
    as_product = parsed["products"].get("O2", 0.0)

    if as_reactant > 0:
        return {"atmosphere": "air or flowing O2",
                "oxygen_coefficient": as_reactant,
                "reason": "the balanced equation CONSUMES {:g} O2, so the "
                          "target is more oxidised than the precursors and "
                          "the furnace must supply oxygen".format(as_reactant)}
    if as_product > 0:
        return {"atmosphere": "inert (N2 or Ar), reducing if it persists",
                "oxygen_coefficient": as_product,
                "reason": "the balanced equation RELEASES {:g} O2, so the "
                          "target is more reduced than the precursors and air "
                          "would oxidise it back".format(as_product)}
    return {"atmosphere": "air",
            "oxygen_coefficient": 0.0,
            "reason": "no oxygen is exchanged, so the atmosphere is not "
                      "doing any chemistry"}


def firing_steps(operations=()) -> dict:
    """How many heating steps the recorded operations used.

    From the corpus where it is available, and 2 otherwise -- because a
    solid-state reaction stops itself: the product forms at the interface
    between the particles and then separates them, so the standard practice
    is to cool, regrind to break that layer up, and fire again.
    """
    heats = sum(1 for op in operations or () if str(op).lower() == "heating")
    return {"firing_cycles": max(heats, 1) if heats else 2,
            "from_record": bool(heats),
            "reason": ("counted from the recorded operations" if heats else
                       "not recorded; 2 is standard, because the product "
                       "layer separates the reactants and regrinding breaks "
                       "it up")}


def build(target: str, precursors, firing_temperature_C: float,
          equation: Optional[str] = None, assessment: Optional[dict] = None,
          operations=(), dwell_hours: Optional[float] = None,
          batch_g: float = DEFAULT_BATCH_G) -> dict:
    """Everything needed to attempt this synthesis, with its provenance.

    Every field says where it came from -- `predicted`, `derived` or
    `convention` -- so a reader can tell which numbers carry a measured error
    bar and which are arithmetic.
    """
    equation = equation or (assessment or {}).get("reaction")

    masses = weigh_out(equation, target, batch_g) if equation else None
    gas = atmosphere(equation) if equation else {"atmosphere": None}
    steps = firing_steps(operations)

    protocol = {
        "target": target,
        "batch_size_g": batch_g,
        "steps": [],
        "provenance": {
            "precursors": "predicted (DAO-S ConditionalLookup: metal, "
                          "anion class, oxidation state)",
            "firing_temperature_C": "predicted (DAO-S TemperatureNet; "
                                    "194 C MAE on unseen targets)",
            "masses_g": "derived (stoichiometry x molar mass)",
            "atmosphere": "derived (which side of the equation O2 is on)",
            "ramp_rate_C_per_min": "convention, not recorded in any corpus",
            "grind_minutes": "convention, not recorded in any corpus",
            "purity": "convention, not recorded in any corpus",
            "dwell_hours": ("predicted (DAO-S dwell network; within 2x for "
                            "48.6% of held-out recipes, median 2.06x off)"
                            if dwell_hours else "not predicted"),
            "feasibility": "derived (reaction.assess)",
            "kinetics": "evidence (A-Lab robot experiments, Precursor "
                        "Genome 2026), pair by pair",
        },
    }

    if masses:
        protocol["steps"].append({
            "step": 1, "operation": "weigh",
            "detail": masses,
            "note": "for {:g} g of {}".format(batch_g, target),
            "purity": PURITY})
    else:
        protocol["steps"].append({
            "step": 1, "operation": "weigh",
            "detail": sorted(precursors),
            "note": "masses unavailable -- the equation could not be balanced"})

    protocol["steps"].append({
        "step": 2, "operation": "grind and mix",
        "detail": "agate mortar, about {} minutes, a few drops of acetone as "
                  "a grinding aid".format(GRIND_MINUTES),
        "note": "a solid-state reaction only happens where two particles "
                "touch, and the diffusion time goes as the square of the "
                "particle size -- halving it quarters the time needed"})

    protocol["steps"].append({
        "step": 3, "operation": "fire",
        "temperature_C": round(firing_temperature_C),
        "dwell_hours": dwell_hours,
        "ramp_rate_C_per_min": RAMP_RATE_C_PER_MIN,
        "atmosphere": gas.get("atmosphere"),
        "atmosphere_reason": gas.get("reason"),
        "cycles": steps["firing_cycles"],
        "note": "ramp slowly: gas leaving too fast lifts the powder out of "
                "the crucible, and thermal shock cracks it"})

    if steps["firing_cycles"] > 1:
        protocol["steps"].append({
            "step": 4, "operation": "regrind and refire",
            "detail": "{} further cycle(s) at the same temperature".format(
                steps["firing_cycles"] - 1),
            "note": steps["reason"]})

    protocol["equation"] = equation
    protocol["temperature_checks"] = temperature_checks(firing_temperature_C,
                                                        assessment)
    try:
        from . import volatility
        lost = volatility.check(precursors, firing_temperature_C)
    except Exception:                                           # noqa: BLE001
        lost = []
    protocol["volatility"] = lost
    protocol["temperature_checks"] += ["VOLATILITY: " + w["advice"]
                                       for w in lost]
    protocol["provenance"]["volatility"] = (
        "convention (which elements) + Precursor Genome melting points "
        "(where)")
    try:
        from . import kinetics
        protocol["kinetics"] = kinetics.check_recipe(
            precursors, firing_temperature_C, dwell_hours)
    except Exception as error:                                  # noqa: BLE001
        protocol["kinetics"] = {"error": "{}: {}".format(
            type(error).__name__, str(error)[:100])}
    if assessment:
        protocol["feasibility"] = {
            "feasible": assessment.get("feasible"),
            "enthalpy_eV": assessment.get("enthalpy_eV"),
            "entropy_term_eV": assessment.get("entropy_term_eV"),
            "free_energy_eV": assessment.get("free_energy_eV"),
            "onset_temperature_C": assessment.get("onset_temperature_C"),
            "ceiling_temperature_C": assessment.get("ceiling_temperature_C"),
            "gases": assessment.get("gases"),
            "in_air": assessment.get("in_air"),
            "verdict_class": assessment.get("verdict_class"),
            "error_band_meV_per_atom":
                assessment.get("error_band_meV_per_atom"),
            "polymorph": assessment.get("polymorph"),
            "driving_force_meV_per_atom":
                assessment.get("driving_force_meV_per_atom"),
            "kinetic_risk": assessment.get("kinetic_risk"),
        }
    return protocol


# A standard box furnace with resistance elements runs to about 1200 C; above
# that needs MoSi2 elements or a tube furnace. The A-Lab never went above
# 1100 C, and DAO-S proposed up to 1445 C for its targets.
BOX_FURNACE_MAX_C = 1200.0


def temperature_checks(firing_temperature_C: float,
                       assessment: Optional[dict]) -> List[str]:
    """The predicted temperature held against what the simulator found.

    DAO-S and the simulator answered separately: the network proposes a
    temperature, the thermodynamics finds a floor or a ceiling, and nothing
    compared the two before the protocol was printed. A recipe fired below
    its floor, or above its ceiling, cannot work however long it is held.
    """
    notes = []
    T = firing_temperature_C
    a = assessment or {}
    floor = a.get("onset_temperature_C")
    air = a.get("in_air") or {}
    floor_air = air.get("onset_temperature_C")
    if floor not in (None, float("inf")) and T < floor:
        if floor_air not in (None, float("inf")) and T >= floor_air:
            notes.append("{:.0f} C is below the 1 bar floor ({:.0f} C) but "
                         "above the floor in air ({:.0f} C): it works only if "
                         "the gas can leave -- use an open crucible and "
                         "flowing air".format(T, floor, floor_air))
        else:
            notes.append("{:.0f} C is BELOW the thermodynamic floor ({:.0f} C "
                         "at 1 bar{}): fire at {:.0f} C or above".format(
                             T, floor,
                             "" if floor_air in (None, float("inf"))
                             else ", {:.0f} C in air".format(floor_air),
                             (floor if floor_air in (None, float("inf"))
                              else floor_air) + 50))
    ceiling = a.get("ceiling_temperature_C")
    if ceiling is not None and T > ceiling:
        notes.append("{:.0f} C is ABOVE the ceiling ({:.0f} C at 1 bar), where "
                     "the target gives back the oxygen it took up: fire "
                     "below it, or under flowing O2".format(T, ceiling))
    if T > BOX_FURNACE_MAX_C:
        notes.append("{:.0f} C is above a standard box furnace (~{:.0f} C): "
                     "needs MoSi2 elements or a tube furnace".format(
                         T, BOX_FURNACE_MAX_C))
    return notes


def render(protocol: dict) -> str:
    """The protocol as something to print and hand over."""
    lines = ["TARGET   {}   (batch {:g} g)".format(
        protocol["target"], protocol["batch_size_g"])]
    if protocol.get("equation"):
        lines.append("REACTION {}".format(protocol["equation"]))
    lines.append("")

    for step in protocol["steps"]:
        head = "{}. {}".format(step["step"], step["operation"].upper())
        lines.append(head)
        detail = step.get("detail")
        if isinstance(detail, list) and detail and isinstance(detail[0], dict):
            for row in detail:
                lines.append("     {:14s} {:9.4f} g   ({:g} x {:.2f} g/mol)"
                             .format(row["precursor"], row["grams"],
                                     row["coefficient"],
                                     row["molar_mass_g_per_mol"]))
        elif detail is not None:
            lines.append("     {}".format(detail))
        if step.get("purity"):
            lines.append("     purity: {}".format(step["purity"]))
        if step.get("temperature_C") is not None:
            lines.append("     {:.0f} C, ramp {:g} C/min, {} cycle(s)".format(
                step["temperature_C"], step["ramp_rate_C_per_min"],
                step["cycles"]))
            if step.get("dwell_hours"):
                lines.append("     hold ~{:g} h per cycle (predicted; "
                             "typically within a factor of 2)".format(
                                 step["dwell_hours"]))
            else:
                lines.append("     hold time: not predicted for this "
                             "target")
            lines.append("     atmosphere: {}".format(step["atmosphere"]))
            lines.append("       because {}".format(
                step["atmosphere_reason"]))
            for note in protocol.get("temperature_checks") or []:
                lines.append("     CHECK: {}".format(note))
        if step.get("note"):
            lines.append("     ({})".format(step["note"]))
        lines.append("")

    feasibility = protocol.get("feasibility")
    if feasibility:
        verdict = feasibility.get("feasible")
        if feasibility.get("verdict_class") == "uncertain":
            headline = ("UNCERTAIN -- dG is inside the +/-{:.0f} meV/atom the "
                        "energies can resolve".format(
                            feasibility["error_band_meV_per_atom"]))
        else:
            headline = ("thermodynamics does not block this" if verdict
                        else "REJECTED" if verdict is False
                        else "could not be checked")
        lines.append("FEASIBILITY  {}".format(headline))
        if feasibility.get("free_energy_eV") is not None:
            lines.append("     dH {:+.3f} eV   gas {:+.3f} eV   dG {:+.3f} eV"
                         "   (gases at 1 bar)"
                         .format(feasibility["enthalpy_eV"],
                                 feasibility["entropy_term_eV"],
                                 feasibility["free_energy_eV"]))
            air = feasibility.get("in_air")
            if air:
                lines.append("                     gas {:+.3f} eV   dG {:+.3f} eV"
                             "   (in air: CO2 4.2e-4 bar, O2 0.21 bar)"
                             .format(air["entropy_term_eV"],
                                     air["free_energy_eV"]))
        onset = feasibility.get("onset_temperature_C")
        ceiling = feasibility.get("ceiling_temperature_C")
        # A floor below room temperature is printed as what it means, not as
        # a number. LiCoO2 came out at "-220 C" two lines above a verdict
        # saying "downhill at any practical temperature" -- the same fact,
        # stated so one line looked like an error. reaction.verdict already
        # had this rule; the protocol did not.
        if ceiling is not None:
            lines.append("     thermodynamic ceiling about {:.0f} C -- the "
                         "reaction takes up oxygen, and above this the target "
                         "gives it back".format(ceiling))
        elif onset is None and verdict:
            lines.append("     downhill even without the gas term")
        elif onset is not None and onset != float("inf"):
            if onset < 25:
                lines.append("     downhill at any practical temperature -- "
                             "only kinetics decides")
            else:
                lines.append("     thermodynamic floor about {:.0f} C".format(
                    onset))
        force = feasibility.get("driving_force_meV_per_atom")
        if force is not None:
            if feasibility.get("kinetic_risk"):
                lines.append("     driving force {:.0f} meV/atom -- KINETIC "
                             "RISK, below the 50 meV/atom where the A-Lab saw "
                             "reactions stall".format(force))
            elif verdict:
                lines.append("     driving force {:.0f} meV/atom -- above the "
                             "50 meV/atom kinetic threshold".format(force))
        band = feasibility.get("error_band_meV_per_atom")
        if band is not None:
            lines.append("     error band +/-{:.0f} meV/atom -- what these "
                         "energies can resolve".format(band))
        polymorph = feasibility.get("polymorph")
        if polymorph:
            gap = 1000.0 * polymorph["gap_eV_per_atom"]
            lines.append("     polymorph: {:+.0f} meV/atom against the known "
                         "phase {}{}".format(
                             gap, polymorph.get("known_material_id", ""),
                             " -- a synthesis may make the known phase "
                             "instead" if gap > 50 else ""))
    lines.extend(_render_kinetics(protocol.get("kinetics")))
    return "\n".join(lines)


def _render_kinetics(kinetic) -> List[str]:
    """What the robots already saw for each pair of this recipe."""
    if not kinetic or not kinetic.get("pairs"):
        return []
    lines = ["", "KINETICS  what A-Lab robots observed for these precursor "
                 "pairs (Precursor Genome)"]
    for pair in kinetic["pairs"]:
        a, b = pair["pair"]
        if pair["evidence"] == "blocked":
            seen_T, seen_t, _ = pair["seen"]
            lines.append("     {} + {}: NO reaction at {:.0f} C after {:.0f} h "
                         "-- will not react at {:.0f} C; expect this pair to "
                         "be the slow step".format(
                             a, b, seen_T, seen_t / 60.0,
                             kinetic["temperature_C"]))
        elif pair["evidence"] == "observed":
            seen_T, seen_t, _ = pair["seen"]
            lines.append("     {} + {}: reacted by {:.0f} C in {:.0f} h -- "
                         "fast enough here".format(a, b, seen_T,
                                                   seen_t / 60.0))
        else:
            ratio = pair.get("tammann_ratio")
            lines.append("     {} + {}: not observed under these conditions"
                         "{}".format(a, b, "" if ratio is None else
                                     " (T/Tm of the lower-melting one {:.2f};"
                                     " Tammann's rule expects reaction above "
                                     "0.67)".format(ratio)))
    return lines
