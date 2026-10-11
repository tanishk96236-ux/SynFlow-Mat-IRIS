"""Will a downhill reaction actually happen in the time given? A kinetic gate.

WHY THE SIMULATOR NEEDS ONE
---------------------------
reaction.py answers "CAN this reaction go" -- dG < 0. It cannot answer "WILL
it, in four hours at 700 C". On the A-Lab's Supplementary Table 2 that was
the whole of its weakness: 68/71 reactions the robot made were found
downhill, but of 10 pairs the robot heated and saw NOTHING form, the gate
predicted a product for 5. Y2O3 + Ga2O3 at 700 C is downhill to the garnet;
the robot saw no reaction. Thermodynamics was not wrong. It was not the
question.

WHAT DECIDES IT, AND WHAT IS LEARNED HERE
----------------------------------------
A solid-state reaction needs atoms to move, and the diffusion coefficient
climbs steeply as a solid approaches its melting point -- D = D0 exp(-Q/RT),
with Q roughly proportional to the melting point. That is the physics behind
Tammann's rule: reactions between solids start at about 2/3 of the melting
point (in kelvin) of the more mobile reactant.

Tammann's rule is one number. This module learns a little more from
experiment: the Precursor Genome (Walters, McDermott, Rendy, Fei, Persson,
Ceder, 2026; CC BY 4.0), 1,035 pairwise reactions run by the same A-Lab
robots, each XRD-classified as unreacted, transformed, partially or
completely reacted. Every pair was heated at its Tammann temperature, and
only about a third formed a mixed phase in an hour -- so the rule is a floor,
not a prediction, and the rest has to be learned.

THE FEATURES, AND WHY EACH ONE
------------------------------
    h_min    T / Tm of the lower-melting precursor (K/K)   the mobile one
    h_max    T / Tm of the higher-melting precursor        the one that must
                                                           take it in
    log_t    log10 of the hold, minutes                    diffusion length
                                                           grows as sqrt(t)
    dchi     electronegativity gap between the cations     acid-base pairs
                                                           (Ag2O + GeO2)
                                                           react fast
    decomp   precursors that decompose (carbonate,         a decomposing
             hydroxide, ammonium, oxalate, peroxide)       precursor leaves
                                                           fresh, fine oxide

A logistic model on five numbers, so every coefficient can be read and
defended. The thermodynamic driving force is deliberately NOT a feature:
reaction.py already decides whether a reaction is downhill, and this model
is trained only on pairs that were (by more than 5 meV/atom), to answer the
one remaining question.

WHAT WAS MEASURED, AND WHAT IS THEREFORE USED (2026-10-08)
----------------------------------------------------------
On A-Lab Table 2, every test pair removed from training (train_kinetics.py):

    gate                          formed          inert      balanced
    thermodynamics only           68/71  95.8%    5/10  50%    72.9%
    + the learned model           53/71  74.6%    5/10  50%    62.3%
    + Tammann (2/3 Tm, kelvin)    66/71  93.0%    8/10  80%    86.5%
    + the robots' own evidence    68/71  95.8%    7/10  70%    82.9%

The learned model scores AUC 0.73 inside the Precursor Genome and fails on
Table 2. The reason is the dataset's design: every pair was heated at its own
Tammann temperature, so temperature and precursor identity are tied
together -- ammonium phosphates were only ever seen at 200 C -- and the model
learned that tie, not kinetics. It is kept, and not used.

Tammann's rule helps on Table 2 but has no power inside the Precursor
Genome (reacted 35% below 2/3, 28% above), so three fixed rows out of ten
are not enough to trust it as a veto. It is reported as information.

What is used is evidence(): what the robots actually saw for the same pair,
read through the one thing certain about kinetics -- more heat and more time
never undo a reaction. It fixed two inert rows and broke no formed one.
Ten inert rows are a small test; that is said wherever the number is.

MELTING POINTS
--------------
From the Precursor Genome's own precursor table (46 compounds, with CAS
numbers and supplier product numbers). For carbonates, hydroxides and the
like that value is where the compound decomposes, which is the temperature
at which it stops being itself -- the same convention the dataset used. A
precursor outside that table has no melting point here, and the gate says
"unknown" rather than guessing one.
"""

import functools
import json
import math
import os
from typing import Dict, Optional

LEDGER = "data/precursor_genome/ledger_precursor_genome.json"
MODEL = "runs/kinetics_model.json"

FEATURES = ("h_min", "h_max", "log_t", "dchi", "decomp")

# Melting points not in the Precursor Genome table, each with its source.
# Kept to values that are reference standards, not looked-up guesses.
EXTRA_MELTING_C = {
    "Ag": 961.78,      # ITS-90 defining fixed point (freezing point of silver)
}

_LIGHT = frozenset({"O", "C", "H", "N"})


@functools.lru_cache(maxsize=1)
def melting_points(path: str = LEDGER) -> Dict[str, float]:
    """reduced formula -> melting (or decomposition) point, C."""
    from pymatgen.core import Composition

    table: Dict[str, float] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            ledger = json.load(handle)
        for sample in ledger["samples"]:
            for precursor in sample.get("precursors") or []:
                value = precursor.get("melting_point_c")
                if value is not None:
                    table[Composition(precursor["formula"])
                          .reduced_formula] = float(value)
    for formula, value in EXTRA_MELTING_C.items():
        table.setdefault(Composition(formula).reduced_formula, value)
    return table


def melting_point(formula: str) -> Optional[float]:
    from pymatgen.core import Composition
    try:
        return melting_points().get(Composition(formula).reduced_formula)
    except Exception:                                           # noqa: BLE001
        return None


def _cation_electronegativity(formula: str) -> Optional[float]:
    """Mean Pauling electronegativity of the non-light elements, by atom."""
    from pymatgen.core import Composition
    try:
        composition = Composition(formula)
    except Exception:                                           # noqa: BLE001
        return None
    total = weight = 0.0
    for element, amount in composition.items():
        if str(element) in _LIGHT:
            continue
        chi = element.X
        if chi is None or math.isnan(chi):
            return None
        total += amount * chi
        weight += amount
    return total / weight if weight else None


def decomposes(formula: str) -> bool:
    """Carbonate, hydroxide, ammonium salt, oxalate or peroxide."""
    from pymatgen.core import Composition
    try:
        composition = Composition(formula)
    except Exception:                                           # noqa: BLE001
        return False
    elements = {str(e) for e in composition}
    if elements & {"C", "H"}:
        return True
    # a peroxide holds more oxygen than its cations' usual states allow
    return composition.reduced_formula in ("BaO2", "SrO2", "CaO2", "MgO2",
                                           "Li2O2", "Na2O2", "ZnO2")


def features(precursor_a: str, precursor_b: str, temperature_C: float,
             dwell_minutes: float) -> Optional[Dict[str, float]]:
    """The five numbers, or None if a melting point is unknown."""
    tm = [melting_point(precursor_a), melting_point(precursor_b)]
    chi = [_cation_electronegativity(precursor_a),
           _cation_electronegativity(precursor_b)]
    if None in tm or None in chi or not dwell_minutes:
        return None
    low, high = sorted(tm)
    kelvin = temperature_C + 273.15
    return {
        "h_min": kelvin / (low + 273.15),
        "h_max": kelvin / (high + 273.15),
        "log_t": math.log10(dwell_minutes),
        "dchi": abs(chi[0] - chi[1]),
        "decomp": float(decomposes(precursor_a) + decomposes(precursor_b)),
    }


# ---------------------------------------------------------------------------
# what the robots already saw: the Precursor Genome as evidence, pair by pair
# ---------------------------------------------------------------------------
#
# A reaction that did not happen at 1100 C in an hour will not happen at
# 900 C in four: the extent of a solid-state reaction only grows with
# temperature and with time. That monotonicity is the whole rule, and it
# needs no fitting -- only an exchange rate between time and temperature.
#
# THE EXCHANGE RATE. Equal diffusion length needs D(T1) t1 = D(T2) t2, so
# with D = D0 exp(-Q/RT) a hold four times shorter is matched by
#
#     1/T1 - 1/T2 = R ln 4 / Q
#
# For cation diffusion in oxides Q is roughly 200-500 kJ/mol; at 300 kJ/mol
# and 1000-1300 K that is 40-65 K. So an observation at one hour is taken to
# speak for four hours only 50 K lower. Stated before it was applied.
TIME_MARGIN_K = 50.0


REACTED = ("partially_reacted", "completely_reacted")


@functools.lru_cache(maxsize=1)
def _observations(path: str = LEDGER) -> Dict[frozenset, list]:
    """pair -> [(temperature C, hold minutes, outcome category)]."""
    from pymatgen.core import Composition

    out: Dict[frozenset, list] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as handle:
        ledger = json.load(handle)
    for sample in ledger["samples"]:
        category = (sample.get("outcome") or {}).get("reaction_category")
        synthesis = sample.get("synthesis") or {}
        if category not in ("unreacted", "transformed", "partially_reacted",
                            "completely_reacted"):
            continue
        if synthesis.get("heating_temperature") is None \
                or synthesis.get("heating_time") is None:
            continue
        pair = frozenset(Composition(p["formula"]).reduced_formula
                         for p in sample["precursors"])
        out.setdefault(pair, []).append((
            float(synthesis["heating_temperature"]),
            float(synthesis["heating_time"]), category))
    return out


def evidence(precursor_a: str, precursor_b: str, temperature_C: float,
             dwell_minutes: float = 240.0) -> Optional[dict]:
    """What the A-Lab already observed for this pair, if it settles the case.

    Returns {"verdict": "blocked" | "observed", "seen": (T, minutes,
    category)} or None when no observation bears on these conditions.

      blocked    UNREACTED -- every phase still a precursor -- at a
                 temperature at least as high (50 K higher if that hold was
                 shorter): it will not react here
      observed   a mixed phase formed at a temperature no higher, in a hold
                 no longer: it will react here

    WHY "TRANSFORMED" IS NOT EVIDENCE OF NO REACTION
    The dataset's "transformed" means new phases appeared that its automated
    refinement did not attribute to both precursors. For MgO + NiO at
    1100 C it reported MgO entirely consumed and METALLIC Mg formed -- in air,
    which cannot be right; what forms is the (Mg,Ni)O rocksalt solid
    solution, whose pattern sits between the two. The first version of this
    rule counted "transformed" as no reaction and so blocked MgO + NiO,
    which the A-Lab did make at 900 C. Only "unreacted" says nothing
    happened. (Changed after that case was seen; both versions are
    reported by scripts/train_kinetics.py.)
    """
    from pymatgen.core import Composition
    try:
        pair = frozenset(Composition(f).reduced_formula
                         for f in (precursor_a, precursor_b))
    except Exception:                                           # noqa: BLE001
        return None
    for seen_T, seen_t, category in _observations().get(pair, []):
        margin = 0.0 if seen_t >= dwell_minutes else TIME_MARGIN_K
        if category == "unreacted" and seen_T >= temperature_C + margin:
            return {"verdict": "blocked", "seen": (seen_T, seen_t, category)}
        if category in REACTED and seen_T <= temperature_C \
                and seen_t <= dwell_minutes:
            return {"verdict": "observed",
                    "seen": (seen_T, seen_t, category)}
    return None


def check_recipe(precursors, temperature_C: float,
                 dwell_hours: Optional[float] = None) -> dict:
    """Every precursor pair of a recipe against what the robots saw.

    A recipe of four precursors is six pairs, and a solid-state reaction
    proceeds through them two at a time. A pair the robots saw stay
    unreacted at least this hot is a pair that will not react here, and a
    route that needs it is slower than its free energy suggests.

    The Tammann ratio -- T over the lower melting point, in kelvin -- is
    reported for each pair as information, not as a verdict: on the
    Precursor Genome's own pairs it did not separate reacted from unreacted
    (reacted 35% below 2/3, 28% above).
    """
    from itertools import combinations

    minutes = 60.0 * dwell_hours if dwell_hours else 240.0
    pairs = []
    for a, b in combinations(sorted(set(precursors)), 2):
        found = evidence(a, b, temperature_C, minutes)
        x = features(a, b, temperature_C, minutes)
        pairs.append({
            "pair": (a, b),
            "evidence": found["verdict"] if found else None,
            "seen": found["seen"] if found else None,
            "tammann_ratio": round(x["h_min"], 3) if x else None,
        })
    blocked = [p["pair"] for p in pairs if p["evidence"] == "blocked"]
    return {"temperature_C": temperature_C, "dwell_minutes": minutes,
            "pairs": pairs, "blocked_pairs": blocked,
            "source": "Precursor Genome (Walters et al. 2026, CC BY 4.0)"}


@functools.lru_cache(maxsize=1)
def _model(path: str = MODEL) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def probability(precursor_a: str, precursor_b: str, temperature_C: float,
                dwell_minutes: float = 240.0,
                model: Optional[dict] = None) -> Optional[float]:
    """P(a mixed phase forms within the hold), or None when unknown."""
    model = model or _model()
    x = features(precursor_a, precursor_b, temperature_C, dwell_minutes)
    if model is None or x is None:
        return None
    z = model["intercept"]
    for name in FEATURES:
        z += model["coef"][name] * (x[name] - model["mean"][name]) \
            / model["scale"][name]
    return 1.0 / (1.0 + math.exp(-z))


def will_react(precursor_a: str, precursor_b: str, temperature_C: float,
               dwell_minutes: float = 240.0) -> Optional[bool]:
    """The gate: probability against the threshold chosen on training data."""
    model = _model()
    p = probability(precursor_a, precursor_b, temperature_C, dwell_minutes,
                    model)
    if p is None:
        return None
    return p >= model["threshold"]
