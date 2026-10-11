"""The larger synthesis corpus: 80,806 reactions, LLM-extracted.

Source: "Text-mined dataset of solid-state syntheses with impurity phases
using Large Language Model", figshare 10.6084/m9.figshare.30423274, CC BY 4.0.
Extraction was gpt-4o-2024-05-13 at temperature 0, few-shot.

WHY A SECOND CORPUS AND NOT A REPLACEMENT
-----------------------------------------
Measured against what synflow/recipes.py loads:

    Kononova et al. 2019       19,488 raw   ->  8,836 with a parseable target
    this one                   80,806 raw   -> 55,371 with a target, precursors
                                               and a firing temperature

6.3x more, and three things the older corpus does not record at all:

    mp_id            the target already resolved to a Materials Project entry,
                     for 28,937 of the usable records. Validating the
                     simulator on the old corpus left 48% of real recipes
                     UNCHECKABLE because the target formula could not be
                     matched to a database entry. This removes that problem
                     rather than working around it.

    impurity_phase   what else came out of the furnace, for 18,869 records.
                     That is the selectivity question -- "will the reaction
                     give THAT product" -- as labelled data. Nothing in this
                     project could previously be trained or tested on it.

    target_reaction  the balanced equation, including the oxygen coefficient.
                     reaction.py derives that itself, so these are ground
                     truth to check our balancer against.

It is kept alongside rather than replacing: the extraction is reported at
precision 0.90 / recall 0.69 for targets, against a hand-engineered pipeline
for the 2019 set. Precursors are 0.95/0.97 and impurity phases 0.86/0.90, so
the parts this project leans on hardest are the parts extracted best -- but a
recall of 0.69 on targets is real noise and belongs in any write-up.

TWO THINGS THE RECORDS GET RIGHT THAT WE HAD TO FIGHT FOR
---------------------------------------------------------
Free variables are in their own fields. `amounts_vars` and `elements_vars`
hold the unresolved doping variables, so a template is identifiable from
metadata instead of by parsing "(Ba1-xCax)(Ti0.85Sn0.15)O3" out of a string --
which is exactly where the old loader silently produced Ba1Ca1 and put a
composition that cannot exist into 14.6% of the test split.

78,994 of 80,806 records are concrete by that test, against 8,836 of 19,488
for the older corpus. Most of the difference is that this is a filter on
recorded metadata rather than a guess.

AND ONE THING THEY GET WRONG
----------------------------
Some temperature values are concatenated rather than parsed: a record whose
paper lists 1200, 1300, 1400 and 1500 C comes back with max = 1200130014001500.
So every temperature is bounds-checked. Without it one recipe asks for twelve
million degrees and the regression's loss is decided by that single row.
"""

import gzip
import hashlib
import json
import os
from typing import List, Optional

from .recipes import Recipe

DATASET = "data/synthesis/SS_rxns_80806.json.gz"
EXPECTED_MD5 = "2cca759682be4689e7d0d3d882d12909"
_FIGSHARE_API = "https://api.figshare.com/v2/articles/30423274/files"

# A solid-state firing is between room temperature and the hottest laboratory
# furnaces. Outside this the value is a parsing artefact, not a condition.
MIN_TEMPERATURE_C = 50.0
MAX_TEMPERATURE_C = 2500.0


class LLMRecipe(Recipe):
    """A Recipe plus what this corpus records and the older one does not."""

    __slots__ = ("mp_id", "e_above_hull", "impurity_phases",
                 "balanced_reaction", "atmosphere", "dwell_hours")

    def __init__(self, target, precursors, operations, temperatures, doi="",
                 mp_id=None, e_above_hull=None, impurity_phases=(),
                 balanced_reaction=None, atmosphere=(), dwell_hours=None):
        super().__init__(target, precursors, operations, temperatures, doi)
        self.mp_id = mp_id
        self.e_above_hull = e_above_hull
        self.impurity_phases = tuple(impurity_phases)
        self.balanced_reaction = balanced_reaction
        self.atmosphere = tuple(atmosphere)
        self.dwell_hours = dwell_hours


def download_dataset(path: str = DATASET,
                     expected_md5: Optional[str] = EXPECTED_MD5) -> str:
    """Fetch the corpus, resolving the file id through the figshare API.

    The id is asked for rather than hardcoded, for the same reason as in
    recipes.py: a new version of the dataset gets a new id, and a pinned URL
    would quietly fetch the old one or 404 long after anyone remembers why.
    """
    import urllib.request

    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with urllib.request.urlopen(_FIGSHARE_API, timeout=120) as response:
        files = json.load(response)
    match = next((f for f in files if f["name"].endswith(".json.gz")), None)
    if match is None:
        raise RuntimeError("no .json.gz in figshare article 30423274")

    urllib.request.urlretrieve(match["download_url"], path)

    if expected_md5:
        with open(path, "rb") as handle:
            digest = hashlib.md5(handle.read()).hexdigest()
        if digest != expected_md5:
            os.remove(path)
            raise RuntimeError(
                "md5 {} does not match the expected {} -- the file was "
                "removed rather than used".format(digest, expected_md5))
    return path


def _max_temperature(record) -> Optional[float]:
    """The hottest Celsius value in the record, or None.

    Bounds-checked: see the note on concatenated values in the module
    docstring. A silently absurd temperature is worse than a missing one,
    because a missing one is excluded from the regression and an absurd one
    dominates it.
    """
    best = None
    for condition in record.get("conditions_forDOI") or []:
        for value in condition.get("temp_values") or []:
            if (value.get("units") or "").strip() != "°C":
                continue
            reading = value.get("max")
            if not isinstance(reading, (int, float)):
                continue            # a concatenated string lands here
            if not MIN_TEMPERATURE_C <= reading <= MAX_TEMPERATURE_C:
                continue
            best = reading if best is None else max(best, reading)
    return best


# Hours per unit, for every spelling the corpus uses. Measured on the raw
# records: h 97,559 / min 16,316 / day 4,781 / hours 3,493 / d 955 /
# minutes 592 / hour 387 / hrs 320 / hr 236.
_HOURS_PER = {"h": 1.0, "hour": 1.0, "hours": 1.0, "hr": 1.0, "hrs": 1.0,
              "min": 1 / 60.0, "mins": 1 / 60.0, "minute": 1 / 60.0,
              "minutes": 1 / 60.0, "d": 24.0, "day": 24.0, "days": 24.0}

# A firing step shorter than a few minutes or longer than three weeks is a
# parsing artefact, the same way a twelve-million-degree temperature was.
MIN_DWELL_H = 0.05
MAX_DWELL_H = 500.0


def _dwell_hours(record) -> Optional[float]:
    """How long the HOTTEST step was held, in hours, or None.

    The time is taken from the same operation as the firing temperature, so
    the two describe one step. A recipe usually has several heating steps --
    drying, calcining, sintering -- and averaging or summing their times
    would describe none of them. Per cycle, not in total: the protocol states
    the number of cycles separately.
    """
    # FIRST THE HOTTEST TEMPERATURE, THEN ONLY ITS OWN TIMES
    #
    # The first version took the hottest step AMONG THOSE THAT RECORDED A
    # TIME. Where the firing step's time was never written down, it reached
    # past it to a cooler one -- a 12 h drying step at 120 C became the dwell
    # of a 1100 C firing, and the protocol paired a temperature from one step
    # with a time from another. It also kept only the first of two steps tied
    # at the top temperature. Both were caught by tests/test_dwell.py.
    steps = []
    for condition in record.get("conditions_forDOI") or []:
        temps = [v.get("max") for v in condition.get("temp_values") or []
                 if (v.get("units") or "").strip() == "°C"
                 and isinstance(v.get("max"), (int, float))
                 and MIN_TEMPERATURE_C <= v["max"] <= MAX_TEMPERATURE_C]
        if not temps:
            continue
        hours = []
        for value in condition.get("time_values") or []:
            per = _HOURS_PER.get((value.get("units") or "").strip().lower())
            amount = value.get("max")
            if per is None or not isinstance(amount, (int, float)):
                continue
            converted = amount * per
            if MIN_DWELL_H <= converted <= MAX_DWELL_H:
                hours.append(converted)
        steps.append((max(temps), hours))

    if not steps:
        return None
    hottest = max(temperature for temperature, _ in steps)
    times = [h for temperature, hours in steps if temperature == hottest
             for h in hours]
    return max(times) if times else None


def _atmosphere(record) -> tuple:
    """Whatever gas environment the operations mention, deduplicated.

    Recorded because the simulator decides which side of the equation oxygen
    belongs on, and that is only meaningful against the atmosphere actually
    used -- LiFePO4 is fired under argon for exactly this reason.
    """
    seen = []
    for condition in record.get("conditions_forDOI") or []:
        for token in condition.get("env_toks") or []:
            token = (token or "").strip()
            if token and token not in seen:
                seen.append(token)
    return tuple(seen)


def load_recipes(path: str = DATASET, concrete_only: bool = True,
                 require_temperature: bool = True,
                 verbose: bool = True) -> List[LLMRecipe]:
    """Parse the corpus into Recipe objects the rest of the project accepts.

    `concrete_only` drops records whose target carries an unresolved doping
    variable, using the dataset's own `amounts_vars` / `elements_vars` fields
    rather than inspecting the formula string.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "{} is missing -- call recipes_llm.download_dataset()".format(path))

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        records = json.load(handle)

    out: List[LLMRecipe] = []
    dropped = {"no target": 0, "templated": 0, "no precursors": 0,
               "no temperature": 0}

    for record in records:
        targets = record.get("target") or []
        if len(targets) != 1:
            dropped["no target"] += 1
            continue
        target = targets[0]
        formula = target.get("material_formula") or ""
        if not formula:
            dropped["no target"] += 1
            continue

        if concrete_only and (target.get("amounts_vars")
                              or target.get("elements_vars")):
            dropped["templated"] += 1
            continue

        precursors = [p.get("material_formula") for p in
                      record.get("precursors") or []
                      if p.get("material_formula")]
        if not precursors:
            dropped["no precursors"] += 1
            continue

        temperature = _max_temperature(record)
        if require_temperature and temperature is None:
            dropped["no temperature"] += 1
            continue

        operations = tuple(
            c.get("act_type") for c in record.get("conditions_forDOI") or []
            if c.get("act_type"))

        out.append(LLMRecipe(
            target=formula,
            precursors=precursors,
            operations=operations,
            temperatures=() if temperature is None else (temperature,),
            doi=record.get("DOI") or "",
            mp_id=target.get("mp_id"),
            e_above_hull=target.get("mp_eabovehull"),
            impurity_phases=tuple(
                p.get("material_formula") for p in
                record.get("impurity_phase") or []
                if p.get("material_formula")),
            balanced_reaction=(record.get("target_reaction") or [[None]*4])[0][-1]
            if record.get("target_reaction") else None,
            atmosphere=_atmosphere(record),
            dwell_hours=_dwell_hours(record),
        ))

    if verbose:
        print("{}: {:,} of {:,} records usable".format(
            os.path.basename(path), len(out), len(records)))
        for reason, count in dropped.items():
            if count:
                print("   dropped, {:16s} {:,}".format(reason, count))
        with_mp = sum(1 for r in out if r.mp_id)
        with_imp = sum(1 for r in out if r.impurity_phases)
        print("   with an mp_id    {:,}".format(with_mp))
        print("   with an impurity {:,}".format(with_imp))

    return out


def describe(recipes) -> dict:
    """The numbers worth quoting about a loaded corpus."""
    temperatures = sorted(r.max_temperature for r in recipes
                          if r.max_temperature is not None)
    targets = {r.target for r in recipes}
    return {
        "recipes": len(recipes),
        "targets": len(targets),
        "with_mp_id": sum(1 for r in recipes if r.mp_id),
        "with_impurity": sum(1 for r in recipes if r.impurity_phases),
        "with_atmosphere": sum(1 for r in recipes if r.atmosphere),
        "median_temperature_C": (temperatures[len(temperatures) // 2]
                                 if temperatures else None),
        "mean_precursors": (sum(len(r.precursors) for r in recipes)
                            / max(len(recipes), 1)),
    }
