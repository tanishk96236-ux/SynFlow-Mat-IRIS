"""Dwell time: how long the firing step is held.

Read from the corpus's time_values and paired with the firing temperature, so
the two must describe the SAME step -- a recipe usually has several heating
steps (drying, calcining, sintering) and a time from one beside a temperature
from another describes nothing that happened.
"""

import math

from synflow.recipes_llm import _dwell_hours


def _record(*steps):
    """A minimal corpus record: (temperature C, time, unit) per heating step."""
    return {"conditions_forDOI": [
        {"act_type": "Heating",
         "temp_values": [] if t is None else
         [{"max": t, "units": "°C"}],
         "time_values": [] if h is None else [{"max": h, "units": u}]}
        for t, h, u in steps]}


def test_the_time_comes_from_the_hottest_step():
    record = _record((120, 12, "h"), (1100, 6, "h"))
    assert _dwell_hours(record) == 6


def test_a_hottest_step_without_a_time_gives_none_not_a_cooler_steps_time():
    """The case that was wrong. The parser took the hottest step AMONG THOSE
    THAT RECORD A TIME, so a 12 h drying step at 120 C was reported as the
    dwell of a 1100 C firing whose own time was never written down. The
    firing temperature and the dwell then described different steps."""
    record = _record((120, 12, "h"), (1100, None, None))
    assert _dwell_hours(record) is None


def test_two_steps_at_the_firing_temperature_share_it():
    """Two cycles at the same top temperature are the same kind of step, so
    either one's time is a fair answer; the longer is kept."""
    record = _record((1100, 6, "h"), (1100, 12, "h"))
    assert _dwell_hours(record) == 12


def test_every_spelling_of_a_unit_becomes_hours():
    """Measured spellings in the corpus: h, hours, hour, hrs, hr, min,
    minutes, day, d."""
    assert _dwell_hours(_record((900, 30, "min"))) == 0.5
    assert _dwell_hours(_record((900, 90, "minutes"))) == 1.5
    assert _dwell_hours(_record((900, 2, "day"))) == 48
    assert _dwell_hours(_record((900, 1, "d"))) == 24
    assert _dwell_hours(_record((900, 3, "hrs"))) == 3
    assert _dwell_hours(_record((900, 5, "hr"))) == 5


def test_an_unknown_unit_is_dropped_rather_than_guessed():
    assert _dwell_hours(_record((900, 30, "s"))) is None


def test_absurd_times_are_dropped():
    """The same guard as for temperature: a value outside what a furnace
    step can be is a parsing artefact."""
    assert _dwell_hours(_record((900, 1000, "day"))) is None    # 24,000 h
    assert _dwell_hours(_record((900, 0.5, "min"))) is None     # 30 seconds


def test_the_prediction_is_rounded_to_a_conventional_value():
    """The corpus's commonest dwell times are 2, 4, 12 and 24 h and the model
    is typically within a factor of two, so 7.3 h would claim a precision it
    does not have. Rounded in log space: 7.3 h is nearer 8 than 6."""
    import torch

    from synflow.dao_s import RecipePredictor, TemperatureNet
    from synflow.recipe_metrics import ConditionalLookup
    from synflow.recipes import Recipe

    lookup = ConditionalLookup(
        [Recipe("BaTiO3", ["BaCO3", "TiO2"], (), (1100,))] * 5, min_count=3)
    network = TemperatureNet(2, hidden=4, layers=1, mean_temperature=10.0)
    dwell = TemperatureNet(2, hidden=4, layers=1,
                           mean_temperature=math.log(7.3), minimum=None)
    with torch.no_grad():
        dwell.head.weight.zero_()      # output is exactly ln(7.3)
        dwell.head.bias.zero_()
    predictor = RecipePredictor(lookup, network, ["BaCO3", "TiO2"],
                                dwell_network=dwell,
                                dwell_vocabulary=["BaCO3", "TiO2"])
    assert predictor.predict_dwell("BaTiO3") == 8
