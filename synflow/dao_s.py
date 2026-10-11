"""DAO-S: predict a synthesis recipe from a target composition.

One model, two heads, a shared encoder: which precursors to mix, and how hot
to fire. Both are what a chemist needs before they can attempt anything.

WHY NOT AUTOREGRESSIVE
----------------------
The blueprint specifies a sequence-to-sequence transformer. The output here is
not a sequence. {Li2CO3, FeC2O4} and {FeC2O4, Li2CO3} are the same recipe --
precursors form a SET, and the dataset records no order among them.

An autoregressive decoder would have to be taught an ordering the data does
not contain, spending capacity on an artefact. With 6,737 training recipes and
two overfitting failures already behind this project, that is capacity we do
not have. Multi-label classification is the native formulation for a set: one
independent decision per precursor.

Sequence modelling becomes right later, for the full recipe -- the operation
order, durations and ramp rates, where order carries real information. This
predicts the two fields that decide whether a synthesis can be attempted at
all.

WHAT THE BASELINES SAY THIS MUST BEAT
-------------------------------------
    most-common   jaccard 0.072   temp MAE 256 C
    retrieval     jaccard 0.508   temp MAE 204 C

Retrieval -- copy the recipe of the nearest composition -- is strong on
precursors and weak on temperature. Composition largely determines what to
mix; it barely constrains how hot. The room for a model is mostly in the
second column.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .data.periodic_table import ELEMENTS, MAX_Z
from .recipes import composition

_SYMBOL_TO_Z = {v[0]: z for z, v in ELEMENTS.items()}

# Elements shared by almost every compound in this data, which therefore say
# nothing about whether a precursor suits a target. 87% of these syntheses are
# oxides, so matching on oxygen links essentially everything to everything --
# CuO and BaTiO3 share it, and CuO makes no barium titanate. What carries the
# information is the metal: BaCO3 makes BaTiO3 because of the barium.
_UBIQUITOUS = frozenset({"O", "C", "H", "N"})


def composition_tensor(formula: str) -> torch.Tensor:
    """[MAX_Z + 1] element fractions.

    Fractions, not counts, so Fe2O3 and Fe4O6 -- the same compound written
    twice -- give the same vector.
    """
    vector = torch.zeros(MAX_Z + 1)
    counts = composition(formula)
    total = sum(counts.values())
    if not total:
        return vector
    for symbol, count in counts.items():
        z = _SYMBOL_TO_Z.get(symbol)
        if z is not None:
            vector[z] = count / total
    return vector


class RecipeNet(nn.Module):
    """Shared composition encoder, one head per output.

    Deliberately small. 6,737 training recipes against a 1,044-way output is
    already a stretch, and this project's two previous models both fitted
    their training data perfectly and generalised to nothing.
    """

    def __init__(self, n_precursors: int, hidden: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(MAX_Z + 1, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
        )
        # logits, not probabilities: BCEWithLogitsLoss is the numerically
        # stable pairing and lets the threshold be tuned after training
        self.precursor_head = nn.Linear(hidden, n_precursors)
        # temperature is predicted in units of 100 C so it sits near the scale
        # of the precursor logits and neither loss drowns the other
        self.temperature_head = nn.Linear(hidden, 1)

    def forward(self, composition_batch):
        h = self.encoder(composition_batch)
        return self.precursor_head(h), self.temperature_head(h).squeeze(-1)


class TemperatureNet(nn.Module):
    """Firing temperature from the composition AND the precursors.

    THREE MEASURED REASONS THIS EXISTS SEPARATELY FROM RecipeNet
    -----------------------------------------------------------
    1. RecipeNet shares one encoder between a 463-way precursor head and a
       single temperature output, and `HybridDAOS` then DISCARDS the precursor
       predictions in favour of the metal-lookup rule, which beats them 0.733
       to 0.596. So the representation the temperature head reads was being
       shaped largely by a task whose output is thrown away, with no weighting
       between the two losses.

    2. More than half of RecipeNet's 211k parameters -- 119k of them -- sit in
       that discarded head. Asking whether the model is "too small" is not
       answerable while most of it is unused.

    3. The precursors were never an input. The rule picks them BEFORE the
       temperature is needed, and which precursors you use is one of the
       strongest things deciding how hot you fire: a carbonate route has to
       get past carbonate decomposition, a nitrate or oxalate route does not.
       That signal was sitting in the pipeline, already computed, and the
       regression could not see it.

    Reason 3 is the one expected to matter. The old number was 183 C mean
    error against a median firing temperature of 1050 C, and the stated
    explanation was that composition alone does not record what decides a
    temperature -- which was true, and also not the whole story, because the
    precursors were available and unused.

    The precursors enter as a multi-hot vector over the same vocabulary the
    precursor head used, so no new data and no new featurisation is involved.
    """

    def __init__(self, n_precursors: int, hidden: int = 256,
                 layers: int = 2, dropout: float = 0.2,
                 mean_temperature: float = 10.0,
                 minimum: Optional[float] = 0.5):
        super().__init__()
        # The floor stops a negative firing temperature. The same network
        # predicts ln(dwell hours) for the dwell model, where a negative
        # output is a real answer -- ln(0.5 h) -- so there it is None.
        # Not a buffer: checkpoints saved before this existed load unchanged.
        self.minimum = minimum
        widths = [MAX_Z + 1 + n_precursors] + [hidden] * layers
        stack: List[nn.Module] = []
        for inp, out in zip(widths, widths[1:]):
            stack += [nn.Linear(inp, out), nn.LayerNorm(out), nn.SiLU(),
                      nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*stack)
        self.head = nn.Linear(hidden, 1)

        # Predict a RESIDUAL from the training mean rather than the value.
        # The target is around 10.5 in units of 100 C with a spread of about
        # 3, so a head initialised near zero starts roughly 1000 C wrong and
        # spends its early epochs travelling rather than fitting. Registered
        # as a buffer so it is saved with the checkpoint and cannot drift
        # between training and inference.
        self.register_buffer("mean_temperature",
                             torch.tensor(float(mean_temperature)))

        # LayerNorm is new here too. The input is element fractions, mostly
        # zero, summing to one across about four non-zero entries, now
        # concatenated with a multi-hot of about three non-zero entries out of
        # several hundred. Without normalisation the two blocks arrive at very
        # different scales.
        nn.init.zeros_(self.head.bias)

    def forward(self, composition_batch, precursor_batch):
        features = torch.cat([composition_batch, precursor_batch], dim=-1)
        residual = self.head(self.encoder(features)).squeeze(-1)
        out = residual + self.mean_temperature
        # Clamped above zero for temperature: the output is in units of 100 C
        # and a negative firing temperature is not a cautious prediction, it
        # is a wrong one.
        if self.minimum is not None:
            out = torch.clamp(out, min=self.minimum)
        return out


class DAOS:
    """Trained recipe predictor.

    `threshold` decides how many precursors come back. It is tuned on
    validation rather than left at 0.5: the head is trained on a
    heavily imbalanced target -- about 2.8 positives out of 1,044 -- so the
    calibrated cut is far below a half.
    """

    TEMPERATURE_SCALE = 100.0

    def __init__(self, model: RecipeNet, vocabulary: Sequence[str],
                 threshold: float = 0.5, chemical_mask: bool = True):
        self.model = model
        self.vocabulary = list(vocabulary)
        self.threshold = threshold
        self.chemical_mask = chemical_mask
        self._elements = [frozenset(composition(p)) for p in self.vocabulary]

    def _allowed(self, target: str):
        """Which precursors could plausibly make this target.

        Real precursors carry one of the target's metals: you make BaTiO3
        from BaCO3 and TiO2. Retrieval gets this free by copying a neighbour's
        recipe; the network was being made to rediscover it across 463
        independent decisions from 6,128 examples.

        Matching on any shared element is not enough. 87% of this data is
        oxides, so oxygen links nearly every precursor to nearly every target
        and the filter passes CuO for BaTiO3. Ubiquitous elements are excluded
        so the match is on the metal, which is what actually decides it.

        Supplying it directly is a chemistry prior, not a leak -- it uses only
        the target's own formula, which is the model's input.
        """
        target_elements = frozenset(composition(target)) - _UBIQUITOUS
        if not target_elements:
            return None
        return torch.tensor([bool((e - _UBIQUITOUS) & target_elements) or not e
                             for e in self._elements])

    @torch.no_grad()
    def _logits(self, targets: Sequence[str]):
        self.model.eval()
        x = torch.stack([composition_tensor(t) for t in targets])
        return self.model(x)

    # Real recipes use 2.8 precursors on average and rarely more than six. A
    # prediction of fifty is not a cautious answer, it is an unusable one, and
    # it wrecks precision while flattering recall.
    MAX_PRECURSORS = 6

    def _select(self, probabilities, min_precursors: int, allowed=None):
        if allowed is not None and self.chemical_mask:
            probabilities = probabilities * allowed.to(probabilities.dtype)
        chosen = (probabilities >= self.threshold).nonzero().flatten().tolist()
        if len(chosen) < min_precursors:
            chosen = probabilities.topk(min_precursors).indices.tolist()
        elif len(chosen) > self.MAX_PRECURSORS:
            chosen = probabilities.topk(self.MAX_PRECURSORS).indices.tolist()
        return frozenset(self.vocabulary[i] for i in chosen)

    def predict(self, target: str, min_precursors: int = 1) -> frozenset:
        """Predicted precursor set.

        Never returns nothing: a recipe with no ingredients is not a
        conservative answer, it is an unusable one, so the most likely
        candidate is kept even when every probability is below the threshold.
        """
        logits, _ = self._logits([target])
        return self._select(torch.sigmoid(logits[0]), min_precursors,
                            self._allowed(target))

    def predict_temperature(self, target: str) -> float:
        _, temperature = self._logits([target])
        return float(temperature[0]) * self.TEMPERATURE_SCALE

    def predict_batch(self, targets: Sequence[str], min_precursors: int = 1):
        logits, temperatures = self._logits(list(targets))
        probabilities = torch.sigmoid(logits)
        return [(self._select(row, min_precursors, self._allowed(target)),
                 float(temperature) * self.TEMPERATURE_SCALE)
                for target, row, temperature
                in zip(targets, probabilities, temperatures)]


def encode_batch(recipes, vocabulary: Sequence[str]):
    """Recipes -> (compositions, precursor targets, temperatures, mask).

    The mask marks recipes that record a temperature. 24% do not, and training
    a regressor against an invented value would teach it the invention.
    """
    index = {p: i for i, p in enumerate(vocabulary)}
    x = torch.stack([composition_tensor(r.target) for r in recipes])
    y = torch.zeros(len(recipes), len(vocabulary))
    temperatures = torch.zeros(len(recipes))
    mask = torch.zeros(len(recipes), dtype=torch.bool)

    for row, recipe in enumerate(recipes):
        for precursor in recipe.precursors:
            if precursor in index:
                y[row, index[precursor]] = 1.0
        if recipe.max_temperature is not None:
            temperatures[row] = recipe.max_temperature / DAOS.TEMPERATURE_SCALE
            mask[row] = True

    return x, y, temperatures, mask


def tune_threshold(model: RecipeNet, vocabulary: Sequence[str], recipes,
                   references_by_target, candidates: Optional[List[float]] = None):
    """Pick the probability cut that maximises Jaccard on validation.

    With ~2.8 positives among 1,044 classes the default 0.5 returns almost
    nothing, so leaving it there would report a failure caused by the cut
    rather than by the model.
    """
    from .recipe_metrics import evaluate

    # The range must span BOTH sides of 0.5. Training uses a large pos_weight
    # -- without it the loss is minimised by predicting nothing among ~2.8
    # positives in 463 classes -- and that pushes probabilities UP, so the
    # calibrated cut sits above 0.5, not below. A search capped at 0.5 returned
    # 0.5 every time and 58 precursors per recipe against a true mean of 2.8.
    candidates = candidates or [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9,
                                0.95, 0.98, 0.99, 0.995]
    targets = sorted({r.target for r in recipes})
    best = (0.5, -1.0)
    for threshold in candidates:
        daos = DAOS(model, vocabulary, threshold)
        predictions = dict(zip(targets,
                               (p for p, _ in daos.predict_batch(targets))))
        score = evaluate(predictions, references_by_target)["jaccard"]
        if score > best[1]:
            best = (threshold, score)
    return best


class HybridDAOS:
    """The honest best system: rule for precursors, network for temperature.

    Measured on held-out targets, neither component wins both columns:

        metal-lookup rule   jaccard 0.711   exact 55.4%   temp MAE 203 C
        network             jaccard 0.572   exact 31.3%   temp MAE 186 C

    So each does the half it is better at. Giving the network the precursors
    as well would cost 0.14 Jaccard and 24 points of exact match to keep one
    model tidy, and the point of DAO-S is to produce recipes a chemist could
    follow, not to be architecturally uniform.

    That split is not an accident of this dataset. Precursor choice is close
    to deterministic chemistry -- you need barium, you reach for BaCO3 -- and
    counting measures a deterministic mapping more accurately than 6,128
    examples can teach one. Firing temperature depends on particle size,
    atmosphere, dwell time and what the group had in the furnace, none of
    which the composition records, so no rule captures it and a model has room
    to find weak regularities.
    """

    def __init__(self, rule, network: "DAOS"):
        self.rule = rule
        self.network = network

    def predict(self, target: str) -> frozenset:
        """Precursors from the rule; the network's set is discarded.

        Falls back to the network when the rule knows none of the target's
        metals -- a composition built from elements absent from training.
        """
        precursors = self.rule.predict(target)
        return precursors if precursors else self.network.predict(target)

    def predict_temperature(self, target: str) -> float:
        return self.network.predict_temperature(target)

    def predict_recipe(self, target: str) -> Optional[dict]:
        """What a chemist would actually be handed, or None.

        None when the formula cannot be turned into a composition at all.
        Both halves of this model are then answering a question neither was
        asked: the rule finds no metals and returns nothing, and the network
        receives an all-zero composition vector and emits its unconditional
        prior -- the training set's six most common precursors, identical for
        every unparseable target and containing elements the compound does
        not have.

        That was not a hypothetical. A spinel sweep produced thirty recipes
        reading "Bi2O3, CuO, Fe2O3, Li2CO3, Na2CO3, TiO2 at 749 C" for
        thirty different compounds, and it looked like output rather than
        like failure. A refusal is the correct answer and the only one a
        chemist can act on.
        """
        if not composition(target):
            return None
        return {
            "target": target,
            "precursors": sorted(self.predict(target)),
            "firing_temperature_C": round(self.predict_temperature(target)),
        }

    def __repr__(self):
        return "HybridDAOS(precursors=rule, temperature=network)"


class RecipePredictor:
    """The shipped DAO-S: context-aware lookup for precursors, TemperatureNet
    for the firing temperature.

    Replaces HybridDAOS, which paired the plain metal-lookup rule with the
    old RecipeNet. Measured on 981 targets from the hand-built corpus that
    appear nowhere in training:

                                precursors          temperature
        plain rule + RecipeNet  0.662 J, 47.5% ex   (trained on a corrupted
                                                     corpus, selected on the
                                                     wrong metric)
        this                    0.724 J, 55.7% ex   TemperatureNet

    The network's precursor input is the lookup's own prediction, and the
    checkpoint records which predictor it was trained against. Loading a
    network trained on one predictor behind another is refused rather than
    allowed to run on inputs it never saw.
    """

    # Dwell times in the corpus are overwhelmingly round numbers -- 2, 4, 12,
    # 24 h are the commonest four -- and the model is typically within a
    # factor of two. Printing "7.3 h" would claim a precision it does not
    # have, so the protocol states the nearest conventional value.
    CONVENTIONAL_HOURS = (0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 36,
                          48, 72)

    def __init__(self, lookup, network: "TemperatureNet", vocabulary,
                 dwell_network: Optional["TemperatureNet"] = None,
                 dwell_vocabulary=None):
        self.lookup = lookup
        self.network = network
        self.vocabulary = list(vocabulary)
        self._index = {p: i for i, p in enumerate(self.vocabulary)}
        self.network.eval()
        self.dwell_network = dwell_network
        self.dwell_vocabulary = list(dwell_vocabulary or [])
        self._dwell_index = {p: i for i, p in
                             enumerate(self.dwell_vocabulary)}
        if self.dwell_network is not None:
            self.dwell_network.eval()

    def predict_dwell(self, target: str, precursors=None) -> Optional[float]:
        """Hours at the firing temperature, per cycle, rounded to the nearest
        conventional value. None if no dwell model is loaded."""
        import math

        if self.dwell_network is None:
            return None
        if precursors is None:
            precursors = self.predict(target)
        multi_hot = torch.zeros(len(self.dwell_vocabulary))
        for precursor in precursors:
            index = self._dwell_index.get(precursor)
            if index is not None:
                multi_hot[index] = 1.0
        with torch.no_grad():
            log_hours = float(self.dwell_network(
                composition_tensor(target).unsqueeze(0),
                multi_hot.unsqueeze(0))[0])
        return min(self.CONVENTIONAL_HOURS,
                   key=lambda h: abs(math.log(h) - log_hours))

    def predict(self, target: str) -> frozenset:
        return self.lookup.predict(target)

    def predict_temperature(self, target: str,
                            precursors=None) -> Optional[float]:
        if precursors is None:
            precursors = self.predict(target)
        multi_hot = torch.zeros(len(self.vocabulary))
        for precursor in precursors:
            index = self._index.get(precursor)
            if index is not None:
                multi_hot[index] = 1.0
        with torch.no_grad():
            units = self.network(composition_tensor(target).unsqueeze(0),
                                 multi_hot.unsqueeze(0))
        return float(units[0]) * DAOS.TEMPERATURE_SCALE

    def predict_recipe(self, target: str) -> Optional[dict]:
        """What a chemist would be handed, or None.

        None when the formula cannot be read -- the same refusal HybridDAOS
        made, for the same reason: an unreadable formula gives the network an
        all-zero composition, and it answers with its prior, the same recipe
        for every such target. A refusal is the only answer a chemist can act
        on.
        """
        from .recipes import composition as parse
        if not parse(target):
            return None
        precursors = self.predict(target)
        if not precursors:
            return None
        return {
            "target": target,
            "precursors": sorted(precursors),
            "firing_temperature_C": round(
                self.predict_temperature(target, precursors)),
            "dwell_hours": self.predict_dwell(target, precursors),
        }

    def explain(self, target: str) -> list:
        return self.lookup.explain(target)

    def __repr__(self):
        return "RecipePredictor(precursors=ConditionalLookup(min_count={}), " \
               "temperature=TemperatureNet)".format(self.lookup.min_count)


def load_shipped(checkpoint: str = "runs/dao_s_temperature/best.pt",
                 lookup_cache: str = "runs/precursor_lookup.json",
                 verbose: bool = True,
                 dwell_checkpoint: Optional[str] = "runs/dao_s_dwell/best.pt",
                 ) -> "RecipePredictor":
    """The one place the shipped DAO-S is assembled.

    The lookup is fitted on the WHOLE larger corpus -- it is applied to new
    generated materials, none of which are in it -- and cached, because
    fitting computes an oxidation state for about 20,000 targets.
    """
    import json
    import os

    from .recipe_metrics import ConditionalLookup

    blob = torch.load(checkpoint, weights_only=False, map_location="cpu")
    if blob.get("precursors") != "conditional":
        raise RuntimeError(
            "{} was trained on precursor inputs from {!r}, not the "
            "conditional lookup it would be paired with here -- retrain with "
            "scripts/train_temperature.py --precursors conditional".format(
                checkpoint, blob.get("precursors", "the plain rule")))
    min_count = blob.get("min_count", 3)

    lookup = None
    if os.path.exists(lookup_cache):
        with open(lookup_cache, encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("min_count") == min_count:
            lookup = ConditionalLookup.from_json(cached)
    if lookup is None:
        from .recipes_llm import load_recipes
        if verbose:
            print("fitting the precursor lookup on the full corpus "
                  "(once; cached to {})".format(lookup_cache))
        lookup = ConditionalLookup(load_recipes(verbose=False),
                                   min_count=min_count)
        os.makedirs(os.path.dirname(lookup_cache) or ".", exist_ok=True)
        with open(lookup_cache, "w", encoding="utf-8") as handle:
            json.dump(lookup.to_json(), handle)

    network = TemperatureNet(len(blob["vocabulary"]), hidden=blob["hidden"],
                             layers=blob["layers"], dropout=blob["dropout"],
                             mean_temperature=blob["mean_temperature"])
    network.load_state_dict(blob["model"])

    # The dwell model is optional: without it the protocol states that the
    # dwell time is not predicted rather than inventing one.
    dwell_network, dwell_vocabulary = None, None
    if dwell_checkpoint and os.path.exists(dwell_checkpoint):
        dwell = torch.load(dwell_checkpoint, weights_only=False,
                           map_location="cpu")
        if dwell.get("precursors") != "conditional":
            raise RuntimeError("{} was not trained on the conditional "
                               "lookup's precursors".format(dwell_checkpoint))
        dwell_network = TemperatureNet(
            len(dwell["vocabulary"]), hidden=dwell["hidden"],
            layers=dwell["layers"], dropout=dwell["dropout"],
            mean_temperature=dwell["mean_temperature"], minimum=None)
        dwell_network.load_state_dict(dwell["model"])
        dwell_vocabulary = dwell["vocabulary"]

    return RecipePredictor(lookup, network, blob["vocabulary"],
                           dwell_network=dwell_network,
                           dwell_vocabulary=dwell_vocabulary)
