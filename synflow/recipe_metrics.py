"""Scoring recipe predictions, when there is no single correct answer.

Of the 2,229 targets with more than one recorded recipe, only 38% agree on the
precursors. The other 62% are different groups making the same compound
different ways, and all of them worked. So a metric that demands the exact
recorded set punishes a model for proposing a real alternative, and would have
told us DAO-S was failing when it was not.

Three consequences, built in here:

* Set overlap, not exact match. Jaccard credits a prediction that shares most
  precursors with the reference.
* Where a target has several known recipes, the prediction is scored against
  the CLOSEST of them. Getting any real route right is getting it right.
* Exact match is still reported, as the strictest reading, but it is a floor
  on performance rather than the measure of it.

The baseline every model must clear is retrieval: return the recipe of the
most chemically similar target in training. If a network cannot beat looking
up the nearest neighbour, it has learned nothing a lookup table does not
already hold -- which is the bar written into the roadmap for Phase IV.
"""

import collections

from .recipes import composition


def jaccard(predicted, reference) -> float:
    """|intersection| / |union| over two precursor sets."""
    predicted, reference = set(predicted), set(reference)
    if not predicted and not reference:
        return 1.0
    union = predicted | reference
    return len(predicted & reference) / len(union) if union else 0.0


def best_over_references(predicted, references) -> float:
    """Jaccard against the closest known recipe for this target.

    A target with three recorded routes has three right answers, and matching
    any of them is success.
    """
    return max((jaccard(predicted, r) for r in references), default=0.0)


def precision_recall_f1(predicted, reference):
    predicted, reference = set(predicted), set(reference)
    if not predicted or not reference:
        return 0.0, 0.0, 0.0
    hits = len(predicted & reference)
    precision = hits / len(predicted)
    recall = hits / len(reference)
    f1 = (2 * precision * recall / (precision + recall)) if hits else 0.0
    return precision, recall, f1


def evaluate(predictions, references_by_target, temperatures=None,
             true_temperatures=None) -> dict:
    """Score a set of predictions.

    Args:
        predictions: {target: predicted precursor set}
        references_by_target: {target: [precursor set, ...]} -- every recorded
            recipe for that target, since any of them is a valid answer
        temperatures: optional {target: predicted max temperature, C}
        true_temperatures: optional {target: true max temperature, C}
    """
    jaccards, exact, f1s = [], 0, []
    for target, predicted in predictions.items():
        references = references_by_target.get(target, [])
        if not references:
            continue
        jaccards.append(best_over_references(predicted, references))
        exact += any(set(predicted) == set(r) for r in references)
        # precision/recall against the closest reference, for the same reason
        closest = max(references, key=lambda r: jaccard(predicted, r))
        f1s.append(precision_recall_f1(predicted, closest)[2])

    n = len(jaccards)
    result = {
        "targets_scored": n,
        "jaccard": sum(jaccards) / n if n else float("nan"),
        "exact_match": exact / n if n else float("nan"),
        "f1": sum(f1s) / n if n else float("nan"),
    }

    if temperatures and true_temperatures:
        errors = [abs(temperatures[t] - true_temperatures[t])
                  for t in temperatures
                  if t in true_temperatures
                  and temperatures[t] is not None
                  and true_temperatures[t] is not None]
        result["temperature_mae_C"] = (
            sum(errors) / len(errors) if errors else float("nan"))
        result["temperature_scored"] = len(errors)
    return result


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def composition_vector(formula: str) -> dict:
    """Element fractions, so Fe2O3 and Fe4O6 look the same."""
    counts = composition(formula)
    total = sum(counts.values())
    return {e: c / total for e, c in counts.items()} if total else {}


def composition_similarity(a: str, b: str) -> float:
    """Cosine similarity between two formulas' element fractions."""
    va, vb = composition_vector(a), composition_vector(b)
    if not va or not vb:
        return 0.0
    shared = set(va) & set(vb)
    dot = sum(va[e] * vb[e] for e in shared)
    na = sum(v * v for v in va.values()) ** 0.5
    nb = sum(v * v for v in vb.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class RetrievalBaseline:
    """Return the recipe of the most chemically similar training target.

    This is the bar. A recipe model that cannot beat it has learned nothing
    beyond what a nearest-neighbour lookup already provides -- and given that
    similar compositions genuinely do use similar precursors, the bar is not
    low.
    """

    def __init__(self, train_recipes):
        self.by_target = collections.defaultdict(list)
        for r in train_recipes:
            self.by_target[r.target].append(r)
        self.targets = list(self.by_target)
        # index by element set, so a candidate sharing no elements is never
        # compared -- this is what makes the baseline fast enough to be fair
        self.by_element = collections.defaultdict(set)
        for t in self.targets:
            for element in composition(t):
                self.by_element[element].add(t)

    def nearest(self, target: str):
        candidates = set()
        for element in composition(target):
            candidates |= self.by_element.get(element, set())
        candidates.discard(target)
        if not candidates:
            return None
        return max(candidates, key=lambda t: composition_similarity(target, t))

    def predict(self, target: str):
        """Predicted precursor set, or empty if nothing comparable exists."""
        neighbour = self.nearest(target)
        if neighbour is None:
            return frozenset()
        return self.by_target[neighbour][0].precursors

    def predict_temperature(self, target: str):
        neighbour = self.nearest(target)
        if neighbour is None:
            return None
        temps = [r.max_temperature for r in self.by_target[neighbour]
                 if r.max_temperature is not None]
        return sum(temps) / len(temps) if temps else None


class MostCommonBaseline:
    """Always predict the most frequent precursor set. The true floor --
    it ignores the target entirely, so anything at this level has learned
    nothing about chemistry at all."""

    def __init__(self, train_recipes):
        counts = collections.Counter(r.precursors for r in train_recipes)
        self.answer = counts.most_common(1)[0][0] if counts else frozenset()
        temps = [r.max_temperature for r in train_recipes
                 if r.max_temperature is not None]
        self.temperature = sorted(temps)[len(temps) // 2] if temps else None

    def predict(self, target: str):
        return self.answer

    def predict_temperature(self, target: str):
        return self.temperature


class MetalLookupBaseline:
    """For each metal in the target, the precursor most often used to supply it.

    A chemical rule plus a frequency count. No learning, no parameters, no
    training loop -- and on held-out targets it reaches jaccard 0.711 and 55.4%
    exact, against 0.508 / 19.7% for nearest-composition retrieval and
    0.572 / 31.3% for a 211k-parameter network trained for 60 epochs.

    This is the bar DAO-S actually has to clear, and the earlier retrieval
    baseline was too easy. Precursor selection is largely not a learning
    problem: you need barium, so you reach for BaCO3.

    Ubiquitous elements are excluded from the matching. 87% of this data is
    oxides, so oxygen connects nearly everything to everything -- it is the
    metal that decides which precursor supplies what.
    """

    UBIQUITOUS = frozenset({"O", "C", "H", "N"})

    def __init__(self, train_recipes):
        counts = collections.defaultdict(collections.Counter)
        temperatures = collections.defaultdict(list)
        for r in train_recipes:
            metals = self._metals(r.target)
            for precursor in r.precursors:
                for metal in self._metals(precursor) & metals:
                    counts[metal][precursor] += 1
            if r.max_temperature is not None:
                for metal in metals:
                    temperatures[metal].append(r.max_temperature)

        self.per_metal = {m: c.most_common(1)[0][0] for m, c in counts.items()}
        self.temperature_per_metal = {
            m: sorted(v)[len(v) // 2] for m, v in temperatures.items() if v}
        all_temps = [r.max_temperature for r in train_recipes
                     if r.max_temperature is not None]
        self.fallback_temperature = (
            sorted(all_temps)[len(all_temps) // 2] if all_temps else None)

    @classmethod
    def _metals(cls, formula: str) -> frozenset:
        return frozenset(composition(formula)) - cls.UBIQUITOUS

    @staticmethod
    def _same_compound(a: str, b: str) -> bool:
        """Whether two formulas name the same compound.

        Compared by composition rather than by string, because "make X from X"
        is just as circular written two different ways, and this project has
        already been caught comparing raw formula strings -- "Li3NiMnCoO6"
        against Materials Project's "Li3MnCoNiO6" -- and concluding the
        database held no NMC cathodes.
        """
        left, right = composition(a), composition(b)
        if not left or not right:
            return False
        total_l, total_r = sum(left.values()), sum(right.values())
        if not total_l or not total_r:
            return False
        keys = set(left) | set(right)
        return all(abs(left.get(k, 0.0) / total_l
                       - right.get(k, 0.0) / total_r) < 1e-6 for k in keys)

    def predict(self, target: str) -> frozenset:
        """Precursors for this target, never including the target itself.

        WHY THE EXCLUSION IS NEEDED
        ---------------------------
        The lookup is per metal, and many targets are themselves standard
        reagents. Asked for Al2O3 it returned {Al2O3}, because Al2O3 is the
        commonest way to supply aluminium -- which is true, and useless:
        "make Al2O3 from Al2O3" is not a recipe. The same happened for TiO2,
        Fe2O3 and ZnO, all of which are both common targets and common
        precursors.

        It did not look like a failure. It looked like a confident one-step
        synthesis, and it would have been handed to the simulator, which would
        have balanced X -> X, found dH = 0, and reported it as feasible.

        Only the target itself is removed, not every precursor that is a
        target somewhere else. Ba2TiO4 from BaTiO3 and BaCO3 is a real route,
        and BaTiO3 is a target in its own right; excluding it would throw away
        correct chemistry.

        An empty set after the exclusion is returned as empty. HybridDAOS
        already treats that as "the rule knows nothing here" and falls back,
        and a refusal is the right answer for a compound whose only known
        source is itself.
        """
        chosen = frozenset(self.per_metal[m] for m in self._metals(target)
                           if m in self.per_metal)
        return frozenset(p for p in chosen
                         if not self._same_compound(p, target))

    def predict_temperature(self, target: str):
        values = [self.temperature_per_metal[m] for m in self._metals(target)
                  if m in self.temperature_per_metal]
        return sum(values) / len(values) if values else self.fallback_temperature


class ConditionalLookup(MetalLookupBaseline):
    """The metal-lookup rule, made to look at the target before choosing.

    The plain rule answers "how is iron usually supplied?" once, for every
    target. Measured per metal on held-out recipes it is right 67.0% of the
    time, and its misses are not random:

        Fe(3+) targets   n=1,541   the rule's Fe2O3 is right 85%
        Fe(2+) targets   n=418     the rule's Fe2O3 is right 28%

    and it errs in both directions -- an oxide for targets with no oxygen,
    where iron, cobalt, aluminium and silicon are supplied as the bare metal,
    and the bare metal for oxides, where it picks Ni because nickel alloys
    dominate its count while NiO is what an oxide synthesis uses.

    So the question is asked at three levels of detail, most specific first:

        1. (metal, anion class of the target, oxidation state of the metal)
        2. (metal, anion class)
        3. (metal)                         -- the plain rule

    and a level is trusted only when training saw that key at least
    `min_count` times. Below that the answer comes from the coarser level,
    which is why this cannot do worse than the plain rule on any key it has
    never seen: it IS the plain rule there.

    No training loop and no parameters. This tests whether context matters
    at all before anything is learned.
    """

    def __init__(self, train_recipes, min_count: int = 5,
                 use_state: bool = True):
        super().__init__(train_recipes)
        from .valence import anion_class, oxidation_state

        self.min_count = min_count
        self.use_state = use_state
        self.by_state = collections.defaultdict(collections.Counter)
        self.by_anion = collections.defaultdict(collections.Counter)

        for recipe in train_recipes:
            cls = anion_class(recipe.target)
            if cls is None:
                continue
            metals = self._metals(recipe.target)
            for precursor in recipe.precursors:
                for metal in self._metals(precursor) & metals:
                    self.by_anion[(metal, cls)][precursor] += 1
                    if use_state:
                        state = oxidation_state(recipe.target, metal)
                        if state is not None:
                            self.by_state[(metal, cls, state)][precursor] += 1

    def _choose(self, metal: str, cls, state):
        """The most specific trusted answer for one metal, and which level
        supplied it."""
        levels = []
        if self.use_state and state is not None:
            levels.append(("state", self.by_state.get((metal, cls, state))))
        levels.append(("anion", self.by_anion.get((metal, cls))))
        for name, counter in levels:
            if counter and sum(counter.values()) >= self.min_count:
                return counter.most_common(1)[0][0], name
        return self.per_metal.get(metal), "metal"

    def predict(self, target: str) -> frozenset:
        from .valence import anion_class, oxidation_state

        cls = anion_class(target)
        chosen = set()
        for metal in self._metals(target):
            state = oxidation_state(target, metal) if self.use_state else None
            precursor, _ = self._choose(metal, cls, state)
            if precursor:
                chosen.add(precursor)
        return frozenset(p for p in chosen
                         if not self._same_compound(p, target))

    def explain(self, target: str) -> list:
        """Which level answered for each metal -- so a prediction can be
        traced to the evidence behind it."""
        from .valence import anion_class, oxidation_state

        cls = anion_class(target)
        out = []
        for metal in sorted(self._metals(target)):
            state = oxidation_state(target, metal) if self.use_state else None
            precursor, level = self._choose(metal, cls, state)
            out.append({"metal": metal, "anion_class": cls, "state": state,
                        "precursor": precursor, "level": level})
        return out

    # ------------------------------------------------------------------
    # persistence
    #
    # Fitting needs an oxidation state for every target in the corpus --
    # about 20,000 of them, a few minutes -- and the tables it produces do
    # not change unless the corpus does. Rebuilding them on every pipeline
    # run would make each run start with minutes of identical arithmetic.
    # Saved as JSON rather than pickled, so the file stays readable and
    # survives changes to this class.
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        def flat(table):
            return {"|".join(str(part) for part in key): dict(counter)
                    for key, counter in table.items()}
        return {"min_count": self.min_count, "use_state": self.use_state,
                "per_metal": dict(self.per_metal),
                "by_anion": flat(self.by_anion),
                "by_state": flat(self.by_state)}

    @classmethod
    def from_json(cls, blob: dict) -> "ConditionalLookup":
        model = cls.__new__(cls)
        model.min_count = blob["min_count"]
        model.use_state = blob["use_state"]
        model.per_metal = dict(blob["per_metal"])
        model.temperature_per_metal = {}
        model.fallback_temperature = None

        def unflat(table, with_state):
            out = collections.defaultdict(collections.Counter)
            for key, counts in table.items():
                parts = key.split("|")
                if with_state:
                    parts[2] = int(parts[2])
                out[tuple(parts)] = collections.Counter(counts)
            return out

        model.by_anion = unflat(blob["by_anion"], with_state=False)
        model.by_state = unflat(blob["by_state"], with_state=True)
        return model
