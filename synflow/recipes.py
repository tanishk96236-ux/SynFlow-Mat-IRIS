"""DAO-S data: text-mined synthesis recipes, parsed and split honestly.

Source: Kononova et al., "Text-mined dataset of inorganic materials synthesis
recipes", Scientific Data 6:203 (2019). 19,488 solid-state recipes, CC BY 4.0.

Two properties of this data shape everything downstream, and both were
measured rather than assumed.

ONE TARGET, MANY RECIPES. 19,488 recipes cover only 13,009 distinct targets --
LiFePO4 alone has 118, BaTiO3 106. 45% of recipes share a target with another.
A random split would put the same compound in train and test, and the model
would score well by recall. So splits are BY TARGET, the same rule the
polymorph work needed after a random split there leaked 92% of validation
compositions into training.

NO SINGLE CORRECT ANSWER. Of the 2,229 targets with more than one recipe, only
38% agree on the precursors. The other 62% are different groups making the same
compound different ways, all of them real. So exact match is the wrong metric:
it would mark a valid recipe wrong for not being the one that happened to be
recorded. Metrics here are set-overlap based, and where a target has several
known recipes, a prediction is credited against the closest of them.
"""

import collections
import json
import os
import re

from .data.periodic_table import ELEMENTS

# Real element symbols. Without this check "YAG" parses as Y + A + G, because
# a capital letter optionally followed by a lowercase one is the shape of an
# element symbol whether or not the element exists. The dataset is full of
# trade names and site placeholders (A0.5Ti0.5NbO4, where A marks a site) that
# pass a purely syntactic test and put invented elements into the composition.
_SYMBOLS = frozenset(v[0] for v in ELEMENTS.values())

DATASET = "data/synthesis/solid-state_dataset_2019-06-27_upd.json"

# figshare article 9722159, the DOI the paper gives. The file id is asked
# for through the API rather than hardcoded, because a new version of the
# dataset gets a new file id and a pinned URL would quietly fetch the old
# one -- or 404 long after anyone remembers why.
_FIGSHARE_API = "https://api.figshare.com/v2/articles/9722159/files"


def download_dataset(path: str = DATASET, expected_md5: str | None =
                     "a60a0107a784c0cc3f564ace5b136676") -> str:
    """Fetch the recipe dataset, 91 MB, CC BY 4.0.

    `data/` is gitignored, so a fresh clone -- on Colab, or on a new machine
    -- has the DAO-S checkpoint but not the corpus it was built from. That
    combination used to kill the discovery run outright.

    The md5 is checked because a truncated download of a 91 MB JSON file
    fails at `json.load` with a parse error that says nothing about the
    cause. Pass expected_md5=None to accept whatever arrives, which is what
    a later version of the dataset would need.
    """
    import hashlib
    import urllib.request

    if os.path.exists(path):
        print("already here: {}".format(path))
        return path

    with urllib.request.urlopen(_FIGSHARE_API, timeout=60) as response:
        files = json.load(response)
    match = next((f for f in files if f["name"] == os.path.basename(path)),
                 files[0])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print("downloading {:.0f} MB from figshare...".format(
        match["size"] / 1e6))
    partial = path + ".part"
    urllib.request.urlretrieve(match["download_url"], partial)

    if expected_md5:
        digest = hashlib.md5()
        with open(partial, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_md5:
            os.remove(partial)
            raise RuntimeError(
                "checksum mismatch: got {}, expected {}. The download was "
                "incomplete or the file has changed.".format(
                    digest.hexdigest(), expected_md5))

    os.replace(partial, path)
    print("written to {}".format(path))
    return path

# A target we can attach a composition to. Free variables (Sr2Ir1-xRuxO4),
# composites and trade names like "YAG" name no single compound.
_CLEAN_FORMULA = re.compile(r"^([A-Z][a-z]?\d*(\.\d+)?)+$")
_ELEMENT_COUNT = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


class Recipe:
    """One codified synthesis: what to make, from what, and how hot."""

    __slots__ = ("target", "precursors", "operations", "temperatures", "doi")

    def __init__(self, target, precursors, operations, temperatures, doi=""):
        self.target = target
        self.precursors = frozenset(precursors)
        self.operations = tuple(operations)
        self.temperatures = tuple(temperatures)
        self.doi = doi

    @property
    def max_temperature(self):
        """The firing temperature -- the hottest step, which is the one that
        forms the phase. Intermediate drying and calcination steps are lower
        and are not what a chemist means by 'the temperature'."""
        return max(self.temperatures) if self.temperatures else None

    def __repr__(self):
        return "Recipe({} <- {} @ {}C)".format(
            self.target, sorted(self.precursors), self.max_temperature)


_GROUP = re.compile(r"\(([^()]+)\)(\d*\.?\d*)")


def _expand_groups(formula: str, limit: int = 8) -> str:
    """Ba(SbO2)2 -> BaSb2O4. Left alone if there is nothing to expand.

    pymatgen writes reduced formulas with parentheses whenever it helps
    readability, so `Composition("BaSb2O4").reduced_formula` is
    "Ba(SbO2)2" -- and the discovery loop hands exactly those strings to
    DAO-S. Without this the parser returned {} for them, and the
    consequences were worse than a crash: the metal-lookup rule found no
    metals, fell through to the network, and the network -- given an
    all-zero composition -- emitted its unconditional prior. Thirty
    different spinels came back with the same six precursors, Bi2O3, CuO,
    Fe2O3, Li2CO3, Na2CO3 and TiO2, none of which contain barium or
    antimony.

    Nested groups are expanded outermost-last by repeating the innermost
    pass, with a limit so a malformed string cannot loop forever.

    A GROUP IT CANNOT FULLY READ IS LEFT ALONE, NOT PARTLY READ
    -----------------------------------------------------------
    This used to keep whatever `_ELEMENT_COUNT` matched and drop the rest of
    the group, which is silent corruption rather than a parse failure. The
    corpus is full of formula templates copied from paper titles, with the
    doping variable left unresolved:

        (Ba1-xCax)(Ti0.85Sn0.15)O3     ->  Ba1Ca1Ti0.85Sn0.15O3
        (Ba1-xEux)2MgSi2O7             ->  Ba2Eu2MgSi2O7
        Ba3Ca1.18Nb1.82O(9-delta)      ->  Ba3Ca1.18Nb1.82O9

    In the first, "1-x" was read as 1 and "x" vanished, so an A site holding
    one atom between them came out holding two. The result then passed the
    clean-formula check, because by that point nothing suspicious was left in
    the string, and went on to become a composition vector and a reaction.

    82 of the 563 targets in the test split -- 14.6% -- are formulas like
    these. So the fix is to refuse: if the matched element-count pairs do not
    reconstruct the group exactly, return the group untouched and let
    `_CLEAN_FORMULA` reject the formula as a whole.
    """
    def expand(match):
        content, multiplier = match.group(1), match.group(2)
        pieces = _ELEMENT_COUNT.findall(content)
        if "".join(element + count for element, count in pieces) != content:
            return match.group(0)
        scale = float(multiplier) if multiplier else 1.0
        return "".join(
            "{}{:g}".format(element, (float(count) if count else 1.0) * scale)
            for element, count in pieces if element)

    for _ in range(limit):
        expanded = _GROUP.sub(expand, formula)
        if expanded == formula:
            return formula
        formula = expanded
    return formula


def composition(formula: str) -> dict:
    """{element: count} from a clean formula. Empty dict if unparseable.

    Every symbol is checked against the real periodic table, not just against
    the shape of a symbol. "YAG" is syntactically Y + A + G and chemically
    nothing; "A0.5Ti0.5NbO4" uses A as a site label. Both would otherwise
    contribute invented elements to a composition vector and to every
    similarity computed from it.
    """
    formula = _expand_groups(formula or "")
    if not _CLEAN_FORMULA.match(formula):
        return {}
    out = collections.Counter()
    for element, count in _ELEMENT_COUNT.findall(formula):
        if not element:
            continue
        if element not in _SYMBOLS:
            return {}
        out[element] += float(count) if count else 1.0
    return dict(out)


def _target_of(raw) -> str:
    target = raw.get("target")
    if isinstance(target, dict):
        for key in ("material_formula", "formula", "material_string"):
            if target.get(key):
                return str(target[key])
    elif isinstance(target, str):
        return target
    return ""


def _precursors_of(raw) -> list:
    out = []
    for p in raw.get("precursors") or []:
        if isinstance(p, dict):
            name = p.get("material_formula") or p.get("material_string")
            if name:
                out.append(str(name))
        elif isinstance(p, str):
            out.append(p)
    return out


def _temperatures_of(raw) -> list:
    values = []
    for op in raw.get("operations") or []:
        conditions = (op or {}).get("conditions") or {}
        for entry in conditions.get("heating_temperature") or []:
            for v in (entry or {}).get("values") or []:
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    pass
    return values


def load_recipes(path: str = DATASET, clean_targets_only: bool = True,
                 require_precursors: bool = True, verbose: bool = True) -> list:
    """Parse the dataset into Recipe objects.

    Args:
        clean_targets_only: keep only targets that name one concrete compound.
            55.9% of raw targets carry free variables or are composites; they
            cannot be given a composition, so a model cannot be conditioned on
            them.
    """
    if not os.path.exists(path):
        raise SystemExit(
            "not found: {}\nDownload it first -- see docs/dao_s_feasibility.md"
            .format(path))

    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    raw_list = blob if isinstance(blob, list) else next(
        blob[k] for k in ("reactions", "data", "recipes") if k in blob)

    recipes, dropped = [], collections.Counter()
    for raw in raw_list:
        target = _target_of(raw)
        if not target:
            dropped["no target"] += 1
            continue
        if clean_targets_only and not composition(target):
            dropped["target not a concrete formula"] += 1
            continue
        precursors = _precursors_of(raw)
        if require_precursors and not precursors:
            dropped["no precursors"] += 1
            continue
        recipes.append(Recipe(
            target=target,
            precursors=precursors,
            operations=[(op or {}).get("type") for op in raw.get("operations") or []],
            temperatures=_temperatures_of(raw),
            doi=str(raw.get("doi", "")),
        ))

    if verbose:
        print("parsed {:,} recipes from {:,} raw entries".format(
            len(recipes), len(raw_list)))
        for reason, n in dropped.most_common():
            print("   dropped {:,}: {}".format(n, reason))
    return recipes


def group_by_target(recipes) -> dict:
    """{target: [Recipe, ...]}. The unit a split must not break apart."""
    groups = collections.defaultdict(list)
    for r in recipes:
        groups[r.target].append(r)
    return dict(groups)


def split_by_target(recipes, seed: int = 0, fractions=(0.8, 0.1, 0.1)) -> dict:
    """Split so every recipe for a target lands on the same side.

    Without this, LiFePO4's 118 recipes scatter across train and test and the
    model scores by recalling one it already saw. 45% of recipes share a target
    with at least one other, so the leak would be large.
    """
    import random

    groups = group_by_target(recipes)
    names = sorted(groups)
    random.Random(seed).shuffle(names)

    n_train = int(fractions[0] * len(names))
    n_val = int(fractions[1] * len(names))
    assignment = {
        "train": names[:n_train],
        "val": names[n_train:n_train + n_val],
        "test": names[n_train + n_val:],
    }
    return {split: [r for name in targets for r in groups[name]]
            for split, targets in assignment.items()}


def precursor_vocabulary(recipes, min_count: int = 1) -> list:
    """Precursors ordered by frequency. 1,845 distinct across the full set,
    with the top 100 covering 88.8% of mentions -- small enough that the
    decoder's output space is not the hard part."""
    counts = collections.Counter(p for r in recipes for p in r.precursors)
    return [p for p, c in counts.most_common() if c >= min_count]


def describe(recipes) -> dict:
    groups = group_by_target(recipes)
    with_temp = [r for r in recipes if r.temperatures]
    multi = {t: v for t, v in groups.items() if len(v) > 1}
    disagreeing = sum(1 for v in multi.values()
                      if len({r.precursors for r in v}) > 1)
    return {
        "recipes": len(recipes),
        "distinct_targets": len(groups),
        "distinct_precursors": len(precursor_vocabulary(recipes)),
        "mean_precursors_per_recipe": (
            sum(len(r.precursors) for r in recipes) / max(len(recipes), 1)),
        "with_temperature": len(with_temp),
        "median_max_temperature": (
            sorted(r.max_temperature for r in with_temp)[len(with_temp) // 2]
            if with_temp else None),
        "targets_with_multiple_recipes": len(multi),
        "of_those_disagreeing_on_precursors": disagreeing,
    }


# --------------------------------------------------------------------------
# precursor name normalisation
# --------------------------------------------------------------------------
#
# 1,005 distinct precursor strings collapse to 975 once notation is
# normalised, and the 26 groups that merge cover 13.6% of all mentions. A
# prediction of "Al2O3" scored as wrong against a recorded "a-Al2O3" is a
# string mismatch, not a chemical one.
#
# But the two levels are not equivalent and are kept apart deliberately.
# Whitespace and the middle dot in hydrates are pure notation. A phase prefix
# is not: a-Al2O3 and g-Al2O3 are different phases with different
# reactivities, and a chemist handed one instead of the other may get a
# different product. Collapsing them makes the score look better by discarding
# real chemistry, so it is offered as an option and reported separately rather
# than applied by default.

_PHASE_PREFIX = re.compile(
    r"^(?:[α-ω]|alpha|beta|gamma|delta|epsilon|lambda)\s*-\s*",
    re.IGNORECASE)
_STATE_SUFFIX = re.compile(r"\s*\(\s*(?:s|l|g|aq|cr)\s*\)$", re.IGNORECASE)


def normalise_precursor(name: str, strip_phase: bool = False) -> str:
    """Canonical form of a precursor name.

    Args:
        strip_phase: also remove phase prefixes. Lossy -- see above.
    """
    s = (name or "").strip()
    s = _STATE_SUFFIX.sub("", s)
    s = s.replace("·", ".").replace(" ", "")
    if strip_phase:
        s = _PHASE_PREFIX.sub("", s)
    return s


def normalise_set(precursors, strip_phase: bool = False) -> frozenset:
    return frozenset(normalise_precursor(p, strip_phase) for p in precursors)
