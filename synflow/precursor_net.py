"""A learned precursor picker, to re-test whether precursor choice is learnable.

On 6,128 training recipes a 211k-parameter network lost to a zero-parameter
rule, 0.596 to 0.733 Jaccard, and the conclusion recorded was that choosing
precursors "is largely not a learning problem". The corpus is now 36,787, and
that conclusion was never re-tested at the larger size.

The bar has also moved. ConditionalLookup -- the rule keyed on (metal, anion
class, oxidation state) with backoff -- reaches 0.724 Jaccard and 55.7% exact
on unseen targets. A network has to beat THAT, not the plain rule.

WHAT A NETWORK CAN SEE THAT THE LOOKUP CANNOT
---------------------------------------------
The lookup conditions on three facts about the target. The network sees the
whole composition, so it can learn interactions -- barium in a hexaferrite
alongside iron is not barium in a titanate -- and it can share evidence
between similar compositions where the lookup, below its trust threshold,
falls all the way back to "how is this metal usually supplied".

THE STRUCTURE IS KEPT
---------------------
Same question as the lookup: for each metal in the target, which precursor
supplies it. The output is masked to precursors that CONTAIN that metal, so
the network can only choose among chemically possible answers; it cannot
propose TiO2 to supply barium. Choosing a different factorisation would make
any comparison with the lookup a comparison of two things at once.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .data.periodic_table import ELEMENTS, MAX_Z
from .dao_s import composition_tensor

_SYMBOL_TO_Z = {v[0]: z for z, v in ELEMENTS.items()}
_CLASSES = ("oxide", "chalcogenide", "nitride", "halide", "carbide", "boride",
            "pnictide", "hydride", "intermetallic")
_STATES = list(range(-4, 9))            # -4 .. +8, and one slot for unknown

N_FEATURES = (MAX_Z + 1) * 2 + (len(_CLASSES) + 1) + (len(_STATES) + 1) + 1


def featurize(target: str, metal: str) -> Optional[torch.Tensor]:
    """[N_FEATURES] for one (target, metal) question, or None if unreadable.

    composition | which metal | anion class | its oxidation state | its share
    """
    from .recipes import composition
    from .valence import anion_class, oxidation_state

    counts = composition(target)
    if not counts or metal not in counts:
        return None
    z = _SYMBOL_TO_Z.get(metal)
    if z is None:
        return None

    which = torch.zeros(MAX_Z + 1)
    which[z] = 1.0

    cls = anion_class(target)
    cls_vec = torch.zeros(len(_CLASSES) + 1)
    cls_vec[_CLASSES.index(cls) if cls in _CLASSES else len(_CLASSES)] = 1.0

    state = oxidation_state(target, metal)
    state_vec = torch.zeros(len(_STATES) + 1)
    state_vec[_STATES.index(state) if state in _STATES
              else len(_STATES)] = 1.0

    share = torch.tensor([counts[metal] / sum(counts.values())])
    return torch.cat([composition_tensor(target), which, cls_vec, state_vec,
                      share])


class PrecursorNet(nn.Module):
    """Logits over the precursor vocabulary, for one (target, metal)."""

    def __init__(self, n_precursors: int, hidden: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(N_FEATURES, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, n_precursors)

    def forward(self, features):
        return self.head(self.body(features))


def supply_masks(vocabulary: List[str]) -> Dict[str, torch.Tensor]:
    """metal -> boolean mask over the vocabulary of precursors containing it.

    Ubiquitous elements are excluded from "containing", as in the lookup:
    every carbonate contains carbon and every oxide oxygen, and neither
    decides which precursor supplies the metal.
    """
    from .recipe_metrics import MetalLookupBaseline
    from .recipes import composition

    ubiquitous = MetalLookupBaseline.UBIQUITOUS
    masks: Dict[str, torch.Tensor] = {}
    for i, precursor in enumerate(vocabulary):
        for metal in frozenset(composition(precursor)) - ubiquitous:
            if metal not in masks:
                masks[metal] = torch.zeros(len(vocabulary), dtype=torch.bool)
            masks[metal][i] = True
    return masks


class NeuralPrecursorPicker:
    """PrecursorNet behind the same interface as ConditionalLookup.

    A metal the network has no candidate for -- one that no vocabulary
    precursor contains -- is answered by the lookup, so the picker never
    returns less than the lookup would for that metal.
    """

    UBIQUITOUS = frozenset({"O", "C", "H", "N"})

    def __init__(self, network: PrecursorNet, vocabulary: List[str],
                 fallback=None):
        self.network = network
        self.vocabulary = list(vocabulary)
        self.masks = supply_masks(self.vocabulary)
        self.fallback = fallback
        self.network.eval()

    def _metals(self, formula: str):
        from .recipes import composition
        return frozenset(composition(formula)) - self.UBIQUITOUS

    def choose(self, target: str, metal: str) -> Optional[str]:
        """The network's pick for one metal, or None if it has no candidate."""
        mask = self.masks.get(metal)
        features = featurize(target, metal)
        if mask is None or features is None:
            return None
        with torch.no_grad():
            logits = self.network(features.unsqueeze(0))[0]
        logits = logits.masked_fill(~mask, float("-inf"))
        return self.vocabulary[int(torch.argmax(logits))]

    def _fallback(self, target: str, metal: str) -> Optional[str]:
        if self.fallback is None:
            return None
        if hasattr(self.fallback, "_choose"):
            from .valence import anion_class, oxidation_state
            precursor, _ = self.fallback._choose(
                metal, anion_class(target), oxidation_state(target, metal))
            return precursor
        return self.fallback.per_metal.get(metal)

    def predict(self, target: str) -> frozenset:
        from .recipe_metrics import MetalLookupBaseline

        chosen = set()
        for metal in self._metals(target):
            precursor = (self.choose(target, metal)
                         or self._fallback(target, metal))
            if precursor:
                chosen.add(precursor)
        return frozenset(p for p in chosen
                         if not MetalLookupBaseline._same_compound(p, target))
