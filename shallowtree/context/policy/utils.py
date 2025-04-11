""" Module containing helper routines for policies
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shallowtree.chem import TreeMolecule
    from shallowtree.chem.reaction import RetroReaction
    from shallowtree.utils.type_utils import Any, Union


def _make_fingerprint(
    obj: Union[TreeMolecule, RetroReaction], model: Any, chiral: bool = False
) -> np.ndarray:
    fingerprint = obj.fingerprint(radius=2, nbits=len(model), chiral=chiral)
    return fingerprint.reshape([1, len(model)])
