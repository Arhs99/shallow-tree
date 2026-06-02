""" Module containing all types and type imports
"""
from typing import Any, Dict, Optional, Tuple, Union
# pylint: disable=unused-import
from typing import Callable  # noqa
from typing import Iterable  # noqa
from typing import List  # noqa
from typing import Sequence  # noqa
from typing import Set  # noqa
from typing import TypeVar  # noqa

from rdkit import Chem

StrDict = Dict[str, Any]
RdReaction = Chem.rdChemReactions.ChemicalReaction
PilColor = Union[str, Tuple[int, int, int]]
FrameColors = Optional[Dict[bool, PilColor]]
