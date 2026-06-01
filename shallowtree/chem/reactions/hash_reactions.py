from __future__ import annotations

import hashlib
from typing import Iterable, Union

from shallowtree.chem.reactions.fixed_retro_reaction import FixedRetroReaction
from shallowtree.chem.reactions.retro_reaction import RetroReaction


def hash_reactions(
        reactions: Union[Iterable[RetroReaction], Iterable[FixedRetroReaction]],
        sort: bool = True,
) -> str:
    """
    Creates a hash for a list of reactions

    :param reactions: the reactions to hash
    :param sort: if True will sort all molecules, defaults to True
    :return: the hash string
    """
    hash_list = []
    for reaction in reactions:
        hash_list.extend(reaction.hash_list())
    if sort:
        hash_list.sort()
    hash_list_str = ".".join(hash_list)
    return hashlib.sha224(hash_list_str.encode("utf8")).hexdigest()

