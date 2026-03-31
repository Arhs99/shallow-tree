from __future__ import annotations


from shallowtree.chem import RetroReaction, TemplatedRetroReaction
from shallowtree.configs.filter_configuration import FilterConfiguration
from shallowtree.context.filters.filter_strategy import FilterStrategy
from shallowtree.utils.exceptions import RejectionException


class ReactantsCountFilter(FilterStrategy):
    """
    Check that the number of reactants is was expected from the template

    :param key: the key or label
    :param config: the configuration of the tree search
    """

    def __init__(self, key: str, config: FilterConfiguration) -> None:
        super().__init__(key)
        self._logger.info(f"Loading reactants count filter to {key}")

    def apply(self, reaction: RetroReaction) -> None:
        if not isinstance(reaction, TemplatedRetroReaction):
            raise ValueError(
                "Reactants count filter can only be used on templated retro reaction "
            )

        reactants = reaction.reactants[reaction.index]
        if len(reactants) > reaction.rd_reaction.GetNumProductTemplates():
            raise RejectionException(
                f"{reaction} was filtered out because number of reactants disagree with the template"
            )
