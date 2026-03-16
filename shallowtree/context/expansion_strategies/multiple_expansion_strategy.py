from __future__ import annotations

from typing import Any, List, Sequence, Optional, Tuple

import numpy as np

from shallowtree.chem import TreeMolecule, RetroReaction
from shallowtree.context.expansion_strategies.expansion_strategies import ExpansionStrategy
# from shallowtree.context.config import Configuration
from shallowtree.utils.type_utils import StrDict


class MultiExpansionStrategy(ExpansionStrategy):
    """
    A base class for combining multiple expansion strategies.

    The strategy can be used by either calling the `get_actions` method
    or by calling the instantiated class with a list of molecules.

    :ivar expansion_strategy_keys: the keys of the selected expansion strategies
    :ivar additive_expansion: a conditional setting to specify whether all the actions
        and priors of the selected expansion strategies should be combined or not.
        Defaults to False.
    :ivar expansion_strategy_weights: a list of weights for each expansion strategy.
        The weights should sum to one. Exception is the default, where unity weight
        is associated to each strategy.

    :param key: the key or label
    :param config: the configuration of the tree search
    :param expansion_strategies: the keys of the selected expansion strategies. All keys
        of the selected expansion strategies must exist in the expansion policies listed
        in config
    """

    _required_kwargs = ["expansion_strategies"]

    def __init__(
        self,
        key: str,
        config: "Configuration",
        **kwargs: Any,
    ):
        super().__init__(key, config, **kwargs)
        self._config = config
        self._expansion_strategies: List[ExpansionStrategy] = []
        self.expansion_strategy_keys = kwargs["expansion_strategies"]

        self.cutoff_number = kwargs.get("cutoff_number")
        if self.cutoff_number:
            print(f"Setting multi-expansion cutoff_number: {self.cutoff_number}")

        self.expansion_strategy_weights = self._set_expansion_strategy_weights(kwargs)
        self.additive_expansion: bool = bool(kwargs.get("additive_expansion", False))
        self._logger.info(
            f"Multi-expansion strategy with policies: {self.expansion_strategy_keys}"
            f", and corresponding weights: {self.expansion_strategy_weights}"
        )

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules, using the selected policies.

        The default implementation combines all the actions and priors of the
        selected expansion strategies into two lists respectively if the
        'additive_expansion' setting is set to True. This function can be overridden by
        a sub class to combine different expansion strategies in different ways.

        :param molecules: the molecules to consider
        :param cache_molecules: additional molecules to submit to the expansion
            policy but that only will be cached for later use
        :return: the actions and the priors of those actions
        :raises: PolicyException: if the policy isn't selected
        """
        expansion_strategies = self._get_expansion_strategies_from_config()

        all_possible_actions = []
        all_priors = []
        for expansion_strategy, expansion_strategy_weight in zip(
            expansion_strategies, self.expansion_strategy_weights
        ):
            possible_actions, priors = expansion_strategy.get_actions(
                molecules, cache_molecules
            )

            all_possible_actions.extend(possible_actions)
            if not self.additive_expansion and all_possible_actions:
                all_priors.extend(priors)
                break

            weighted_prior = [expansion_strategy_weight * p for p in priors]

            all_priors.extend(weighted_prior)

        all_possible_actions, all_priors = self._prune_actions(
            all_possible_actions, all_priors
        )
        return all_possible_actions, all_priors

    def _get_expansion_strategies_from_config(self) -> List[ExpansionStrategy]:
        if self._expansion_strategies:
            return self._expansion_strategies

        if not all(
            key in self._config.expansion_policy.items
            for key in self.expansion_strategy_keys
        ):
            raise ValueError(
                "The input expansion strategy keys must exist in the "
                "expansion policies listed in config"
            )
        self._expansion_strategies = [
            self._config.expansion_policy[key] for key in self.expansion_strategy_keys
        ]

        for expansion_strategy, weight in zip(self._expansion_strategies, self.expansion_strategy_weights):
            if not getattr(expansion_strategy, "rescale_prior", True) and weight < 1:
                setattr(expansion_strategy, "rescale_prior", True)
                self._logger.info(f"Enforcing {expansion_strategy.key}.rescale_prior=True")
        return self._expansion_strategies

    def _prune_actions(
        self, actions: List[RetroReaction], priors: List[float]
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Prune the actions if a maximum number of actions is specified.

        :param actions: list of predicted actions
        :param priors: list of prediction probabilities
        :return: the top 'self.cutoff_number' actions and corresponding priors.
        """
        if not self.cutoff_number:
            return actions, priors

        sortidx = np.argsort(np.array(priors))[::-1].astype(int)
        priors = [priors[idx] for idx in sortidx[0 : self.cutoff_number]]
        actions = [actions[idx] for idx in sortidx[0 : self.cutoff_number]]
        return actions, priors

    def _set_expansion_strategy_weights(self, kwargs: StrDict) -> List[float]:
        """
        Set the weights of each expansion strategy using the input kwargs from config.
        The weights in the config should sum to one.
        If not set in the config file, the weights default to one for each strategy
        (for backwards compatibility).

        :param kwargs: input arguments to the MultiExpansionStrategy
        :raises: ValueError if weights from the config file do not sum to one.
        :return: a list of expansion strategy weights
        """
        if not "expansion_strategy_weights" in kwargs:
            return [1.0 for _ in self.expansion_strategy_keys]

        expansion_strategy_weights = kwargs["expansion_strategy_weights"]
        sum_weights = sum(expansion_strategy_weights)

        if sum_weights != 1:
            raise ValueError(
                "The expansion strategy weights in MultiExpansion should "
                "sum to one. -> "
                f"sum({expansion_strategy_weights})={sum_weights}."
            )

        return expansion_strategy_weights
