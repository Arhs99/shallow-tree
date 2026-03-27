from __future__ import annotations

from typing import List, Any, Tuple

import numpy as np

from shallowtree.chem import RetroReaction
from shallowtree.configs.filter_configuration import FilterConfiguration
from shallowtree.context.filters.filter_strategy import FilterStrategy
from shallowtree.context.policy.utils import _make_fingerprint
from shallowtree.utils.exceptions import RejectionException
from shallowtree.utils.models import load_model


class QuickKerasFilter(FilterStrategy):
    """
    Filter quick-filter trained on artificial negative data

    :ivar use_remote_models: a boolean to connect to remote TensorFlow servers. Defaults
        to False.
    :ivar filter_cutoff: the cut-off value

    :param key: the key or label
    :param config: the configuration of the tree search
    :param model: the source of the policy model
    """

    _required_kwargs: List[str] = ["model"]

    def __init__(self, key: str, config: FilterConfiguration) -> None:
        super().__init__(key)
        source = config.model
        self._logger.info(f"Loading filter policy model from {source} to {key}")
        self.use_remote_models: bool = config.use_remote_models
        self.model = load_model(source, key, self.use_remote_models)
        self._prod_fp_name = config.prod_fp_name
        self._rxn_fp_name = config.rxn_fp_name
        self._exclude_from_policy: List[str] = config.exclude_from_policy
        self.filter_cutoff: float = config.filter_cutoff

    def apply(self, reaction: RetroReaction) -> None:
        if reaction.metadata.get("policy_name", "") in self._exclude_from_policy:
            return

        feasible, prob = self.feasibility(reaction)
        if not feasible:
            raise RejectionException(f"{reaction} was filtered out with prob {prob}")

    def feasibility(self, reaction: RetroReaction) -> Tuple[bool, float]:
        """
        Computes if a given reaction is feasible by given
        the reaction fingerprint to a network model

        :param reaction: the reaction to query
        :return: if the reaction is feasible
        """
        if not reaction.reactants:
            return False, 0.0

        prob = self._predict(reaction)
        feasible = prob >= self.filter_cutoff
        return feasible, prob

    def _predict(self, reaction: RetroReaction) -> float:
        prod_fp, rxn_fp = self._reaction_to_fingerprint(reaction, self.model)
        kwargs = {self._prod_fp_name: prod_fp, self._rxn_fp_name: rxn_fp}
        return self.model.predict(prod_fp, rxn_fp, **kwargs)[0][0]

    def batch_feasibility(
        self, reactions: List["RetroReaction"]
    ) -> List[Tuple[bool, float]]:
        """
        Batch prediction for multiple reactions.

        :param reactions: list of reactions to evaluate
        :return: list of (feasible, probability) tuples in same order as input
        """
        if not reactions:
            return []

        # Separate reactions with/without reactants and compute fingerprints
        valid_indices = []
        prod_fps = []
        rxn_fps = []

        for i, reaction in enumerate(reactions):
            if not reaction.reactants:
                continue
            valid_indices.append(i)
            prod_fp, rxn_fp = self._reaction_to_fingerprint(reaction, self.model)
            prod_fps.append(prod_fp)
            rxn_fps.append(rxn_fp)

        # Initialize results with (False, 0.0) for all reactions
        results: List[Tuple[bool, float]] = [(False, 0.0)] * len(reactions)

        # Handle case where no reactions have valid reactants
        if not valid_indices:
            return results

        # Stack fingerprints into batches
        prod_batch = np.vstack(prod_fps)
        rxn_batch = np.vstack(rxn_fps)

        # Single batched prediction
        kwargs = {self._prod_fp_name: prod_batch, self._rxn_fp_name: rxn_batch}
        probs = self.model.predict(prod_batch, rxn_batch, **kwargs)

        # Handle different output shapes
        if probs.ndim > 1:
            probs = probs[:, 0]

        # Populate results for valid reactions
        for idx, prob in zip(valid_indices, probs):
            prob_float = float(prob)
            results[idx] = (prob_float >= self.filter_cutoff, prob_float)

        return results

    @staticmethod
    def _reaction_to_fingerprint(
        reaction: RetroReaction, model: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        rxn_fp = _make_fingerprint(reaction, model)
        prod_fp = _make_fingerprint(reaction.mol, model)
        return prod_fp, rxn_fp
