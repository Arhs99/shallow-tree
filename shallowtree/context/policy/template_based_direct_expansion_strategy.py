from __future__ import annotations

from typing import Sequence, Optional, Tuple, List

from shallowtree.chem import TreeMolecule, RetroReaction, SmilesBasedRetroReaction
from shallowtree.context.policy.template_based_expansion_strategy import TemplateBasedExpansionStrategy


class TemplateBasedDirectExpansionStrategy(TemplateBasedExpansionStrategy):
    """
    A template-based expansion strategy that will return `SmilesBasedRetroReaction` objects upon expansion
    by directly applying the template

    :param key: the key or label
    :param config: the configuration of the tree search
    :param source: the source of the policy model
    :param templatefile: the path to a HDF5 file with the templates
    :raises PolicyException: if the length of the model output vector is not same as the number of templates
    """

    def get_actions(
        self,
        molecules: Sequence[TreeMolecule],
        cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> Tuple[List[RetroReaction], List[float]]:
        """
        Get all the probable actions of a set of molecules, using the selected policies and given cutoffs

        :param molecules: the molecules to consider
        :param cache_molecules: additional molecules to submit to the expansion
            policy but that only will be cached for later use
        :return: the actions and the priors of those actions
        """
        possible_actions = []
        priors = []

        super_actions, super_priors = super().get_actions(molecules, cache_molecules)
        for templated_action, prior in zip(super_actions, super_priors):
            for reactants in templated_action.reactants:
                reactants_str = ".".join(mol.smiles for mol in reactants)
                new_action = SmilesBasedRetroReaction(
                    templated_action.mol,
                    metadata=templated_action.metadata,
                    reactants_str=reactants_str,
                )
                possible_actions.append(new_action)
                priors.append(prior)

        return possible_actions, priors  # type: ignore
