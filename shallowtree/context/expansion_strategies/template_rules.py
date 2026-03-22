from __future__ import annotations

from typing import Sequence, Optional, List

import pandas as pd

from shallowtree.chem import TreeMolecule, RetroReaction, TemplatedRetroReaction
from shallowtree.context.expansion_strategies.expansion_strategies import ExpansionStrategy


class TemplateRules(ExpansionStrategy):
    def __init__(self, rules_csv: str):
        super().__init__(None, None)
        self.templates: pd.DataFrame = pd.read_csv(rules_csv, index_col=0, sep="\t")
        self.template_column = "retro_template"
        self.use_rdchiral = False

    def get_actions(
            self, molecules: Sequence[TreeMolecule], cache_molecules: Optional[Sequence[TreeMolecule]] = None,
    ) -> List[RetroReaction]:
        possible_actions = []

        for mol in molecules:
            for idx, (move_index, move) in enumerate(self.templates.iterrows()):
                metadata = dict(move)
                del metadata[self.template_column]
                metadata["policy_probability"] = 1.0
                metadata["policy_probability_rank"] = idx
                metadata["policy_name"] = 'rules'
                metadata["template_code"] = move_index
                metadata["template"] = move[self.template_column]
                template = TemplatedRetroReaction(mol, smarts=move[self.template_column], metadata=metadata,
                                                  use_rdchiral=self.use_rdchiral)
                possible_actions.append(template)
        return possible_actions
