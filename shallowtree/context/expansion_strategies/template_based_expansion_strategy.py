from __future__ import annotations

from typing import Optional, Dict, Tuple, Sequence, List

import numpy as np
import pandas as pd

from shallowtree.chem import TreeMolecule, RetroReaction, TemplatedRetroReaction
from shallowtree.configs.expansion_configuration import ExpansionConfiguration
from shallowtree.context.expansion_strategies.expansion_strategies import ExpansionStrategy
from shallowtree.context.policy.utils import _make_fingerprint
from shallowtree.utils.exceptions import PolicyException
from shallowtree.utils.models import load_model


class TemplateBasedExpansionStrategy(ExpansionStrategy):
    """
    A template-based expansion strategy that will return `TemplatedRetroReaction` objects upon expansion.

    :ivar template_column: the column in the template file that contains the templates
    :ivar cutoff_cumulative: the accumulative probability of the suggested templates
    :ivar cutoff_number: the maximum number of templates to returned
    :ivar use_rdchiral: a boolean to apply templates with RDChiral
    :ivar use_remote_models: a boolean to connect to remote TensorFlow servers
    :ivar rescale_prior: a boolean to apply softmax to the priors
    :ivar chiral_fingerprints: if True will base expansion on chiral fingerprint
    :ivar mask: a boolean vector of masks for the reaction templates. The length of the vector should be equal to the
        number of templates. It is set to None if no mask file is provided as input.

    :param key: the key or label
    :param config: the configuration of the tree search
    :param model: the source of the policy model
    :param template: the path to a HDF5 file with the templates
    :raises PolicyException: if the length of the model output vector is not same as the
        number of templates
    """

    def __init__(self, key: str, config: ExpansionConfiguration) -> None:
        super().__init__(key)

        self.template_column: str = config.template_column
        self.cutoff_cumulative: float = config.cutoff_cumulative
        self.cutoff_number: int = config.cutoff_number
        self.use_rdchiral: bool = config.use_rdchiral
        self.use_remote_models: bool = config.use_remote_models
        self.rescale_prior: bool = config.rescale_prior
        self.chiral_fingerprints = config.chiral_fingerprints

        self._logger.info(f"Loading template-based expansion policy model from {config.model} to {self.key}")
        self.model = load_model(config.model, self.key, self.use_remote_models)

        self._logger.info(f"Loading templates from {config.template} to {self.key}")
        if config.template.endswith(".csv.gz") or config.template.endswith(".csv"):
            self.templates: pd.DataFrame = pd.read_csv(config.template, index_col=0, sep="\t")
        else:
            self.templates = pd.read_hdf(config.template, "table")

        self.mask: Optional[np.ndarray] = (self._load_mask_file(config.mask) if config.mask else None)

        if hasattr(self.model, "output_size") and len(self.templates) != self.model.output_size:  # type: ignore
            raise PolicyException(
                f"The number of templates ({len(self.templates)}) does not agree with the "  # type: ignore
                f"output dimensions of the model ({self.model.output_size})"
            )
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

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
        priors: List[float] = []
        cache_molecules = cache_molecules or []
        self._update_cache(list(molecules) + list(cache_molecules))

        for mol in molecules:
            probable_transforms_idx, probs = self._cache[mol.inchi_key]
            possible_moves = self.templates.iloc[probable_transforms_idx]
            if self.rescale_prior:
                probs /= probs.sum()
            priors.extend(probs)
            for idx, (move_index, move) in enumerate(possible_moves.iterrows()):
                metadata = dict(move)
                del metadata[self.template_column]
                metadata["policy_probability"] = float(probs[idx].round(4))
                metadata["policy_probability_rank"] = idx
                metadata["policy_name"] = self.key
                metadata["template_code"] = move_index
                metadata["template"] = move[self.template_column]
                possible_actions.append(
                    TemplatedRetroReaction(
                        mol,
                        smarts=move[self.template_column],
                        metadata=metadata,
                        use_rdchiral=self.use_rdchiral,
                    )
                )
        return possible_actions, priors  # type: ignore

    def reset_cache(self) -> None:
        """Reset the prediction cache"""
        self._cache = {}

    def _cutoff_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Get the top transformations, by selecting those that have:
            * cumulative probability less than a threshold (cutoff_cumulative)
            * or at most N (cutoff_number)
        """
        if self.mask is not None:
            predictions[~self.mask] = 0
        sortidx = np.argsort(predictions)[::-1]
        cumsum: np.ndarray = np.cumsum(predictions[sortidx])
        if any(cumsum >= self.cutoff_cumulative):
            maxidx = int(np.argmin(cumsum < self.cutoff_cumulative))
        else:
            maxidx = len(cumsum)
        maxidx = min(maxidx, self.cutoff_number) or 1
        return sortidx[:maxidx]

    def _load_mask_file(self, maskfile: str) -> np.ndarray:
        self._logger.info(f"Loading masking of templates from {maskfile} to {self.key}")
        mask = np.load(maskfile)["arr_0"]
        if len(mask) != len(self.templates):
            raise PolicyException(
                f"The number of masks {len(mask)} does not match the number of templates {len(self.templates)}"
            )
        return mask

    def _update_cache(self, molecules: Sequence[TreeMolecule]) -> None:
        pred_inchis = []
        fp_list = []
        for molecule in molecules:
            if molecule.inchi_key in self._cache or molecule.inchi_key in pred_inchis:
                continue
            fp_list.append(_make_fingerprint(molecule, self.model, self.chiral_fingerprints))
            pred_inchis.append(molecule.inchi_key)

        if not pred_inchis:
            return

        pred_list = np.asarray(self.model.predict(np.vstack(fp_list)))
        for pred, inchi in zip(pred_list, pred_inchis):
            probable_transforms_idx = self._cutoff_predictions(pred)
            self._cache[inchi] = (
                probable_transforms_idx,
                pred[probable_transforms_idx],
            )
