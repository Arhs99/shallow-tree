"""Shared fixtures for the search_modes tests (StandardSearch / ScaffoldSearch).

Not collected by the test runner (no ``test`` prefix); imported by the
per-class test modules.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from unittest.mock import MagicMock, patch

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.configs.input_configuration import InputConfiguration
from shallowtree.interfaces.search_modes.base_tree_search import BaseTreeSearch
from shallowtree.interfaces.search_modes.standard_search import StandardSearch

_BASE = "shallowtree.interfaces.search_modes.base_tree_search"


def _make_search(cls=StandardSearch, scaffold=None, **overrides):
    """Create a search instance (StandardSearch by default) with all heavy
    setup mocked out. The config-file load and the five _setup_* hooks are
    patched only during construction; the instance keeps the mock attributes."""
    with patch(f"{_BASE}.Configuration.from_json", return_value={}), \
         patch(f"{_BASE}.ApplicationConfiguration") as mock_app_cls, \
         patch.object(BaseTreeSearch, "_setup_filter_policy") as mock_fp, \
         patch.object(BaseTreeSearch, "_setup_expansion_policy") as mock_ep, \
         patch.object(BaseTreeSearch, "_setup_stock") as mock_st, \
         patch.object(BaseTreeSearch, "_setup_redis_cache") as mock_rc, \
         patch.object(BaseTreeSearch, "_setup_rules_expansion") as mock_re:

        mock_app = MagicMock()
        mock_app.search.score_acceptance_threshold = 0.9
        mock_app.search.time_limit = 10**9
        mock_app_cls.return_value = mock_app

        mock_stock = MagicMock()
        mock_stock.__contains__ = MagicMock(return_value=False)

        mock_expansion = MagicMock()
        mock_expansion.get_actions = MagicMock(return_value=([], []))

        mock_rules = MagicMock()
        mock_rules.get_actions = MagicMock(return_value=[])

        mock_filter = MagicMock()

        mock_fp.return_value = mock_filter
        mock_ep.return_value = mock_expansion
        mock_st.return_value = mock_stock
        mock_rc.return_value = None
        mock_re.return_value = mock_rules

        input_config = InputConfiguration(
            app_configuration_path="dummy.json", output_path="", scaffold=scaffold
        )
        search = cls(input_config)

        # Apply overrides
        for k, v in overrides.items():
            setattr(search, k, v)

        return search


def _make_action(reactant_smiles_list, classification="test", policy_name="ml",
                 feasibility=1.0):
    """Create a mock action with real TreeMolecule reactants."""
    parent = TreeMolecule(parent=None, smiles="CCO")
    reactants = tuple(
        TreeMolecule(parent=parent, smiles=smi) for smi in reactant_smiles_list
    )
    action = MagicMock()
    action.reactants = (reactants,)
    action.metadata = {
        "classification": classification,
        "policy_name": policy_name,
        "feasibility": feasibility,
    }
    return action
