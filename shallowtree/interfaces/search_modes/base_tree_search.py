import abc
import hashlib
from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple

from rdkit.Chem.rdchem import Mol
from shallowtree.configs.input_configuration import InputConfiguration

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.configs.application_configuration import ApplicationConfiguration
from shallowtree.configs.cache_configuration import CacheConfiguration
from shallowtree.configs.expansion_configuration import ExpansionConfiguration
from shallowtree.configs.filter_configuration import FilterConfiguration
from shallowtree.configs.stock_configuration import StockConfiguration
from shallowtree.context.cache.redis_cache import RedisCache
from shallowtree.context.config import Configuration
from shallowtree.context.expansion_strategies.template_based_expansion_strategy import TemplateBasedExpansionStrategy
from shallowtree.context.expansion_strategies.template_rules import TemplateRules
from shallowtree.context.filters.quick_keras_filter import QuickKerasFilter
from shallowtree.context.policy.expansion_policy import ExpansionPolicy
from shallowtree.context.policy.filter_policy import FilterPolicy
from shallowtree.context.stock.stock import Stock
# This must be imported first to setup logging for rdkit, tensorflow etc
from shallowtree.utils.logging import logger
from shallowtree.utils.lru import LRUCache


class BaseTreeSearch(abc.ABC):
    def __init__(self, input_config: InputConfiguration):
        config_dict = Configuration.from_json(input_config.app_configuration_path)
        app_config = ApplicationConfiguration(**config_dict)
        self._logger = logger()
        self._input_config = input_config
        self.app_config = app_config

        self._intern_cache: LRUCache = LRUCache(maxsize=2000)
        self.filter_policy = self._setup_filter_policy(app_config.filter)
        self.expansion_policy = self._setup_expansion_policy(app_config.expansion)

        self.stock = self._setup_stock(app_config.stock)

        self.redis_cache = self._setup_redis_cache(app_config.cache)

        self.rules_expansion = self._setup_rules_expansion(app_config)
        self.cache = dict()
        self.solved = dict()

    def req_search_tree(self, mol: TreeMolecule, depth: int, ancestors: frozenset = frozenset()) -> Tuple[float, bool]:
        # Returns (score, resolved). ``score`` is the soft recursive feasibility
        # average, kept as a ranking signal; ``resolved`` is True only when every
        # leaf of the chosen route is in stock. A route is committed to self.solved
        # and the search stops expanding this node ONLY when it is fully resolved —
        # a route that bottoms out on a non-stock leaf is not a real synthesis.
        # Because resolved implies score == 1.0 by induction, the first resolved
        # action is optimal, so we return on it.

        # Cycle guard: a molecule that reappears on its own retrosynthetic path
        # (e.g. an acid -> tert-butyl ester -> acid protect/deprotect loop) is a
        # dead end, not a solution. Score it 0.0/unresolved and DO NOT cache it —
        # this verdict is path-dependent while self.cache is keyed only by inchi_key.
        if mol.inchi_key in ancestors:
            return 0.0, False
        if depth > self._input_config.depth:
            return 0.0, False

        if mol.inchi_key in self.cache:
            cdepth, cscore, cresolved = self.cache[mol.inchi_key]
            if cdepth <= depth:
                return cscore, cresolved

        if mol in self.stock:
            self.cache[mol.inchi_key] = (0, 1.0, True)
            return 1.0, True

        # Check Redis cache if local cache miss
        if self.redis_cache and mol.inchi_key not in self.cache:
            dto = self.redis_cache.get_cache(mol.inchi_key)
            if dto.exists:
                self.cache[mol.inchi_key] = (dto.depth, dto.score, dto.resolved)  # Populate local cache
                if dto.depth <= depth:
                    # Only a resolved result has a persisted route to load
                    if dto.resolved:
                        solved_dto = self.redis_cache.get_solved(mol.inchi_key)
                        if solved_dto.exists:
                            self.solved[mol.inchi_key] = (solved_dto.reactants, solved_dto.score, solved_dto.classification)
                    return dto.score, dto.resolved

        feasible_actions = self._determine_feasible_actions(mol)

        # No action fully resolves this molecule: return the MAX soft score over
        # actions (the best achievable ranking signal), not the last action tried
        # — the latter is order-dependent and gives an arbitrary sub-threshold score.
        best_score = 0.0
        for action in feasible_actions:
            reactants = action.reactants[0]
            child_results = [self.req_search_tree(x, depth + 1, ancestors | {mol.inchi_key}) for x in reactants]
            score = sum(s for s, _ in child_results) / len(reactants)
            if all(resolved for _, resolved in child_results):
                self.solved[mol.inchi_key] = (reactants, score, action.metadata['classification'])
                self._update_cache(mol, depth, score, True)
                return score, True
            if score > best_score:
                best_score = score

        self._update_cache(mol, depth, best_score, False)
        return best_score, False

    @abstractmethod
    def search(self, *args, **kwargs) -> List:
        pass

    @abstractmethod
    def best_route(self, *args, **kwargs):
        pass


    def _determine_rules_and_actions(self, mol: TreeMolecule):
        expansion_policy, _ = self.expansion_policy.get_actions([mol])
        rules_expansion = self.rules_expansion.get_actions([mol])

        # Separate rules-based from ML-based actions, filtering out empty reactants
        actions = rules_expansion + expansion_policy
        rules_actions = []
        ml_actions = []
        for action in actions:
            if not action.reactants:
                continue
            if action.metadata['policy_name'] == 'rules':
                action.metadata["feasibility"] = 1.0
                rules_actions.append(action)
            else:
                ml_actions.append(action)
        return rules_actions, ml_actions

    def _apply_ml_actions(self, ml_actions: List):
        # Batch predict all ML actions at once
        ml_feasibilities = []
        if ml_actions:
            filter_name = next(iter(self.filter_policy.selection))
            ml_feasibilities = self.filter_policy[filter_name].batch_feasibility(ml_actions)
            for action, (_, prob) in zip(ml_actions, ml_feasibilities):
                action.metadata["feasibility"] = prob
        return ml_feasibilities

    def _collect_actions_above_threshold(self, ml_actions: List, rules_actions: List, ml_feasibilities: List):
        # Collect all feasible actions (rules + ML with prob >= 0.5)
        feasible_actions=rules_actions
        for action, (_, prob) in zip(ml_actions, ml_feasibilities):
            if prob >= 0.5:
                feasible_actions.append(action)
        return feasible_actions

    def _determine_feasible_actions(self, mol: TreeMolecule):
        rules_actions, ml_actions = self._determine_rules_and_actions(mol)
        ml_feasibilities = self._apply_ml_actions(ml_actions)
        feasible_actions = self._collect_actions_above_threshold(ml_actions, rules_actions, ml_feasibilities)

        return feasible_actions

    @abstractmethod
    def _update(self, *args, **kwargs) -> List:
        pass

    def _update_cache(self, mol: TreeMolecule, depth: int, score: float, resolved: bool):
        self.cache[mol.inchi_key] = (depth, score, resolved)
        if self.redis_cache:
            self.redis_cache.set_cache(mol.inchi_key, depth, score, resolved)

    def _load_from_redis(self, root_mol: TreeMolecule) -> None:
        """Pre-populate local caches from Redis for the root molecule subtree."""
        if self.redis_cache:
            dto = self.redis_cache.get_cache(root_mol.inchi_key)
            if dto.exists:
                self.cache[root_mol.inchi_key] = (dto.depth, dto.score, dto.resolved)
                # Only a resolved result has a persisted route to load
                if dto.resolved:
                    solved_dto = self.redis_cache.get_solved(root_mol.inchi_key)
                    if solved_dto.exists:
                        self.solved[root_mol.inchi_key] = (solved_dto.reactants, solved_dto.score, solved_dto.classification)
                        self._load_solved_subtree(solved_dto.reactants)

    def _load_solved_subtree(self, reactants: List[TreeMolecule]) -> None:
        """Recursively load solved entries for reactants."""
        for mol in reactants:
            if mol.inchi_key not in self.solved:
                solved_dto = self.redis_cache.get_solved(mol.inchi_key)
                if solved_dto.exists:
                    self.solved[mol.inchi_key] = (solved_dto.reactants, solved_dto.score, solved_dto.classification)
                    self._load_solved_subtree(solved_dto.reactants)

    def _save_to_redis(self, start_time) -> None:
        """Persist all solved routes to Redis."""
        if self.redis_cache:
            for inchi_key, (reactants, score, classification) in list(self.solved.items()):
                self.redis_cache.set_solved(inchi_key, reactants, score, classification, start_time)
                if inchi_key in self.cache:
                    depth, cache_score, cache_resolved = self.cache[inchi_key]
                    self.redis_cache.set_cache(inchi_key, depth, cache_score, cache_resolved)

    def _setup_redis_cache(self, cache_config: CacheConfiguration):
        if cache_config.enabled:
            scaffold = self._input_config.scaffold
            if scaffold:
                scaffold_hash = hashlib.sha256(scaffold.strip().encode()).hexdigest()[:16]
                cache_config.namespace = f"scaffold:{scaffold_hash}"
            redis_cache = RedisCache(
                filter_policy=self.filter_policy,
                expansion_policy=self.expansion_policy,
                stock=self.stock,
                cache_config=cache_config
            )
            return redis_cache
        else:
            return None

    def _setup_stock(self, stock_configs: List[StockConfiguration]) -> Stock:
        if self._input_config.prebuilt_stock is None:
            stock = Stock(stock_configs)
        else:
            stock = self._input_config.prebuilt_stock
            stock.select_first()
        return stock

    def _setup_expansion_policy(self, expansion_configs: List[ExpansionConfiguration]) -> ExpansionPolicy:
        strategies = []
        for expansion_config in expansion_configs:#TODO: logic below should be handled by a factory
            expansion_strategy = TemplateBasedExpansionStrategy(expansion_config.configuration_name, expansion_config, self._intern_cache)
            strategies.append(expansion_strategy)
        expansion_policy = ExpansionPolicy(strategies)
        return expansion_policy

    def _setup_filter_policy(self, filter_configs: List[FilterConfiguration]) -> FilterPolicy:
        strategies = []
        for filter_config in filter_configs:#TODO: logic below should be handled by a factory
            filter_strategy = QuickKerasFilter(filter_config.filter_name, filter_config)
            strategies.append(filter_strategy)
        filter_policy = FilterPolicy(strategies)
        return filter_policy

    def _setup_rules_expansion(self, app_config) -> TemplateRules :
        # Default rules file lives at shallowtree/rules/direct.csv. This module is
        # at shallowtree/interfaces/search_modes/, so go up three levels.
        extra_template_path = app_config.extra_template_path if app_config.extra_template_path \
            else Path(__file__).parents[2] / 'rules' / 'direct.csv'

        return TemplateRules(extra_template_path, self._intern_cache)