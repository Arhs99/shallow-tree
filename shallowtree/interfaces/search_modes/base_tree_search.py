import abc
from abc import abstractmethod
from pathlib import Path
from typing import List

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

        self.stock = self._setup_stock(app_config.stock) if self._input_config.prebuilt_stock is None \
            else self._input_config.prebuilt_stock

        self.redis_cache = self._setup_redis_cache(app_config.cache)

        self.rules_expansion = self._setup_rules_expansion(app_config)
        self.max_depth = 2
        self.cache = dict()
        self.solved = dict()
        self.BBs = []

    def req_search_tree(self, mol: TreeMolecule, depth: int) -> float:
        if depth > self.max_depth:
            return 0.0

        if mol.inchi_key in self.cache:
            cdepth, cscore = self.cache[mol.inchi_key]
            if cdepth <= depth:
                return cscore

        if mol in self.stock:
            self.cache[mol.inchi_key] = (0, 1.0)
            return 1.0

        # Check Redis cache if local cache miss
        if self.redis_cache and mol.inchi_key not in self.cache:
            redis_cache_data = self.redis_cache.get_cache(mol.inchi_key)
            if redis_cache_data:
                cdepth, cscore = redis_cache_data
                self.cache[mol.inchi_key] = (cdepth, cscore)  # Populate local cache
                if cdepth <= depth:
                    # Also try to load solved data
                    solved_data = self.redis_cache.get_solved(mol.inchi_key)
                    if solved_data:
                        self.solved[mol.inchi_key] = solved_data
                    return cscore

        feasible_actions = self._determine_feasible_actions(mol)

        score = 0.0
        for action in feasible_actions:
            reactants = action.reactants
            score = sum([self.req_search_tree(x, depth + 1) for x in reactants[0]]) / len(reactants[0])
            if score > self.app_config.search.score_acceptance_threshold:
                self.solved[mol.inchi_key] = (reactants[0], score, action.metadata['classification'])
                self._update_cache(mol, depth, score)
                return score

        self._update_cache(mol, depth, score)
        return score

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

    def _update_cache(self, mol: TreeMolecule, depth: int, score: float):
        self.cache[mol.inchi_key] = (depth, score)
        if self.redis_cache:
            self.redis_cache.set_cache(mol.inchi_key, depth, score)

    def _load_from_redis(self, root_mol: TreeMolecule) -> None:
        """Pre-populate local caches from Redis for the root molecule subtree."""
        if self.redis_cache:
            cache_data = self.redis_cache.get_cache(root_mol.inchi_key)
            if cache_data:
                self.cache[root_mol.inchi_key] = cache_data
                solved_data = self.redis_cache.get_solved(root_mol.inchi_key)
                if solved_data:
                    self.solved[root_mol.inchi_key] = solved_data
                    self._load_solved_subtree(solved_data[0])

    def _load_solved_subtree(self, reactants: List[TreeMolecule]) -> None:
        """Recursively load solved entries for reactants."""
        for mol in reactants:
            if mol.inchi_key not in self.solved:
                solved_data = self.redis_cache.get_solved(mol.inchi_key)
                if solved_data:
                    self.solved[mol.inchi_key] = solved_data
                    self._load_solved_subtree(solved_data[0])

    def _save_to_redis(self) -> None:
        """Persist all solved routes to Redis."""
        if self.redis_cache:
            for inchi_key, (reactants, score, classification) in list(self.solved.items()):
                self.redis_cache.set_solved(inchi_key, reactants, score, classification)
                if inchi_key in self.cache:
                    depth, cache_score = self.cache[inchi_key]
                    self.redis_cache.set_cache(inchi_key, depth, cache_score)

    def _setup_redis_cache(self, cache_config: CacheConfiguration):
        if cache_config.enabled:
            redis_cache = RedisCache(
                host=cache_config.host,
                port=cache_config.port,
                db=cache_config.db,
                password=cache_config.password,
                socket_timeout=cache_config.socket_timeout,
                filter_policy=self.filter_policy,
                expansion_policy=self.expansion_policy,
                stock=self.stock
            )
            return redis_cache
        else:
            return None

    def _setup_stock(self, stock_configs: List[StockConfiguration]):
        stock = Stock()
        for stock_config in stock_configs:
            stock.load_stocks(stock_config)
        return stock

    def _setup_expansion_policy(self, expansion_configs: List[ExpansionConfiguration]):
        expansion_policy = ExpansionPolicy()
        for expansion_config in expansion_configs:
            expansion_strategy = TemplateBasedExpansionStrategy(expansion_config.configuration_name, expansion_config,
                                                                self._intern_cache)
            expansion_policy.load(expansion_strategy)
        return expansion_policy

    def _setup_filter_policy(self, filter_configs: List[FilterConfiguration]):
        filter_policy = FilterPolicy()
        for filter_config in filter_configs:
            filter_strategy = QuickKerasFilter(filter_config.filter_name, filter_config)
            filter_policy.load(filter_strategy)
        return filter_policy

    def _setup_rules_expansion(self, app_config) -> TemplateRules :
        extra_template_path = app_config.extra_template_path if app_config.extra_template_path \
            else Path(__file__).parent.parent / 'rules' / 'direct.csv'

        return TemplateRules(extra_template_path, self._intern_cache)