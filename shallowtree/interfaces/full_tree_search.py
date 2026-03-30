# shalow-tree - Retrosynthetic analysis and scoring
# Copyright (C) 2025  Kostas Papadopoulos <kostasp97@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd
from rdkit import Chem

from shallowtree.chem import Molecule, TreeMolecule
from shallowtree.configs.application_configuration import ApplicationConfiguration
from shallowtree.configs.cache_configuration import CacheConfiguration
from shallowtree.configs.expansion_configuration import ExpansionConfiguration
from shallowtree.configs.filter_configuration import FilterConfiguration
from shallowtree.configs.stock_configuration import StockConfiguration
from shallowtree.context.cache.redis_cache import RedisCache
from shallowtree.context.expansion_strategies.template_based_expansion_strategy import TemplateBasedExpansionStrategy
from shallowtree.context.expansion_strategies.template_rules import TemplateRules
from shallowtree.context.filters.quick_keras_filter import QuickKerasFilter
from shallowtree.context.policy.expansion_policy import ExpansionPolicy
from shallowtree.context.policy.filter_policy import FilterPolicy
from shallowtree.context.stock.stock import Stock
from shallowtree.tools.profile_search import timer
# This must be imported first to setup logging for rdkit, tensorflow etc
from shallowtree.utils.logging import logger


class Expander:

    def __init__(self, app_config: ApplicationConfiguration):
        self._logger = logger()

        self.filter_policy = self._setup_filter_policy(app_config.filter)
        self.expansion_policy = self._setup_expansion_policy(app_config.expansion)
        self.stock = self._setup_stock(app_config.stock)
        self.redis_cache = self._setup_redis_cache(app_config.cache)

        self.rules_expansion = self._setup_rules_expansion(app_config)
        self.max_depth = 2
        self.cache = dict()
        self.solved = dict()
        self._profiling = False
        self._timers = {}
        self.BBs = []
        self._counter = 0
        self._cache_counter = 0

    def context_search(self, smiles: List[str], scaffold_str: str, max_depth=2) -> pd.DataFrame:
        self.max_depth = max_depth
        rows = []
        scaffold = Chem.MolFromSmarts(scaffold_str)

        for smi in smiles:
            solution = defaultdict(list)
            mol = TreeMolecule(parent=None, smiles=smi)
            self.BBs = []
            self._counter = 0
            self._cache_counter = 0

            # Pre-populate from Redis if available
            self._load_from_redis(mol)

            score = 0.0
            actions, _ = self.expansion_policy.get_actions([mol])
            det_actions = self.rules_expansion.get_actions([mol])

            # Separate rules-based from ML-based actions, filtering out empty reactants
            rules_actions = []
            ml_actions = []
            for action in det_actions + actions:
                if not action.reactants:
                    continue
                if action.metadata['policy_name'] == 'rules':
                    action.metadata["feasibility"] = 1.0
                    rules_actions.append(action)
                else:
                    ml_actions.append(action)

            # Batch predict all ML actions at once
            ml_feasibilities = []
            if ml_actions:
                filter_name = next(iter(self.filter_policy.selection))
                ml_feasibilities = self.filter_policy[filter_name].batch_feasibility(ml_actions)
                for action, (_, prob) in zip(ml_actions, ml_feasibilities):
                    action.metadata["feasibility"] = prob

            # Collect all feasible actions (rules + ML with prob >= 0.5)
            feasible_actions = rules_actions + [
                action for action, (_, prob) in zip(ml_actions, ml_feasibilities)
                if prob >= 0.5
            ]

            root_match = set(mol.index_to_mapping[x] for x in mol.rd_mol.GetSubstructMatch(scaffold))
            for action in feasible_actions:
                reactants = action.reactants
                for r in reactants[0]:
                    r_match = set(r.index_to_mapping[x] for x in r.rd_mol.GetSubstructMatch(scaffold))
                    if r_match and len(r_match ^ root_match) == 2:
                        score = sum([self.req_search_tree(x, 1) for x in reactants[0] if x != r]) / (len(
                            reactants[0]) - 1)
                        if score > 0.9:
                            self.solved[mol.inchi_key] = (reactants[0], score, action.metadata['classification'])
                        break
            # Persist to Redis if available and successful
            if self.redis_cache and score > 0.9:
                self._save_to_redis()

            if score > 0.9:
                self.best_route(mol, 0, solution)
                # json_data = json.dumps(dict(solution), indent=2)
            rows.append({'SMILES': smi, 'score': score, 'route': dict(solution), 'BBs': self.BBs})
        df = pd.DataFrame(rows)
        return df

    def search_tree(self, smiles: List[str], max_depth=2) -> pd.DataFrame:
        """
        """
        self.max_depth = max_depth
        rows = []
        for smi in smiles:
            solution = defaultdict(list)
            mol = TreeMolecule(parent=None, smiles=smi)
            self.BBs = []
            self._counter = 0
            self._cache_counter = 0

            # Pre-populate from Redis if available
            if self.redis_cache:
                self._load_from_redis(mol)

            score = self.req_search_tree(mol, depth=0)

            if self._profiling:
                self._timers["_counter_total"] = self._timers.get("_counter_total", 0) + self._counter
                self._timers["_cache_counter_total"] = self._timers.get("_cache_counter_total", 0) + self._cache_counter

            # Persist to Redis if available and successful
            if self.redis_cache and score > 0.9:
                self._save_to_redis()

            if score > 0.9:
                self.best_route(mol, 0, solution)
                # json_data = json.dumps(dict(solution), indent=2)
            rows.append({'SMILES': smi, 'score': score, 'route': dict(solution), 'BBs': self.BBs})
        df = pd.DataFrame(rows)
        return df

    def req_search_tree(self, mol: TreeMolecule, depth: int) -> float:
        if depth > self.max_depth:
            return 0.0
        self._counter += 1
        _p = self._profiling

        if _p:
            with timer(self._timers, "cache_lookup"):
                cache_hit = mol.inchi_key in self.cache
        else:
            cache_hit = mol.inchi_key in self.cache
        if cache_hit:
            self._cache_counter += 1
            cdepth, cscore = self.cache[mol.inchi_key]
            if cdepth <= depth:
                return cscore

        if _p:
            with timer(self._timers, "stock_check"):
                in_stock = mol in self.stock
        else:
            in_stock = mol in self.stock
        if in_stock:
            self.cache[mol.inchi_key] = (0, 1.0)
            return 1.0

        # Check Redis cache if local cache miss
        if self.redis_cache and mol.inchi_key not in self.cache:
            redis_cache_data = self.redis_cache.get_cache(mol.inchi_key)
            if redis_cache_data:
                cdepth, cscore = redis_cache_data
                self.cache[mol.inchi_key] = (cdepth, cscore)  # Populate local cache
                if cdepth <= depth:
                    self._cache_counter += 1
                    # Also try to load solved data
                    solved_data = self.redis_cache.get_solved(mol.inchi_key)
                    if solved_data:
                        self.solved[mol.inchi_key] = solved_data
                    return cscore

        if _p:
            with timer(self._timers, "expansion_policy"):
                actions, _ = self.expansion_policy.get_actions([mol])
                det_actions = self.rules_expansion.get_actions([mol])
        else:
            actions, _ = self.expansion_policy.get_actions([mol])
            det_actions = self.rules_expansion.get_actions([mol])

        # Separate rules-based from ML-based actions, filtering out empty reactants
        rules_actions = []
        ml_actions = []
        for action in actions + det_actions:
            if not action.reactants:
                continue
            if action.metadata['policy_name'] == 'rules':
                action.metadata["feasibility"] = 1.0
                rules_actions.append(action)
            else:
                ml_actions.append(action)

        # Batch predict all ML actions at once
        if _p:
            with timer(self._timers, "filter_batch"):
                ml_feasibilities = []
                if ml_actions:
                    filter_name = next(iter(self.filter_policy.selection))
                    ml_feasibilities = self.filter_policy[filter_name].batch_feasibility(ml_actions)
                    for action, (_, prob) in zip(ml_actions, ml_feasibilities):
                        action.metadata["feasibility"] = prob
        else:
            ml_feasibilities = []
            if ml_actions:
                filter_name = next(iter(self.filter_policy.selection))
                ml_feasibilities = self.filter_policy[filter_name].batch_feasibility(ml_actions)
                for action, (_, prob) in zip(ml_actions, ml_feasibilities):
                    action.metadata["feasibility"] = prob

        # Collect all feasible actions (rules + ML with prob >= 0.5)
        feasible_actions = rules_actions + [
            action for action, (_, prob) in zip(ml_actions, ml_feasibilities)
            if prob >= 0.5
        ]

        score = 0.0
        if _p:
            with timer(self._timers, "recursive_dfs"):
                for action in feasible_actions:
                    reactants = action.reactants
                    score = sum([self.req_search_tree(x, depth + 1) for x in reactants[0]]) / len(reactants[0])
                    if score > 0.9:
                        self.solved[mol.inchi_key] = (reactants[0], score, action.metadata['classification'])
                        self.cache[mol.inchi_key] = (depth, score)
                        if self.redis_cache:
                            self.redis_cache.set_cache(mol.inchi_key, depth, score)
                        return score
        else:
            for action in feasible_actions:
                reactants = action.reactants
                score = sum([self.req_search_tree(x, depth + 1) for x in reactants[0]]) / len(reactants[0])
                if score > 0.9:
                    self.solved[mol.inchi_key] = (reactants[0], score, action.metadata['classification'])
                    self.cache[mol.inchi_key] = (depth, score)
                    if self.redis_cache:
                        self.redis_cache.set_cache(mol.inchi_key, depth, score)
                    return score
        self.cache[mol.inchi_key] = (depth, score)
        if self.redis_cache:
            self.redis_cache.set_cache(mol.inchi_key, depth, score)
        return score

    def best_route(self, mol: Molecule, depth: int, tree: defaultdict):
        while depth <= self.max_depth:
            tup = self.solved.get(mol.inchi_key)
            if tup is None:
                self.BBs.append(mol.smiles)
                return
            else:
                rxn, score, clas = tup
                reactants = '.'.join([m.smiles for m in rxn])
                tree[depth + 1].append([f'{mol.smiles} => {reactants}', clas])
                for x in rxn:
                    self.best_route(x, depth + 1, tree)
                return

    def _load_from_redis(self, root_mol: TreeMolecule) -> None:
        """Pre-populate local caches from Redis for the root molecule subtree."""
        if not self.redis_cache:
            return

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
            for inchi_key, (reactants, score, classification) in self.solved.items():
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

    def _setup_stock(self, config_dict: StockConfiguration):
        stock = Stock()
        stock.load_from_config(config_dict)
        return stock

    def _setup_expansion_policy(self, expansion_config: ExpansionConfiguration):
        expansion_strategy = TemplateBasedExpansionStrategy(expansion_config.configuration_name, expansion_config)
        expansion_policy = ExpansionPolicy(expansion_strategy)
        return expansion_policy

    def _setup_filter_policy(self, filter_config: FilterConfiguration):
        filter_strategy = QuickKerasFilter('all', filter_config)
        filter_policy = FilterPolicy(filter_strategy)
        return filter_policy

    def _setup_rules_expansion(self, app_config) -> TemplateRules :
        extra_template_path = app_config.extra_template_path if app_config.extra_template_path \
            else Path(__file__).parent.parent / 'rules' / 'direct.csv'

        return TemplateRules(extra_template_path)