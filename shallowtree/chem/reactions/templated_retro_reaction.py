from __future__ import annotations

from functools import partial, lru_cache
from typing import Optional, Any, Tuple, Set

from rdchiral import main as rdc
from rdkit import Chem
from rdkit.Chem import AllChem

from shallowtree.chem.molecules.tree_molecule import TreeMolecule
from shallowtree.chem.reactions.rdchiral_production_wrapper import _RdChiralProductWrapper, _rdchiral_product_cache
from shallowtree.chem.reactions.retro_reaction import RetroReaction
from shallowtree.utils.exceptions import MoleculeException
from shallowtree.utils.logging import logger
from shallowtree.utils.lru import LRUCache
from shallowtree.utils.type_utils import StrDict


@lru_cache(maxsize=2048)
def _cached_rdchiral_reaction(smarts: str):
    return rdc.rdchiralReaction(smarts)


@lru_cache(maxsize=2048)
def _cached_rdkit_reaction(smarts: str):
    return AllChem.ReactionFromSmarts(smarts)


class TemplatedRetroReaction(RetroReaction):
    """
    A retrosynthesis reaction that uses a reaction SMARTS and RDChiral to produce reactant molecules.
    The SMILES representation of the reaction is the SMARTS (modified by RDKit)

    :param mol: the molecule
    :param index: the index, defaults to 0
    :param metadata: some meta data
    :param smarts: a string representing the template
    """

    _required_kwargs = ["smarts"]

    def __init__(
            self,
            mol: TreeMolecule,
            index: int = 0,
            metadata: Optional[StrDict] = None,
            **kwargs: Any,
    ):
        super().__init__(mol, index, metadata, **kwargs)
        self.smarts: str = kwargs["smarts"]
        self._use_rdchiral: bool = kwargs.get("use_rdchiral", False)
        self._rd_reaction: Optional[Chem.rdChemReactions.ChemicalReaction] = None

    def __str__(self) -> str:
        return (
            f"retro reaction from template {self.smarts} on molecule {self.mol.smiles}"
        )

    @property
    def rd_reaction(self) -> Chem.rdChemReactions.ChemicalReaction:
        """Return the RDKit reaction created from the SMART"""
        if self._rd_reaction is None:
            self._rd_reaction = AllChem.ReactionFromSmarts(self.smarts)
        return self._rd_reaction

    def to_dict(self) -> StrDict:
        dict_ = super().to_dict()
        dict_["smarts"] = self.smarts
        return dict_

    def _cached_rdchiral_product_wrapper(self, mol):
        key = mol.mapped_smiles
        if key in _rdchiral_product_cache:
            return _rdchiral_product_cache[key]
        wrapper = _RdChiralProductWrapper(mol)
        _rdchiral_product_cache[key] = wrapper
        return wrapper

    def _apply(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        if self._use_rdchiral:
            return self._apply_with_rdchiral()
        return self._apply_with_rdkit()

    def _apply_with_rdchiral(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        """
        Apply a reactions smarts to a molecule and return the products (reactants for retro templates)
        Will try to sanitize the reactants, and if that fails it will not return that molecule
        """
        reaction = _cached_rdchiral_reaction(self.smarts)
        rct = self._cached_rdchiral_product_wrapper(self.mol)
        try:
            reactants = rdc.rdchiralRun(reaction, rct, keep_mapnums=True)
        except RuntimeError as err:
            logger().debug(
                f"Runtime error in RDChiral with template {self.smarts} on {self.mol.smiles}\n{err}"
            )
            reactants = []
        except KeyError as err:
            logger().debug(
                f"Index error in RDChiral with template {self.smarts} on {self.mol.mapped_smiles}\n{err}"
            )
            reactants = []

        # Turning rdchiral outcome into rdkit tuple of tuples to maintain compatibility
        outcomes = []
        for reactant_str in reactants:
            smiles_list = reactant_str.split(".")
            exclude_nums = set(self.mol.mapping_to_index.keys())
            update_func = partial(
                self._update_unmapped_atom_num, exclude_nums=exclude_nums
            )
            try:
                rct_objs = tuple(
                    TreeMolecule(
                        parent=self.mol,
                        smiles=smi,
                        sanitize=True,
                        mapping_update_callback=update_func,
                    )
                    for smi in smiles_list
                )
            except MoleculeException:
                pass
            else:
                outcomes.append(rct_objs)
        self._reactants = tuple(outcomes)

        return self._reactants

    def _apply_with_rdkit(self) -> Tuple[Tuple[TreeMolecule, ...], ...]:
        rxn = _cached_rdkit_reaction(self.smarts)
        try:
            reactants_list = rxn.RunReactants([self.mol.mapped_mol])
        except:  # pylint: disable=bare-except
            reactants_list = []

        outcomes = []
        for reactants in reactants_list:
            exclude_nums = set(self.mol.mapping_to_index.keys())
            update_func = partial(self._inherit_atom_mapping, exclude_nums=exclude_nums)
            try:
                mols = tuple(
                    self._make_or_intern_reactant(rdmol, update_func, self.mol.intern_cache)
                    for rdmol in reactants
                )
            except MoleculeException:
                pass
            else:
                outcomes.append(mols)
        self._reactants = tuple(outcomes)

        return self._reactants

    def _make_or_intern_reactant(self, rdmol, update_func, intern_cache: LRUCache|None) -> TreeMolecule:
        """Construct a reactant TreeMolecule, reusing an interned instance if one
        already exists for the same inchi_key in ``intern_cache``.

        When ``intern_cache`` is None, falls back to the original eager
        construction path. When provided, pre-sanitizes the rdmol so its
        inchi_key can be used as the lookup key; a cache hit returns the
        previously-seen instance and skips the rest of TreeMolecule init
        (deep copy of mapped_mol, mapped_smiles compute, remove_atom_mapping).
        """
        if intern_cache:
            try:
                Chem.SanitizeMol(rdmol)
            except Exception as err:  # noqa: BLE001 — RDKit raises many things
                raise MoleculeException(f"sanitize failed before interning: {err}") from err
            ik = Chem.MolToInchiKey(rdmol)
            if not ik:
                raise MoleculeException("could not compute inchi_key for interning")

            cached = intern_cache.get(ik, None)

            if cached:
                return cached
            else:
                new_mol = TreeMolecule(
                    parent=self.mol,
                    rd_mol=rdmol,
                    sanitize=False,
                    mapping_update_callback=update_func,
                )
                new_mol._is_sanitized = True #TODO: find a better solution
                intern_cache[ik] = new_mol
                return new_mol

        return TreeMolecule(
            parent=self.mol,
            rd_mol=rdmol,
            sanitize=True,
            mapping_update_callback=update_func,
        )

    def _make_smiles(self):
        return AllChem.ReactionToSmiles(self.rd_reaction)

    def _inherit_atom_mapping(self, mol: TreeMolecule, exclude_nums: Set[int]) -> None:
        """
        Update the internal atom mapping dictionary by inspecting the `reaction_atom_idx`
        property of the atoms and comparing it with the parent-molecule.

        This is used for child molecules created by RDKit reaction application.
        RDChiral takes care of this automatically.
        """
        if mol.parent is None:
            return

        for atom in mol.mapped_mol.GetAtoms():
            if not atom.HasProp("react_atom_idx"):
                continue
            index = atom.GetProp("react_atom_idx")
            mapping = mol.parent.index_to_mapping.get(int(index))
            if mapping:
                atom.SetAtomMapNum(mapping)

        self._update_unmapped_atom_num(mol, exclude_nums)


