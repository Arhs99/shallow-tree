from typing import List, Sequence

from pydantic import BaseModel


class FilterConfiguration(BaseModel):
    model: str
    use_remote_models: bool =  False
    prod_fp_name: str = 'input_1'
    rxn_fp_name: str = 'input_2'
    exclude_from_policy: List[str] = []
    filter_cutoff: float = 0.05
    freeze_bonds: Sequence[Sequence[int]]|None = None