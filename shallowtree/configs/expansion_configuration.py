from typing import List

from pydantic import BaseModel


class ExpansionConfiguration(BaseModel):
    configuration_name: str = "full"
    model: str
    template: str
    cutoff_number: int = 50
    mask: str = ""
    template_column: str = "retro_template"
    cutoff_cumulative: float = 0.995
    use_rdchiral: bool = False
    use_remote_models: bool = False
    rescale_prior: bool = False
    chiral_fingerprints: bool = False
    additive_expansion: bool = False
    expansion_strategy_weights: List[float] = []
