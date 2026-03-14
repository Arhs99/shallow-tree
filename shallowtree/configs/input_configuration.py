from dataclasses import dataclass, field
from typing import List


@dataclass
class InputConfiguration:
    configuration_yml_path: str
    output_path: str
    scaffold: str
    smiles: List[str] = field(default_factory=list)
    routes: bool = True
    depth: int = 2