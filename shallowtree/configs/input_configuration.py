from dataclasses import dataclass, field
from typing import List


@dataclass
class InputConfiguration:
    app_configuration_path: str
    output_path: str
    scaffold: str = None
    smiles: List[str] = field(default_factory=list)
    routes: bool = True
    depth: int = 2
    parallel_processes: int = 1