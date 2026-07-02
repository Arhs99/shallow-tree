from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class InputConfiguration:
    app_configuration_path: str
    output_path: str
    scaffold: str = None
    smiles: List[str] = field(default_factory=list)
    routes: bool = True
    depth: int = 2
    parallel_processes: int = 1
    prebuilt_stock: Optional[Any] = None
    # Iterative-deepening: when enabled, sweep max_depth from d_start upward and
    # report the minimal resolving depth per target (d_max defaults to ``depth``).
    iterative_deepening: bool = False
    d_start: int = 2
    d_max: int = 2