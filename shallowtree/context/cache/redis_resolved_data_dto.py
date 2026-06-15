from dataclasses import dataclass, field
from typing import List


@dataclass
class RedisResolvedDataDTO:
    inchi_key: str
    classification: str = None
    reactants: List = field(default_factory=list)
    reactants_smiles: List = field(default_factory=list)
    score: float = 0
    resolved: bool = False
    exists: bool = False #TODO decide if this is needed
    time_seconds: int = -1