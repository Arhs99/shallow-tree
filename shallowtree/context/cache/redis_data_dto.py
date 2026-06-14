from dataclasses import dataclass, field
from typing import List


@dataclass
class RedisDataDTO:
    inchi_key: str
    # Remaining depth budget (max_depth - depth) at which the verdict held, NOT
    # an absolute tree-depth; reuse is budget-aware (see BaseTreeSearch._can_reuse).
    budget: int = 0
    score: float = 0
    resolved: bool = False
    exists: bool = False #TODO decide if this is needed
    timestamp: int = -1

