from dataclasses import dataclass, field
from typing import List


@dataclass
class RedisDataDTO:
    inchi_key: str
    depth: int = 0
    score: float = 0
    resolved: bool = False
    exists: bool = False #TODO decide if this is needed
    time_seconds: int = -1

