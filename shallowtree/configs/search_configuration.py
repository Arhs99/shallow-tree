from pydantic import BaseModel


class SearchConfiguration(BaseModel):
    time_limit: int = 120 # TODO: review - its not used
    score_acceptance_threshold: float = 0.9