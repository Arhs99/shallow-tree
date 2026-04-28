from pydantic import BaseModel


class SearchConfiguration(BaseModel):
    algorithm: str
    time_limit: int = 120