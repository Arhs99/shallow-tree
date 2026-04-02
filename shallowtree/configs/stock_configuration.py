from pydantic import BaseModel


class StockConfiguration(BaseModel):
    dataset: str
    inchi_key_col: str = "inchi_key"
    price_col: str|None = None
    stop_criteria: dict|None = None
