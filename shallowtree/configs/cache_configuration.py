from pydantic import BaseModel


class CacheConfiguration(BaseModel):
    enabled: bool = True
    host: str = 'localhost'
    port: int = 6379
    db: int = 0
    password: str|None = None
    socket_timeout: float = 5.0
