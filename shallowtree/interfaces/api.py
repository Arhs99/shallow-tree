# shalow-tree - Retrosynthetic analysis and scoring
# Copyright (C) 2025  Kostas Papadopoulos <kostasp97@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import hmac
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from rdkit import Chem

from shallowtree.configs.input_configuration import InputConfiguration
from shallowtree.interfaces.search_modes.base_tree_search import BaseTreeSearch
from shallowtree.interfaces.search_modes.tree_search import TreeSearch

# Module-level singletons (one set per uvicorn worker process). The heavy
# state (policies, stock, redis) lives on the search instance, so we build at
# most one per mode and reuse it — keeping the branch cache warm across calls.
_config_path: Optional[str] = None
_searches: dict = {}  # "standard" | "scaffold" -> BaseTreeSearch

# API key: set SHALLOWTREE_API_KEY to enable authentication.
# If unset, no auth is required (local-only use).
_api_key: Optional[str] = os.environ.get("SHALLOWTREE_API_KEY")


class SearchRequest(BaseModel):
    smiles: List[str] = Field(..., min_length=1, max_length=128)
    depth: int = Field(default=2, ge=1, le=10)
    scaffold: Optional[str] = None
    include_routes: bool = False
    include_building_blocks: bool = False


class MoleculeResult(BaseModel):
    smiles: str
    score: float
    route: Optional[dict] = None
    building_blocks: Optional[List[str]] = None


class SearchResponse(BaseModel):
    results: List[MoleculeResult]
    metadata: dict


def _get_search(scaffold: Optional[str]) -> BaseTreeSearch:
    """Return a search instance for the requested mode, building it on first
    use. Scaffold is read fresh from the config on every search() call, so a
    cached ScaffoldSearch can serve any scaffold string."""
    mode = "scaffold" if scaffold else "standard"
    search = _searches.get(mode)
    if search is None:
        input_config = InputConfiguration(
            app_configuration_path=_config_path,
            output_path="",
            scaffold=scaffold,
        )
        search = TreeSearch(input_config)
        _searches[mode] = search
    else:
        search._input_config.scaffold = scaffold
    return search


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _config_path
    _config_path = os.environ.get("SHALLOWTREE_CONFIG", "config.json")
    # Warm the standard-mode singleton so /health is meaningful and the first
    # request doesn't pay model-load latency.
    _get_search(None)
    yield
    _searches.clear()


app = FastAPI(
    title="shallow-tree API",
    description="Retrosynthetic analysis and scoring",
    version="0.1",
    lifespan=lifespan,
)


@app.middleware("http")
async def check_api_key(request: Request, call_next):
    if _api_key and request.url.path not in ("/health", "/docs", "/openapi.json"):
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided, _api_key):
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)


@app.get("/health")
def health():
    if "standard" not in _searches:
        raise HTTPException(status_code=503, detail="Search engine not loaded")
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if _config_path is None:
        raise HTTPException(status_code=503, detail="Search engine not loaded")

    # Validate and canonicalize SMILES
    valid_smiles = []
    invalid = []
    for smi in req.smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(smi)
        else:
            valid_smiles.append(Chem.MolToSmiles(mol))

    if not valid_smiles:
        raise HTTPException(status_code=422, detail=f"No valid SMILES provided. Invalid: {invalid}")

    t0 = time.monotonic()

    search_engine = _get_search(req.scaffold)
    search_engine._input_config.depth = req.depth
    df = search_engine.search(valid_smiles)

    elapsed = time.monotonic() - t0

    results = []
    for _, row in df.iterrows():
        result = MoleculeResult(
            smiles=row["SMILES"],
            score=row["score"],
            route=row["route"] if req.include_routes else None,
            building_blocks=row["BBs"] if req.include_building_blocks else None,
        )
        results.append(result)

    return SearchResponse(
        results=results,
        metadata={
            "elapsed_seconds": round(elapsed, 3),
            "molecules_processed": len(valid_smiles),
            "molecules_invalid": len(invalid),
            "depth": req.depth,
        },
    )


def run():
    import uvicorn

    host = os.environ.get("SHALLOWTREE_HOST", "0.0.0.0")
    port = int(os.environ.get("SHALLOWTREE_PORT", "8000"))
    workers = int(os.environ.get("SHALLOWTREE_WORKERS", "1"))
    uvicorn.run(
        "shallowtree.interfaces.api:app",
        host=host,
        port=port,
        workers=workers,
    )


def generate_key():
    print(secrets.token_urlsafe(32))


if __name__ == "__main__":
    run()
