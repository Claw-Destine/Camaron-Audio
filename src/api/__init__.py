"""FastAPI routers for the OpenAI-compatible audio endpoints."""
import asyncio

from fastapi import Request


def get_registry(request: Request):
    return request.app.state.registry


def get_model(request: Request, name: str):
    return request.app.state.registry.require(name)


async def run_in_pool(request: Request, fn, *args, **kwargs):
    """Run blocking inference on the shared thread pool without stalling the loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(request.app.state.registry.pool, lambda: fn(*args, **kwargs))
