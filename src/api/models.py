"""GET /v1/models — list the models available on this server."""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/v1/models")
def list_models(request: Request) -> dict:
    registry = request.app.state.registry
    data = [
        {"id": name, "object": "model", "owned_by": "camaron-audio"}
        for name in registry.list()
    ]
    return {"object": "list", "data": data}
