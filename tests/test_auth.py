"""Unit: Bearer-token auth on/off, and the OpenAI error envelope shape.

Runs without any model weights (an empty registry is fine) — only exercises the
request middleware / auth dependency / exception handlers.
"""
import pytest
from starlette.testclient import TestClient

from src.__main__ import build_app

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def secured():
    _, registry, app = build_app(["--model-path", "./models", "--api-key", "sekret"])
    with TestClient(app) as c:
        yield c
    registry.shutdown()


@pytest.fixture(scope="module")
def open_client():
    _, registry, app = build_app(["--model-path", "./models"])
    with TestClient(app) as c:
        yield c
    registry.shutdown()


def test_auth_missing_rejected(secured):
    r = secured.get("/v1/models")
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"


def test_auth_wrong_key_rejected(secured):
    r = secured.get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401


def test_auth_correct_key_accepted(secured):
    r = secured.get("/v1/models", headers={"Authorization": "Bearer sekret"})
    assert r.status_code == 200


def test_no_auth_when_key_unset(open_client):
    assert open_client.get("/v1/models").status_code == 200


def test_error_envelope_shape(secured):
    body = secured.get("/v1/models").json()
    assert set(body) == {"error"}
    assert set(body["error"]) >= {"message", "type", "code"}
