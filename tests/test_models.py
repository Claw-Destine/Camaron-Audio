"""E2E: GET /v1/models through the OpenAI client and raw HTTP."""
import pytest
from conftest import requires_models

pytestmark = [pytest.mark.e2e, requires_models]


def test_list_models_openai_client(openai_client):
    result = openai_client.models.list()
    assert result.object == "list"
    ids = [m.id for m in result.data]
    assert "whisper-tiny.en" in ids
    assert "kokoro-82m-v1.0" in ids
    # OpenAI object shape
    first = result.data[0]
    assert first.object == "model"
    assert first.owned_by == "camaron-audio"


def test_list_models_raw(http):
    r = http.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert {m["id"] for m in body["data"]} >= {"whisper-tiny.en", "kokoro-82m-v1.0"}
