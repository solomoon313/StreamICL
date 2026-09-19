import json
from pathlib import Path

from fastapi.testclient import TestClient

from participant_text_memory.app import AddRequest, MemoryStore, _memory_items, create_app


class FakeEmbedder:
    model_name = "fake-embedding"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "bicycle" in text.lower() else [0.0, 1.0]
            for text in texts
        ]


def _client(tmp_path: Path) -> TestClient:
    app = create_app(
        store=MemoryStore(tmp_path / "memory.db", FakeEmbedder.model_name),
        embedder=FakeEmbedder(),
        api_key="test-key",
    )
    return TestClient(app)


def _add_payload() -> dict[str, object]:
    return {
        "request_id": "run:session:1",
        "messages": [
            {"role": "user", "timestamp": 1704067200000, "content": "Alice rides a bicycle."},
            {"role": "assistant", "content": "She rides it on Sundays."},
        ],
        "user_id": "run:user:1",
        "session_id": "session:1",
    }


def test_add_search_idempotency_and_isolation(tmp_path: Path) -> None:
    client = _client(tmp_path)
    auth = {"Authorization": "Bearer test-key"}
    payload = _add_payload()

    first = client.post("/v1/memories/add", json=payload, headers=auth)
    assert first.status_code == 200
    assert first.json() == {
        "success": True,
        "request_id": "run:session:1",
        "user_id": "run:user:1",
        "session_id": "session:1",
    }
    assert client.post("/v1/memories/add", json=payload, headers=auth).status_code == 200

    found = client.post(
        "/v1/memories/search",
        json={
            "query": "What does Alice ride?", "options": ["A. bicycle", "B. train"],
            "user_id": "run:user:1", "top_k": 100,
        },
        headers=auth,
    )
    assert found.status_code == 200
    assert len(found.json()["data"]) == 2
    assert "bicycle" in found.json()["data"][0]["content"]
    assert len({row["id"] for row in found.json()["data"]}) == 2

    empty = client.post(
        "/v1/memories/search",
        json={"query": "Alice", "user_id": "run:user:2", "top_k": 100},
        headers=auth,
    )
    assert empty.json() == {"data": []}

    changed = {**payload, "messages": [{"role": "user", "content": "different"}]}
    assert client.post("/v1/memories/add", json=changed, headers=auth).status_code == 409


def test_authentication_and_input_contract(tmp_path: Path) -> None:
    client = _client(tmp_path)
    assert client.get("/health").json() == {"status": "ok"}
    assert client.post("/v1/memories/add", json=_add_payload()).status_code == 401
    assert client.post("/v1/memories/add", json=_add_payload(), headers={"X-API-Key": "test-key"}).status_code == 200
    assert client.post(
        "/v1/memories/search",
        json={"query": "x", "user_id": "run:user:1", "top_k": 0},
        headers={"Authorization": "Token test-key"},
    ).status_code == 422


def test_positive_feedback_only_and_long_text_chunks(tmp_path: Path) -> None:
    client = _client(tmp_path)
    auth = {"Authorization": "Bearer test-key"}

    for score in (0, 1):
        feedback = {
            "question": "What does Alice ride?", "predicted_answer": "bicycle", "score": score,
        }
        response = client.post(
            "/v1/memories/add",
            json={
                "request_id": f"feedback:{score}",
                "messages": [{"role": "user", "content": "Streaming online QA feedback:\n" + json.dumps(feedback)}],
                "user_id": "run:user:1", "session_id": f"feedback:{score}",
            },
            headers=auth,
        )
        assert response.status_code == 200
    found = client.post(
        "/v1/memories/search",
        json={"query": "bicycle", "user_id": "run:user:1", "top_k": 100},
        headers=auth,
    )
    assert len(found.json()["data"]) == 1
    assert "Successful example" in found.json()["data"][0]["content"]

    request = AddRequest.model_validate(
        {
            "request_id": "long:1", "messages": [{"role": "user", "content": "a" * 8100}],
            "user_id": "run:user:1", "session_id": "long:1",
        }
    )
    pieces = _memory_items(request)
    assert len(pieces) == 3
    assert all(len(piece[2]) <= 4000 for piece in pieces)
