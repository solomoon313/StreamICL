import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from participant_text_memory.app import AddRequest, LocalEmbedder, MemoryStore, _memory_items, _token_pieces, create_app


class FakeEmbedder:
    model_name = "fake-embedding"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "bicycle" in text.lower() else [0.0, 1.0]
            for text in texts
        ]

    def pieces(self, text: str) -> list[str]:
        return [text[i : i + 4000] for i in range(0, len(text), 4000)]


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
    pieces = _memory_items(request, FakeEmbedder())
    assert len(pieces) == 3
    assert all(len(piece[2]) <= 4000 for piece in pieces)


def test_token_chunks_preserve_text() -> None:
    text = "  alpha  beta\ngamma   delta  "
    offsets = [(2, 7), (9, 13), (14, 19), (22, 27)]
    chunks = _token_pieces(text, offsets, 2)
    assert chunks == ["  alpha  beta", "\ngamma   delta  "]
    assert "".join(chunks) == text
    assert _token_pieces("   ", [], 2) == ["   "]


def test_local_embedder_uses_downloaded_model(tmp_path: Path, monkeypatch) -> None:
    class Tokenizer:
        def num_special_tokens_to_add(self, *, pair: bool) -> int:
            assert not pair
            return 2

        def __call__(self, text: str, **kwargs):
            assert kwargs == {
                "add_special_tokens": False, "truncation": False, "return_offsets_mapping": True
            }
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    class Vectors:
        def tolist(self):
            return [[1.0, 0.0]]

    class Model:
        max_seq_length = 4
        tokenizer = Tokenizer()

        def __init__(self, path: str, **kwargs):
            assert path == str(tmp_path)
            assert kwargs == {"device": "cpu", "local_files_only": True}

        def encode(self, texts: list[str], **kwargs):
            assert texts == ["ab"]
            assert kwargs["normalize_embeddings"] is True
            return Vectors()

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    embedder = LocalEmbedder(tmp_path)
    assert embedder.pieces("abcde") == ["ab", "cd", "e"]
    assert asyncio.run(embedder.embed(["ab"])) == [[1.0, 0.0]]
