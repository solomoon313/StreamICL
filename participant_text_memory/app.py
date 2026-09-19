from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import math
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field


MAX_PIECE_CHARS = 4000
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    timestamp: int | None = None


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    async_mode: Literal[False] | None = None


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(gt=0, le=1000)


class Embedder(Protocol):
    model_name: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class APIEmbedder:
    def __init__(self, base_url: str, api_key: str, model_name: str, batch_size: int = 10) -> None:
        if not base_url or not api_key or not model_name:
            raise RuntimeError("embedding base URL, API key, and model are required")
        self.url = base_url.rstrip("/") + "/embeddings"
        self.api_key = api_key
        self.model_name = model_name
        self.batch_size = max(1, batch_size)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        timeout = httpx.Timeout(90.0, connect=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            for offset in range(0, len(texts), self.batch_size):
                batch = texts[offset : offset + self.batch_size]
                for attempt in range(5):
                    try:
                        response = await client.post(
                            self.url,
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            json={"model": self.model_name, "input": batch},
                        )
                        if response.status_code in RETRYABLE_STATUS and attempt < 4:
                            await asyncio.sleep(min(2**attempt, 8))
                            continue
                        response.raise_for_status()
                        data = sorted(response.json()["data"], key=lambda row: row["index"])
                        if len(data) != len(batch):
                            raise RuntimeError("embedding response count does not match request")
                        vectors.extend(_unit_vector([float(x) for x in row["embedding"]]) for row in data)
                        break
                    except httpx.RequestError:
                        if attempt == 4:
                            raise
                        await asyncio.sleep(min(2**attempt, 8))
        return vectors


def _unit_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values] if norm else values


def _pieces(text: str) -> list[str]:
    return [text[i : i + MAX_PIECE_CHARS] for i in range(0, len(text), MAX_PIECE_CHARS)]


def _feedback_experience(content: str) -> str | None:
    prefix = "Streaming online QA feedback:\n"
    if not content.startswith(prefix):
        return None
    try:
        feedback = json.loads(content[len(prefix) :])
        if float(feedback.get("score", 0)) <= 0:
            return ""
        question = str(feedback["question"])
        answer = str(feedback["predicted_answer"])
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise ValueError("invalid streaming feedback") from exc
    return f"Successful example\nQuestion: {question}\nAnswer: {answer}"


def _source_time(timestamp: int | None) -> str:
    if timestamp is None:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


class MemoryStore:
    def __init__(self, path: Path, model_name: str) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                pragma journal_mode=wal;
                create table if not exists settings (name text primary key, value text not null);
                create table if not exists requests (
                    request_id text primary key, fingerprint text not null,
                    user_id text not null, session_id text not null
                );
                create table if not exists memories (
                    id text primary key, request_id text not null, user_id text not null,
                    session_id text not null, ordinal integer not null, piece integer not null,
                    content text not null, created_at text not null, vector text not null
                );
                create index if not exists memories_user on memories(user_id);
                """
            )
            existing = db.execute("select value from settings where name='embedding_model'").fetchone()
            if existing and existing[0] != model_name:
                raise RuntimeError("database was created with a different embedding model")
            db.execute(
                "insert or ignore into settings(name, value) values('embedding_model', ?)",
                (model_name,),
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def request_fingerprint(self, request_id: str) -> str | None:
        with self._connect() as db:
            row = db.execute("select fingerprint from requests where request_id=?", (request_id,)).fetchone()
        return row[0] if row else None

    def put(
        self,
        request: AddRequest,
        fingerprint: str,
        items: list[tuple[int, int, str, str]],
        vectors: list[list[float]],
    ) -> None:
        if len(items) != len(vectors):
            raise ValueError("embedding response count does not match memory items")
        with self._connect() as db:
            db.execute("begin immediate")
            existing = db.execute("select fingerprint from requests where request_id=?", (request.request_id,)).fetchone()
            if existing:
                if existing[0] != fingerprint:
                    raise ValueError("request_id reused with a different payload")
                return
            for (ordinal, piece, content, created_at), vector in zip(items, vectors, strict=True):
                memory_id = hashlib.sha256(f"{request.request_id}:{ordinal}:{piece}".encode()).hexdigest()[:32]
                db.execute(
                    """insert into memories
                    (id, request_id, user_id, session_id, ordinal, piece, content, created_at, vector)
                    values (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        memory_id, request.request_id, request.user_id, request.session_id,
                        ordinal, piece, content, created_at, json.dumps(vector, separators=(",", ":")),
                    ),
                )
            db.execute(
                "insert into requests(request_id, fingerprint, user_id, session_id) values (?, ?, ?, ?)",
                (request.request_id, fingerprint, request.user_id, request.session_id),
            )

    def by_user(self, user_id: str) -> list[tuple[str, str, str, list[float]]]:
        with self._connect() as db:
            rows = db.execute(
                "select id, content, created_at, vector from memories where user_id=?",
                (user_id,),
            ).fetchall()
        return [(row[0], row[1], row[2], json.loads(row[3])) for row in rows]


def _memory_items(request: AddRequest) -> list[tuple[int, int, str, str]]:
    items: list[tuple[int, int, str, str]] = []
    for ordinal, message in enumerate(request.messages):
        feedback = _feedback_experience(message.content)
        if feedback == "":
            continue
        content = feedback if feedback is not None else f"{message.role}: {message.content}"
        for piece, chunk in enumerate(_pieces(content)):
            items.append((ordinal, piece, chunk, _source_time(message.timestamp)))
    return items


def create_app(
    *, store: MemoryStore | None = None, embedder: Embedder | None = None, api_key: str | None = None
) -> FastAPI:
    resolved_embedder = embedder or APIEmbedder(
        os.environ.get("OPENAI_EMBEDDING_API_BASE", "").strip(),
        os.environ.get("OPENAI_EMBEDDING_API_KEY", "").strip(),
        os.environ.get("OPENAI_EMBEDDING_MODEL", "").strip(),
    )
    resolved_store = store or MemoryStore(
        Path(os.environ.get("PARTICIPANT_MEMORY_DATABASE", Path(__file__).parent / "data" / "memory.db")),
        resolved_embedder.model_name,
    )
    resolved_key = api_key if api_key is not None else os.environ.get("PARTICIPANT_MEMORY_API_KEY", "").strip()
    if not resolved_key:
        raise RuntimeError("PARTICIPANT_MEMORY_API_KEY is required")

    app = FastAPI(title="Participant Text Memory", docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        presented = (x_api_key or "").strip()
        if not presented and authorization:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() in {"bearer", "token"}:
                presented = value.strip()
        if not presented or not secrets.compare_digest(presented, resolved_key):
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/memories/add", dependencies=[Depends(authorize)])
    async def add(request: AddRequest) -> dict[str, str | bool]:
        fingerprint = hashlib.sha256(
            json.dumps(request.model_dump(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        existing = resolved_store.request_fingerprint(request.request_id)
        if existing and existing != fingerprint:
            raise HTTPException(status_code=409, detail="request_id reused with a different payload")
        if not existing:
            try:
                items = _memory_items(request)
                vectors = await resolved_embedder.embed([item[2] for item in items]) if items else []
                resolved_store.put(request, fingerprint, items, vectors)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except (httpx.HTTPError, RuntimeError) as exc:
                raise HTTPException(status_code=503, detail="embedding service unavailable") from exc
        return {
            "success": True, "request_id": request.request_id,
            "user_id": request.user_id, "session_id": request.session_id,
        }

    @app.post("/v1/memories/search", dependencies=[Depends(authorize)])
    async def search(request: SearchRequest) -> dict[str, list[dict[str, object]]]:
        candidates = resolved_store.by_user(request.user_id)
        if not candidates:
            return {"data": []}
        query = request.query
        if request.options:
            query += "\nOptions:\n" + "\n".join(request.options)
        try:
            vector = (await resolved_embedder.embed([query]))[0]
        except (httpx.HTTPError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail="embedding service unavailable") from exc
        scored = heapq.nlargest(
            request.top_k,
            candidates,
            key=lambda row: sum(a * b for a, b in zip(vector, row[3], strict=True)),
        )
        return {
            "data": [
                {
                    "id": memory_id,
                    "content": content,
                    "score": round(sum(a * b for a, b in zip(vector, stored, strict=True)), 8),
                    "created_at": created_at,
                }
                for memory_id, content, created_at, stored in scored
            ]
        }

    return app
