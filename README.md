# StreamICL-inspired Text Memory API

This repository contains a text-only Add/Search memory service for Agent Memory Leaderboard. It is inspired by StreamICL, but it is not a reproduction of the original algorithm. The evaluation platform owns answer generation and scoring; this service only stores and retrieves memory.

## Method

- Add stores source messages as text chunks of at most 4,000 characters. Positive-score Streaming QA feedback is stored as a question-answer example; nonpositive feedback is acknowledged but not indexed.
- Embeddings come from an OpenAI-compatible API. The default example configuration uses SiliconFlow's `Qwen/Qwen3-Embedding-8B`.
- SQLite stores text, unit-normalized vectors, and Add request IDs. Search ranks all memories belonging to the exact `user_id` by cosine similarity and returns at most the requested `top_k`. Multiple-choice `options`, when present, are included in the embedding query.
- Add is synchronous: HTTP 200 is returned only after embeddings and SQLite writes complete. Repeating an identical `request_id` is idempotent; reusing one for a different payload returns HTTP 409.

Compared with the original StreamICL baseline, this implementation uses a hosted embedding API instead of local BGE, SQLite and an in-process scan instead of FAISS, message chunks and positive-feedback examples instead of question-keyed full trajectories, and caller-provided `top_k` instead of a fixed top four. It does not assemble prompts or call an answer LLM.

## Run locally

Requires Python 3.10 or newer and access to an OpenAI-compatible embedding API. From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env.local
.\.venv\Scripts\python -c "import secrets; print(secrets.token_hex(32))"
```

Put the generated secret in `.env.local` as `PARTICIPANT_MEMORY_API_KEY`, then fill in the embedding API key and run:

```powershell
.\start.ps1
```

The service listens on `http://127.0.0.1:8094`. The startup script does not create a public tunnel. For external evaluation, deploy behind a stable HTTPS endpoint and keep the API key private. The embedding provider receives Add text and Search queries to create vectors.

## API

`GET /health` requires no authentication. Add and Search require `Authorization: Bearer <PARTICIPANT_MEMORY_API_KEY>`; `Authorization: Token` and `X-API-Key` are also accepted.

```text
POST /v1/memories/add
POST /v1/memories/search
```

Add requires `request_id`, `messages` (`role`, nonempty string `content`, optional Unix-millisecond `timestamp`), `user_id`, and `session_id`. Search requires `query`, `user_id`, and `top_k`; `options` is an optional top-level string array for multiple-choice questions. Search returns `{"data": [...]}` in relevance order, or `{"data": []}`. Only text content is supported.

## Test and data retention

```powershell
.\.venv\Scripts\python -m pip install pytest
.\.venv\Scripts\python -m pytest tests -q
```

The SQLite database defaults to `participant_text_memory/data/memory.db` and is ignored by Git. Do not log or publish evaluation payloads or keys. Delete evaluation data within 30 days after a run unless the platform grants written permission to retain it longer. This implementation has not been load-tested for a full public evaluation.

## Attribution

The StreamICL idea comes from Cheng-Kuang Wu, Zhi Rui Tam, Chieh-Yen Lin, Yun-Nung Chen, and Hung-yi Lee, [*StreamBench: Towards Benchmarking Continuous Improvement of Language Agents*](https://papers.nips.cc/paper_files/paper/2024/hash/c189915371c4474fe9789be3728113fc-Abstract-Datasets_and_Benchmarks_Track.html), with [source code](https://github.com/stream-bench/stream-bench). This service also draws on the [AgentMemoryBench streamICL baseline](https://github.com/solomoon313/AgentMemoryBench). The API service here was written separately and differs in the ways listed above.
