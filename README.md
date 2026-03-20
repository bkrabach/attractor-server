# attractor-server

HTTP server for the [attractor](https://github.com/bkrabach/attractor) DOT-based AI pipeline runner.

[![CI](https://github.com/bkrabach/attractor-server/actions/workflows/ci.yaml/badge.svg)](https://github.com/bkrabach/attractor-server/actions)
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

`attractor-server` wraps the `attractor` pipeline engine in an HTTP API, enabling:

- **REST API** — start, cancel, inspect, and query pipeline runs
- **SSE event stream** — real-time pipeline events via `GET /pipelines/{id}/events`
- **Human-in-the-loop** — post answers to pending questions via `POST /pipelines/{id}/answers`
- **CORS** — configurable origins for browser-based frontends

## Quick start

```bash
# Run a pipeline file once and exit
attractor-server --file pipeline.dot

# Start in persistent server mode
attractor-server --addr 0.0.0.0:8080 --cors-origins http://localhost:3000
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/pipelines` | Start a pipeline |
| `GET` | `/pipelines/{id}` | Get pipeline status |
| `POST` | `/pipelines/{id}/cancel` | Cancel a pipeline |
| `GET` | `/pipelines/{id}/events` | SSE event stream |
| `GET` | `/pipelines/{id}/graph` | Serialised graph |
| `GET` | `/pipelines/{id}/checkpoint` | Current checkpoint |
| `GET` | `/pipelines/{id}/context` | Pipeline context |
| `GET` | `/pipelines/{id}/questions` | Pending questions |
| `POST` | `/pipelines/{id}/answers` | Submit an answer |

## License

MIT
