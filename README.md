# attractor-server

HTTP server for the attractor DOT-based AI pipeline runner.

[![CI](https://github.com/bkrabach/attractor-server/actions/workflows/ci.yaml/badge.svg)](https://github.com/bkrabach/attractor-server/actions)
[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

`attractor-server` wraps the [attractor](https://github.com/bkrabach/attractor) pipeline engine in an HTTP API. It lets clients start, cancel, and inspect pipeline runs over REST, stream real-time execution events via SSE, and handle human-in-the-loop questions through a simple POST endpoint. CORS is configurable for browser-based frontends.

## Features

- **REST API** — start, cancel, inspect, and query pipeline runs
- **SSE event stream** — real-time pipeline events via `GET /pipelines/{id}/events`
- **Human-in-the-loop** — post answers to pending questions via `POST /pipelines/{id}/answers`
- **Graph inspection** — retrieve the serialized graph and current checkpoint for any run
- **Context access** — query the live pipeline context during execution
- **CORS support** — configurable origins for browser-based frontends
- **Single-file mode** — run a pipeline file once and exit without starting a persistent server
- **Persistent server mode** — long-running server for managing multiple concurrent pipelines

## Quick Start

```bash
git clone https://github.com/bkrabach/attractor-server.git
cd attractor-server
cargo build

# Run a pipeline file once and exit
cargo run -- --file pipeline.dot

# Start in persistent server mode
cargo run -- --addr 0.0.0.0:8080 --cors-origins http://localhost:3000
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

## Origin

This project was built from the [attractor](https://github.com/strongdm/attractor) natural language specification (NLSpec) by [strongDM](https://github.com/strongdm). The server wraps the attractor pipeline engine in an HTTP API as described in the NLSpec's server architecture section.

## Ecosystem

| Project | Description |
|---------|-------------|
| [attractor](https://github.com/bkrabach/attractor) | DOT-based pipeline engine |
| [attractor-server](https://github.com/bkrabach/attractor-server) | HTTP API server |
| [attractor-ui](https://github.com/bkrabach/attractor-ui) | Web frontend |
| [unified-llm](https://github.com/bkrabach/unified-llm) | Multi-provider LLM client |
| [coding-agent-loop](https://github.com/bkrabach/coding-agent-loop) | Agentic tool loop |

## License

MIT
