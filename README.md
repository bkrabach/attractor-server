# attractor-server

HTTP server for the Attractor DOT-based AI pipeline runner.

Wraps the [`attractor`](https://github.com/bkrabach/attractor) pipeline engine in a
REST + SSE HTTP interface so pipelines can be submitted, monitored, and controlled
remotely — or embedded directly inside a larger Axum application.

## Quick Start

Install the binary:

```sh
cargo install --git https://github.com/bkrabach/attractor-server.git
```

Run the server (defaults to `0.0.0.0:3000`):

```sh
attractor-server
```

With options:

```sh
attractor-server \
  --port 8080 \
  --host 127.0.0.1 \
  --cors-origins "http://localhost:3000,https://myapp.example.com" \
  --data-dir /var/lib/attractor-server
```

Submit a pipeline:

```sh
curl -X POST http://localhost:3000/pipelines \
  -H 'Content-Type: application/json' \
  -d '{
    "dot": "digraph { graph [goal=\"Build the feature\"] start [shape=Mdiamond] exit [shape=Msquare] start -> exit }",
    "context": {}
  }'
# → {"id":"<uuid>","status":"running"}
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/pipelines` | Parse, validate, and launch a pipeline from DOT source |
| `GET` | `/pipelines/{id}` | Query the current status, completed nodes, and active node |
| `POST` | `/pipelines/{id}/cancel` | Abort a running pipeline |
| `GET` | `/pipelines/{id}/events` | Stream pipeline events as SSE (`?since=N` for replay) |
| `GET` | `/pipelines/{id}/questions` | List pending human-in-the-loop questions |
| `POST` | `/pipelines/{id}/questions/{qid}/answer` | Submit an answer to a pending question |
| `GET` | `/pipelines/{id}/graph` | Annotated DOT or SVG graph (`?format=svg` requires Graphviz) |
| `GET` | `/pipelines/{id}/checkpoint` | Latest checkpoint JSON, or `{checkpoint: null}` |
| `GET` | `/pipelines/{id}/context` | Context values from the latest checkpoint |

## Embedding

Add the dependency:

```toml
[dependencies]
attractor-server = { git = "https://github.com/bkrabach/attractor-server.git" }
```

### `create_service` — batteries included

```rust
use attractor_server::{ServerConfig, create_service};
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    let config = ServerConfig {
        port: 8080,
        data_dir: PathBuf::from("/var/lib/attractor"),
        ..ServerConfig::default()
    };

    let app = create_service(&config);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

### `create_router` — mount into an existing Axum app

```rust
use attractor_server::{create_router, state::AppState};
use axum::Router;
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    let state = AppState::new(PathBuf::from("/var/lib/attractor"));

    let app = Router::new()
        .nest("/api", create_router(state))
        // ... your other routes
        ;

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

## CORS

CORS is enabled on all routes. By default all origins are allowed (`*`). Pass
`--cors-origins` on the command line, or set `cors_origins` on `ServerConfig`,
to restrict to specific origins.

## Server-Sent Events

`GET /pipelines/{id}/events` returns a live `text/event-stream` of `PipelineEvent`
objects serialised as JSON. Subscribe immediately after `POST /pipelines` to avoid
missing early events. Use `?since=N` to replay the in-memory event log from index
`N` before joining the live stream:

```sh
curl -N "http://localhost:3000/pipelines/<id>/events?since=0"
```

## Human-in-the-Loop

When the pipeline reaches a `shape=hexagon` (or `type="wait.human"`) node, the
pipeline pauses and a question appears at `GET /pipelines/{id}/questions`. Submit
the answer to resume:

```sh
# List pending questions
curl http://localhost:3000/pipelines/<id>/questions

# Answer
curl -X POST http://localhost:3000/pipelines/<id>/questions/<qid>/answer \
  -H 'Content-Type: application/json' \
  -d '{"answer": "approve"}'
```

## Related

- [attractor](https://github.com/bkrabach/attractor) — the pipeline engine this server wraps

## License

MIT
