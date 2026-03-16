//! Live end-to-end test — requires `LIVE_TEST=1` and valid LLM credentials.
//!
//! NOT run in CI.  Gate with the `LIVE_TEST=1` environment variable.
//! Requires a real LLM backend; credentials are read from the environment
//! via `unified_llm::Client::from_env()` (e.g. `OPENAI_API_KEY`,
//! `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`).
//!
//! Run manually:
//! ```
//! LIVE_TEST=1 OPENAI_API_KEY=sk-... cargo test -p attractor-server --test live_test
//! ```

use axum::{
    Router,
    body::Body,
    http::{Method, Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::Value;
use tempfile::TempDir;
use tokio::time::{Duration, sleep};
use tower::ServiceExt;

use attractor_server::create_router;
use attractor_server::state::AppState;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` when `LIVE_TEST=1` is set in the environment.
fn is_live_test() -> bool {
    std::env::var("LIVE_TEST").as_deref() == Ok("1")
}

/// Build a test app whose temp directory is kept alive for the test duration.
///
/// Returns the router *and* the [`TempDir`] guard — the guard must remain
/// in scope until the pipeline has finished writing its checkpoint files.
fn live_app() -> (Router, TempDir) {
    let dir = TempDir::new().expect("create temp dir");
    let state = AppState::new(dir.path().to_path_buf());
    let app = create_router(state);
    (app, dir)
}

/// Consume the response body and parse it as JSON.
async fn body_json(resp: axum::http::Response<Body>) -> Value {
    let bytes = resp
        .into_body()
        .collect()
        .await
        .expect("collect body")
        .to_bytes();
    serde_json::from_slice(&bytes).expect("valid JSON body")
}

/// A 3-node pipeline with a real LLM prompt node.
///
/// Topology: `start → greet → exit`
///
/// The `greet` node carries a `prompt` attribute so the attractor engine
/// will invoke the configured LLM provider.  The graph-level `goal`
/// attribute gives the engine high-level direction.
fn live_pipeline_dot() -> String {
    r#"digraph pipeline {
  graph [goal="Say hello"]
  start [shape=Mdiamond]
  greet [prompt="Say hello world in one sentence"]
  exit  [shape=Msquare]
  start -> greet
  greet -> exit
}"#
    .to_string()
}

// ---------------------------------------------------------------------------
// Live test
// ---------------------------------------------------------------------------

/// End-to-end pipeline execution through the HTTP API with a real LLM backend.
///
/// Skipped unless `LIVE_TEST=1` is set.  When enabled, the test:
///
/// 1. Creates the router backed by a fresh temp directory.
/// 2. POSTs a 3-node pipeline whose `greet` node issues a real LLM prompt.
/// 3. Polls `GET /pipelines/{id}` every 2 seconds.
/// 4. Returns successfully once the status reaches `completed`.
/// 5. Panics immediately on `failed` status or a 60-second timeout.
#[tokio::test]
async fn live_pipeline_completes() {
    if !is_live_test() {
        eprintln!("Skipping live test: set LIVE_TEST=1 to enable");
        return;
    }

    let (app, _dir) = live_app();

    // POST the pipeline.
    let dot = live_pipeline_dot();
    let payload = serde_json::json!({ "dot": dot, "context": {} });
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method(Method::POST)
                .uri("/pipelines")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "POST /pipelines must return 200"
    );

    let created: Value = body_json(resp).await;
    let id = created["id"]
        .as_str()
        .expect("response must have a string `id`")
        .to_string();

    eprintln!("Live test: pipeline {id} created, polling for completion...");

    // Poll every 2 seconds for up to 60 seconds (30 attempts).
    const MAX_POLLS: u32 = 30;

    for attempt in 1..=MAX_POLLS {
        sleep(Duration::from_secs(2)).await;

        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri(format!("/pipelines/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "GET /pipelines/{{id}} must return 200"
        );

        let json: Value = body_json(resp).await;
        let status = json["status"].as_str().expect("status must be a string");

        eprintln!("Live test [{attempt}/{MAX_POLLS}]: pipeline {id} status = {status}");

        match status {
            "completed" => {
                eprintln!("Live test: pipeline {id} completed successfully");
                return;
            }
            "failed" => {
                panic!("Live test: pipeline {id} failed");
            }
            "running" => {
                // Continue polling.
            }
            other => {
                eprintln!(
                    "Live test: pipeline {id} has unexpected status '{other}', continuing..."
                );
            }
        }

        if attempt == MAX_POLLS {
            panic!(
                "Live test: pipeline {id} did not complete within 60 seconds (last status: {status})"
            );
        }
    }
}
