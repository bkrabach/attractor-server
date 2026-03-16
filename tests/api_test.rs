//! Integration tests — full HTTP API lifecycle for `attractor-server`.
//!
//! All tests use an in-process axum app (no TCP bind).  Pipelines run with the
//! default `PipelineRunner` backend, which operates in simulation mode and
//! makes no real LLM calls for graphs that complete without work nodes.

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

/// Build a stateless test app backed by a fresh temporary directory.
///
/// The temp dir is created via [`AppState::with_temp_dir`] and persists until
/// the OS reclaims it.  Sufficient for tests that don't inspect the filesystem.
fn test_app() -> Router {
    let state = AppState::with_temp_dir();
    create_router(state, vec![])
}

/// Build a test app whose temp directory is kept alive for the duration of the
/// test.  Returns the router *and* the [`TempDir`] guard — drop the guard to
/// clean up.  Use this when you need checkpoint files to survive between
/// requests (e.g., the checkpoint / context lifecycle tests).
fn persistent_app() -> (Router, TempDir) {
    let dir = TempDir::new().expect("create temp dir");
    let state = AppState::new(dir.path().to_path_buf());
    let app = create_router(state, vec![]);
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

/// POST `/pipelines` with `dot` source and return the parsed JSON response.
///
/// Asserts that the response status is 200 — use the raw helpers for
/// error-path tests.
async fn create_pipeline(app: Router, dot: &str) -> Value {
    let payload = serde_json::json!({ "dot": dot, "context": {} });
    let resp = app
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
        "create_pipeline helper requires 200; check the DOT input"
    );
    body_json(resp).await
}

/// Extract the `id` field from a pipeline JSON response as an owned `String`.
///
/// Panics with a clear message if the field is missing or not a string,
/// so test failures diagnose themselves rather than unwinding silently.
fn pipeline_id(json: &Value) -> String {
    json["id"]
        .as_str()
        .expect("response must have a string `id`")
        .to_string()
}

/// A minimal 3-node digraph used as the default test fixture.
///
/// Contains: `start` (Mdiamond), `task` (regular node), `exit` (Msquare).
fn simple_dot() -> &'static str {
    "digraph pipeline {\n\
       start [shape=Mdiamond]\n\
       task  [label=\"task\"]\n\
       exit  [shape=Msquare]\n\
       start -> task\n\
       task  -> exit\n\
     }"
}

// ---------------------------------------------------------------------------
// Low-level request helpers
// ---------------------------------------------------------------------------

async fn get_req(app: Router, uri: &str) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method(Method::GET)
            .uri(uri)
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn post_empty(app: Router, uri: &str) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method(Method::POST)
            .uri(uri)
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn post_json(app: Router, uri: &str, body: Value) -> axum::http::Response<Body> {
    app.oneshot(
        Request::builder()
            .method(Method::POST)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap(),
    )
    .await
    .unwrap()
}

// ---------------------------------------------------------------------------
// Integration Tests
// ---------------------------------------------------------------------------

/// Settle time shared by tests that wait for a pipeline to make progress.
///
/// Two tests sleep this long; centralising the value makes CI tuning a
/// single-line change.
const TEST_SETTLE_MS: u64 = 2000;

/// Create a pipeline, wait 2 s, then verify status is running or completed.
///
/// The simple_dot graph is tiny; most of the time it will have completed,
/// but we accept "running" for slow CI environments.
#[tokio::test]
async fn full_lifecycle_create_status_complete() {
    let app = test_app();

    let created = create_pipeline(app.clone(), simple_dot()).await;
    let id = pipeline_id(&created);

    // Allow time for the pipeline to execute.
    sleep(Duration::from_millis(TEST_SETTLE_MS)).await;

    let resp = get_req(app, &format!("/pipelines/{id}")).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "GET /pipelines/{{id}} must return 200"
    );

    let json = body_json(resp).await;
    let status = json["status"].as_str().expect("status must be a string");
    assert!(
        status == "running" || status == "completed",
        "status must be 'running' or 'completed' after 2 s; got: {status}"
    );
}

/// Cancel a pipeline immediately after creation; accept 200 (cancelled) or 409
/// (already completed).
///
/// The tiny graph may race to completion before the cancel request arrives.
/// Both outcomes are valid: 200 means the cancel won the race; 409
/// (PIPELINE_NOT_RUNNING) means completion won.  Either confirms the cancel
/// path is exercised.
#[tokio::test]
async fn cancel_pipeline_handles_already_completed() {
    let app = test_app();

    let created = create_pipeline(app.clone(), simple_dot()).await;
    let id = pipeline_id(&created);

    let cancel_resp = post_empty(app.clone(), &format!("/pipelines/{id}/cancel")).await;
    let status = cancel_resp.status();

    // 200 = cancelled successfully; 409 = pipeline already finished before cancel.
    assert!(
        status == StatusCode::OK || status == StatusCode::CONFLICT,
        "cancel must return 200 or 409; got: {status}"
    );

    if status == StatusCode::OK {
        let json = body_json(cancel_resp).await;
        assert_eq!(
            json["status"], "cancelled",
            "cancel response must have status='cancelled'; got: {json}"
        );
    }
}

/// POST `/pipelines` with syntactically invalid DOT must return 400 PARSE_ERROR.
#[tokio::test]
async fn invalid_dot_returns_parse_error() {
    let app = test_app();

    let payload = serde_json::json!({
        "dot": "this is absolutely not valid graphviz dot syntax !!!",
        "context": {}
    });
    let resp = post_json(app, "/pipelines", payload).await;

    assert_eq!(
        resp.status(),
        StatusCode::BAD_REQUEST,
        "invalid DOT must return 400"
    );

    let json = body_json(resp).await;
    assert_eq!(
        json["error"]["code"], "PARSE_ERROR",
        "error code must be PARSE_ERROR; got: {json}"
    );
}

/// Valid DOT syntax but no `shape=Mdiamond` start node must return 422
/// VALIDATION_ERROR with a non-empty `diagnostics` array.
#[tokio::test]
async fn dot_without_start_node_returns_validation_error() {
    let app = test_app();

    // Syntactically valid DOT, but no Mdiamond start node.
    let payload = serde_json::json!({
        "dot": "digraph no_start { a -> b -> c }",
        "context": {}
    });
    let resp = post_json(app, "/pipelines", payload).await;

    assert_eq!(
        resp.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "missing start node must return 422; got: {}",
        resp.status()
    );

    let json = body_json(resp).await;
    assert_eq!(
        json["error"]["code"], "VALIDATION_ERROR",
        "error code must be VALIDATION_ERROR; got: {json}"
    );
    assert!(
        json["error"]["diagnostics"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false),
        "validation error must include a non-empty diagnostics array; got: {json}"
    );
}

/// Two pipelines created sequentially both run concurrently in the background.
/// Verify they have different IDs and are independently retrievable via GET.
#[tokio::test]
async fn concurrent_pipelines_independent() {
    let app = test_app();

    let a = create_pipeline(app.clone(), simple_dot()).await;
    let b = create_pipeline(app.clone(), simple_dot()).await;

    let id_a = pipeline_id(&a);
    let id_b = pipeline_id(&b);

    assert_ne!(id_a, id_b, "two pipelines must have different IDs");

    let resp_a = get_req(app.clone(), &format!("/pipelines/{id_a}")).await;
    let resp_b = get_req(app.clone(), &format!("/pipelines/{id_b}")).await;

    assert_eq!(
        resp_a.status(),
        StatusCode::OK,
        "pipeline A must be findable"
    );
    assert_eq!(
        resp_b.status(),
        StatusCode::OK,
        "pipeline B must be findable"
    );

    let json_a = body_json(resp_a).await;
    let json_b = body_json(resp_b).await;

    assert_eq!(json_a["id"], id_a, "pipeline A status id mismatch");
    assert_eq!(json_b["id"], id_b, "pipeline B status id mismatch");
}

/// `GET /pipelines/{id}/events` must return `Content-Type: text/event-stream`.
///
/// We check only headers — the SSE live stream never closes so we do NOT
/// attempt to collect the body.
#[tokio::test]
async fn events_endpoint_returns_sse_content_type() {
    let app = test_app();

    let created = create_pipeline(app.clone(), simple_dot()).await;
    let id = pipeline_id(&created);

    let resp = get_req(app, &format!("/pipelines/{id}/events")).await;

    assert_eq!(resp.status(), StatusCode::OK, "GET /events must return 200");

    let content_type = resp
        .headers()
        .get("content-type")
        .expect("content-type header must be present")
        .to_str()
        .expect("content-type must be valid utf-8");

    assert!(
        content_type.contains("text/event-stream"),
        "content-type must be text/event-stream; got: {content_type}"
    );
}

/// `GET /pipelines/{id}/graph` must return JSON with `format = "dot"` and a
/// `dot` field whose content contains the word `digraph`.
#[tokio::test]
async fn graph_endpoint_returns_annotated_dot() {
    let app = test_app();

    let created = create_pipeline(app.clone(), simple_dot()).await;
    let id = pipeline_id(&created);

    let resp = get_req(app, &format!("/pipelines/{id}/graph")).await;

    assert_eq!(resp.status(), StatusCode::OK, "GET /graph must return 200");

    let json = body_json(resp).await;
    assert_eq!(
        json["format"], "dot",
        "graph response format must be 'dot'; got: {json}"
    );

    let dot_content = json["dot"].as_str().expect("dot field must be a string");
    assert!(
        dot_content.contains("digraph"),
        "dot content must contain 'digraph'; got: {dot_content}"
    );
}

/// After waiting 2 s for the pipeline to (potentially) complete and write a
/// checkpoint, `GET /checkpoint` must return 200.
///
/// The endpoint always returns 200 — either the checkpoint JSON or
/// `{checkpoint: null}` — so this test verifies the route is reachable and
/// responds correctly regardless of whether execution has finished.
#[tokio::test]
async fn checkpoint_endpoint_after_completion() {
    // Use persistent_app so the logs directory survives across requests.
    let (app, _dir) = persistent_app();

    let created = create_pipeline(app.clone(), simple_dot()).await;
    let id = pipeline_id(&created);

    // Allow time for the pipeline to execute and write a checkpoint.
    sleep(Duration::from_millis(TEST_SETTLE_MS)).await;

    let resp = get_req(app, &format!("/pipelines/{id}/checkpoint")).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "GET /checkpoint must return 200"
    );
}

/// `GET /pipelines/{id}/context` must return a JSON object (possibly empty `{}`
/// when no checkpoint has been written yet).
#[tokio::test]
async fn context_endpoint_returns_json_object() {
    let app = test_app();

    let created = create_pipeline(app.clone(), simple_dot()).await;
    let id = pipeline_id(&created);

    let resp = get_req(app, &format!("/pipelines/{id}/context")).await;
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "GET /context must return 200"
    );

    let json = body_json(resp).await;
    assert!(
        json.is_object(),
        "context endpoint must return a JSON object; got: {json}"
    );
}
