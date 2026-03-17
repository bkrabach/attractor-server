//! HTTP route handlers for pipeline lifecycle management.
//!
//! Exposes endpoints:
//! - `GET  /pipelines`                              — list all pipeline runs
//! - `POST /pipelines`                              — create and launch a pipeline run
//! - `GET  /pipelines/{id}`                         — query the status of a pipeline
//! - `POST /pipelines/{id}/cancel`                  — abort a running pipeline
//! - `GET  /pipelines/{id}/questions`               — list pending human-in-the-loop questions
//! - `POST /pipelines/{id}/questions/{qid}/answer`  — submit an answer for a pending question
//! - `GET  /pipelines/{id}/graph`                   — annotated DOT or SVG graph
//! - `GET  /pipelines/{id}/checkpoint`              — latest checkpoint or null
//! - `GET  /pipelines/{id}/context`                 — context values from latest checkpoint
//! - `GET  /pipelines/{id}/events`                  — stream pipeline events as SSE

use std::collections::HashMap;
use std::sync::Arc;

use std::convert::Infallible;

use axum::{
    Json, Router,
    extract::{Path, Query, State},
    response::sse::{Event, Sse},
    routing::{get, post},
};
use chrono::{DateTime, Utc};
use futures::StreamExt as _;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Mutex, broadcast};
use tokio_stream::wrappers::BroadcastStream;
use uuid::Uuid;

use attractor::{
    Answer, AnswerValue, Checkpoint, Interviewer, PipelineEvent, PipelineRunner, QuestionOption,
    QuestionType, engine::RunConfig, handler::CodergenHandler, parse_dot, validate,
    validation::Severity,
};

use crate::backends::LlmCodergenBackend;

use crate::error::ServerError;
use crate::interviewer::{HttpInterviewer, InterviewerState};
use crate::state::{AppState, PipelineHandle, PipelineInner, PipelineStatus};

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

/// Body for `POST /pipelines`.
#[derive(Debug, Deserialize)]
pub struct CreatePipelineRequest {
    /// DOT source for the pipeline graph.
    pub dot: String,
    /// Initial context values injected into the pipeline.
    pub context: HashMap<String, Value>,
}

/// Successful response from `POST /pipelines`.
#[derive(Debug, Serialize)]
pub struct CreatePipelineResponse {
    /// Unique pipeline run identifier (UUID v4).
    pub id: String,
    /// Lifecycle status — always `running` immediately after creation.
    pub status: PipelineStatus,
}

/// Response body for `GET /pipelines/{id}`.
#[derive(Debug, Serialize)]
pub struct PipelineStatusResponse {
    /// Pipeline run identifier.
    pub id: String,
    /// Current lifecycle status.
    pub status: PipelineStatus,
    /// When the pipeline was spawned.
    pub started_at: DateTime<Utc>,
    /// Node IDs that have completed (in order).
    pub completed_nodes: Vec<String>,
    /// The node currently being executed, if any.
    pub current_node: Option<String>,
}

/// Response body for `GET /pipelines` (list all pipelines).
///
/// Intentionally structurally identical to [`PipelineStatusResponse`]: keeping
/// them as distinct named types preserves independent versioning — either can
/// gain or drop fields without affecting the other endpoint's contract.
#[derive(Debug, Serialize)]
pub struct PipelineSummary {
    /// Pipeline run identifier.
    pub id: String,
    /// Current lifecycle status.
    pub status: PipelineStatus,
    /// When the pipeline was spawned.
    pub started_at: DateTime<Utc>,
    /// Node IDs that have completed (in order).
    pub completed_nodes: Vec<String>,
    /// The node currently being executed, if any.
    pub current_node: Option<String>,
}

/// Response body for `POST /pipelines/{id}/cancel`.
#[derive(Debug, Serialize)]
pub struct CancelResponse {
    /// Final status — always `cancelled` on success.
    pub status: PipelineStatus,
}

/// Response body for `GET /pipelines/{id}/questions`.
#[derive(Debug, Serialize)]
pub struct QuestionsResponse {
    /// Pending questions awaiting a human answer.
    pub questions: Vec<QuestionResponse>,
}

/// A single pending question exposed via the HTTP API.
#[derive(Debug, Serialize)]
pub struct QuestionResponse {
    /// Unique question identifier (UUID v4 string).
    pub qid: String,
    /// The question text to display.
    pub text: String,
    /// The kind of input expected.
    pub question_type: QuestionType,
    /// Options for select-type questions.
    pub options: Vec<QuestionOption>,
    /// When the question was created.
    pub created_at: DateTime<Utc>,
}

/// Body for `POST /pipelines/{id}/questions/{qid}/answer`.
#[derive(Debug, Deserialize)]
pub struct AnswerRequest {
    /// The human's answer — option key, option label, or free text.
    pub answer: String,
}

/// Response body for `POST /pipelines/{id}/questions/{qid}/answer`.
#[derive(Debug, Serialize)]
pub struct AnswerResponse {
    /// Always `"answered"` on success.
    pub status: String,
}

/// Query parameters for `GET /pipelines/{id}/graph`.
#[derive(Debug, Deserialize)]
pub struct GraphQuery {
    /// Output format: `"dot"` (default) or `"svg"` (requires Graphviz installed).
    pub format: Option<String>,
}

/// Query parameters for `GET /pipelines/{id}/events`.
#[derive(Debug, Deserialize)]
pub struct EventsQuery {
    /// If provided, replay events starting at this index in the event log
    /// before connecting to the live broadcast stream.
    pub since: Option<usize>,
}

/// Response body for `GET /pipelines/{id}/graph`.
#[derive(Debug, Serialize)]
pub struct GraphResponse {
    /// The annotated DOT or SVG source.
    pub dot: String,
    /// The format of the returned content: `"dot"` or `"svg"`.
    pub format: String,
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the pipeline lifecycle router.
///
/// Mount this onto your Axum application with `.with_state(app_state)`.
pub fn router() -> Router<AppState> {
    Router::new()
        .route("/pipelines", post(create_pipeline).get(list_pipelines))
        .route("/pipelines/{id}", get(get_pipeline))
        .route("/pipelines/{id}/cancel", post(cancel_pipeline))
        .route("/pipelines/{id}/questions", get(get_questions))
        .route("/pipelines/{id}/questions/{qid}/answer", post(post_answer))
        .route("/pipelines/{id}/graph", get(get_graph))
        .route("/pipelines/{id}/checkpoint", get(get_checkpoint))
        .route("/pipelines/{id}/context", get(get_context))
        .route("/pipelines/{id}/events", get(get_events))
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Core pipeline creation logic — parse, validate, spawn tasks, insert into state.
///
/// Returns the new pipeline ID on success.  Extracted from the HTTP handler so
/// it can be called directly from `main.rs` when pre-loading a `--file`
/// pipeline at server start-up.
pub async fn spawn_pipeline(
    state: &AppState,
    dot: String,
    _context: HashMap<String, Value>,
) -> Result<String, ServerError> {
    // 1. Parse DOT source.
    let graph = parse_dot(&dot).map_err(|e| ServerError::ParseError(e.to_string()))?;

    // 2. Validate — filter to Error-severity diagnostics only.
    let diags = validate(&graph, &[]);
    let errors: Vec<_> = diags
        .into_iter()
        .filter(|d| d.severity == Severity::Error)
        .collect();
    if !errors.is_empty() {
        return Err(ServerError::ValidationError(errors));
    }

    // 3. Generate UUID pipeline ID.
    let pipeline_id = Uuid::new_v4().to_string();

    // 4. Create HttpInterviewer with shared InterviewerState.
    let interviewer_state = Arc::new(InterviewerState::new());
    let interviewer = Arc::new(HttpInterviewer::new(interviewer_state.clone()));

    // 5. Build PipelineRunner with the HttpInterviewer and real LLM backend.
    //
    // SRV-BUG-001: wire up LlmCodergenBackend so codergen nodes call the LLM
    // instead of returning simulated responses.  If no API credentials are
    // available in the environment, from_env() logs a warning and returns None,
    // and CodergenHandler falls back to simulation mode gracefully.
    let codergen_backend: Option<Box<dyn attractor::handler::CodergenBackend>> =
        LlmCodergenBackend::from_env("gpt-4o").map(|b| Box::new(b) as _);
    let codergen_handler = Arc::new(CodergenHandler::new(codergen_backend));

    let (runner, runner_rx) = PipelineRunner::builder()
        .with_interviewer(interviewer as Arc<dyn Interviewer>)
        .with_handler("codergen", codergen_handler)
        .build();

    // 6. Create broadcast channel (capacity 10,000) for external event streaming.
    let (event_tx, _event_rx) = broadcast::channel::<PipelineEvent>(10_000);

    // Create the mutable inner state behind Arc<Mutex<>>.
    let inner: Arc<Mutex<PipelineInner>> = Arc::new(Mutex::new(PipelineInner::new()));

    // Prepare logs root directory.
    let logs_root = state.logs_root_for(&pipeline_id);
    let _ = tokio::fs::create_dir_all(&logs_root).await;

    // 7. Spawn event relay task: forwards runner events → broadcast + event_log,
    //    and tracks current_node / completed_nodes from events.
    {
        let inner_relay = inner.clone();
        let event_tx_relay = event_tx.clone();
        tokio::spawn(async move {
            let mut rx = runner_rx;
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        {
                            let mut lock = inner_relay.lock().await;
                            match &event {
                                PipelineEvent::StageStarted { name, .. } => {
                                    lock.current_node = Some(name.clone());
                                }
                                PipelineEvent::StageCompleted { name, .. } => {
                                    lock.completed_nodes.push(name.clone());
                                    lock.current_node = None;
                                }
                                PipelineEvent::PipelineCompleted { .. } => {
                                    lock.status = PipelineStatus::Completed;
                                    lock.current_node = None;
                                }
                                PipelineEvent::PipelineFailed { .. } => {
                                    lock.status = PipelineStatus::Failed;
                                    lock.current_node = None;
                                }
                                _ => {}
                            }
                            lock.push_event(event.clone());
                        }
                        // Forward to broadcast channel; ignore if no subscribers.
                        let _ = event_tx_relay.send(event);
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // Skip lagged events; keep consuming.
                        continue;
                    }
                }
            }
        });
    }

    // 8. Spawn pipeline execution task.
    let dot_source = dot.clone();
    let inner_exec = inner.clone();
    let logs_root_exec = logs_root.clone();
    let join_handle = tokio::spawn(async move {
        let config = RunConfig::new(logs_root_exec);
        let result = runner.run(&dot_source, config).await;
        let mut lock = inner_exec.lock().await;
        // Only update if not already set by relay task.
        if lock.status == PipelineStatus::Running {
            match result {
                Ok(_) => lock.status = PipelineStatus::Completed,
                Err(_) => lock.status = PipelineStatus::Failed,
            }
        }
    });
    let abort_handle = join_handle.abort_handle();

    // 9. Create PipelineHandle with Arc<Mutex<PipelineInner>>.
    let handle = Arc::new(PipelineHandle {
        id: pipeline_id.clone(),
        dot_source: dot,
        graph,
        event_tx,
        interviewer_state,
        logs_root,
        started_at: Utc::now(),
        abort_handle,
        inner,
    });

    // 10. Store in AppState and return.
    state.insert_pipeline(handle).await;

    Ok(pipeline_id)
}

/// `GET /pipelines` — return a summary of all known pipeline runs.
pub async fn list_pipelines(
    State(state): State<AppState>,
) -> Result<Json<Vec<PipelineSummary>>, ServerError> {
    let pipelines = state.pipelines.read().await;
    let mut summaries = Vec::new();
    for handle in pipelines.values() {
        let inner = handle.inner.lock().await;
        summaries.push(PipelineSummary {
            id: handle.id.clone(),
            status: inner.status.clone(),
            started_at: handle.started_at,
            completed_nodes: inner.completed_nodes.clone(),
            current_node: inner.current_node.clone(),
        });
    }
    // Sort by creation time for stable, deterministic ordering across calls.
    summaries.sort_by_key(|s| s.started_at);
    Ok(Json(summaries))
}

/// `POST /pipelines` — parse, validate, spawn, and return `{id, status: running}`.
pub async fn create_pipeline(
    State(state): State<AppState>,
    Json(req): Json<CreatePipelineRequest>,
) -> Result<Json<CreatePipelineResponse>, ServerError> {
    let pipeline_id = spawn_pipeline(&state, req.dot, req.context).await?;
    Ok(Json(CreatePipelineResponse {
        id: pipeline_id,
        status: PipelineStatus::Running,
    }))
}

/// `GET /pipelines/{id}` — return current pipeline status.
pub async fn get_pipeline(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<PipelineStatusResponse>, ServerError> {
    let handle = state
        .get_pipeline(&id)
        .await
        .ok_or_else(|| ServerError::PipelineNotFound(id.clone()))?;

    let inner = handle.inner.lock().await;

    Ok(Json(PipelineStatusResponse {
        id: handle.id.clone(),
        status: inner.status.clone(),
        started_at: handle.started_at,
        completed_nodes: inner.completed_nodes.clone(),
        current_node: inner.current_node.clone(),
    }))
}

/// `POST /pipelines/{id}/cancel` — abort a running pipeline.
pub async fn cancel_pipeline(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<CancelResponse>, ServerError> {
    let handle = state
        .get_pipeline(&id)
        .await
        .ok_or_else(|| ServerError::PipelineNotFound(id.clone()))?;

    // Verify the pipeline is still running before aborting.
    {
        let inner = handle.inner.lock().await;
        if inner.status != PipelineStatus::Running {
            return Err(ServerError::PipelineNotRunning(id.clone()));
        }
    }

    // Abort the spawned execution task.
    handle.abort_handle.abort();

    // Mark as cancelled.
    let mut inner = handle.inner.lock().await;
    inner.status = PipelineStatus::Cancelled;

    Ok(Json(CancelResponse {
        status: PipelineStatus::Cancelled,
    }))
}

/// `GET /pipelines/{id}/questions` — list pending human-in-the-loop questions.
pub async fn get_questions(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<QuestionsResponse>, ServerError> {
    let handle = state
        .get_pipeline(&id)
        .await
        .ok_or_else(|| ServerError::PipelineNotFound(id.clone()))?;

    let pending = handle.interviewer_state.pending_questions();
    let questions = pending
        .into_iter()
        .map(|pq| QuestionResponse {
            qid: pq.id,
            text: pq.question.text,
            question_type: pq.question.question_type,
            options: pq.question.options,
            created_at: pq.created_at,
        })
        .collect();

    Ok(Json(QuestionsResponse { questions }))
}

/// `POST /pipelines/{id}/questions/{qid}/answer` — submit an answer for a pending question.
pub async fn post_answer(
    State(state): State<AppState>,
    Path((id, qid)): Path<(String, String)>,
    Json(req): Json<AnswerRequest>,
) -> Result<Json<AnswerResponse>, ServerError> {
    let handle = state
        .get_pipeline(&id)
        .await
        .ok_or_else(|| ServerError::PipelineNotFound(id.clone()))?;

    // Look up the pending question to find options and build a typed answer.
    let pending_question = handle
        .interviewer_state
        .pending
        .lock()
        .expect("pending lock poisoned")
        .get(&qid)
        .cloned();

    let pq = pending_question.ok_or_else(|| ServerError::QuestionNotFound(qid.clone()))?;

    // Try to match the submitted answer string against options by key or label
    // (case-insensitive).  Fall back to a free-text answer when there are no
    // options or no match is found.
    let answer = if let Some(opt) = pq.question.options.iter().find(|o| {
        o.key.eq_ignore_ascii_case(&req.answer) || o.label.eq_ignore_ascii_case(&req.answer)
    }) {
        Answer::selected(opt.clone())
    } else {
        // Free-text or unmatched input.
        Answer {
            value: AnswerValue::Selected(req.answer.clone()),
            selected_option: None,
            text: req.answer,
        }
    };

    handle
        .interviewer_state
        .submit_answer(&qid, answer)
        .map_err(|e| match e {
            "not_found" => ServerError::QuestionNotFound(qid.clone()),
            "already_answered" => ServerError::QuestionAlreadyAnswered(qid.clone()),
            _ => ServerError::Internal(e.to_string()),
        })?;

    Ok(Json(AnswerResponse {
        status: "answered".to_string(),
    }))
}

// ---------------------------------------------------------------------------
// Helpers — inspection utilities
// ---------------------------------------------------------------------------

/// Annotate a DOT source string with node status colors.
///
/// Injects `[style=filled fillcolor="..."]` attribute overrides for every
/// node that appears in `graph.nodes`, appended just before the closing `}`
/// of the digraph block so that they layer on top of the original attributes.
///
/// Colour key:
/// - `#4caf50` (green)  — completed nodes
/// - `#ffeb3b` (yellow) — the current node while the pipeline is running
/// - `#f44336` (red)    — the current node when the pipeline has failed
/// - `#9e9e9e` (gray)   — pending nodes (not yet reached)
fn annotate_dot(
    dot_source: &str,
    graph: &attractor::Graph,
    completed_nodes: &[String],
    current_node: Option<&str>,
    failed: bool,
) -> String {
    use std::collections::HashSet;

    let completed_set: HashSet<&str> = completed_nodes.iter().map(String::as_str).collect();

    let mut annotations = String::new();
    for node_id in graph.nodes.keys() {
        let color = if completed_set.contains(node_id.as_str()) {
            "#4caf50" // green = completed
        } else if current_node == Some(node_id.as_str()) {
            if failed { "#f44336" } else { "#ffeb3b" } // red = failed current, yellow = active
        } else {
            "#9e9e9e" // gray = pending
        };
        annotations.push_str(&format!(
            "  {node_id} [style=filled fillcolor=\"{color}\"]\n"
        ));
    }

    // Insert the annotation block just before the last `}` in the DOT source.
    let trimmed = dot_source.trim_end();
    if let Some(pos) = trimmed.rfind('}') {
        format!("{}\n{}{}", &trimmed[..pos], annotations, &trimmed[pos..])
    } else {
        // Malformed DOT — append annotations at the end anyway.
        format!("{}\n{}", trimmed, annotations)
    }
}

/// Render a DOT string to SVG by shelling out to the `dot` binary.
///
/// Returns `Err(ServerError::GraphvizNotAvailable)` when `dot` is not
/// installed or exits with a non-zero status.
fn render_svg(dot: &str) -> Result<String, ServerError> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut child = Command::new("dot")
        .arg("-Tsvg")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|_| ServerError::GraphvizNotAvailable)?;

    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(dot.as_bytes());
    }

    let output = child
        .wait_with_output()
        .map_err(|_| ServerError::GraphvizNotAvailable)?;

    if !output.status.success() {
        return Err(ServerError::GraphvizNotAvailable);
    }

    String::from_utf8(output.stdout).map_err(|e| ServerError::Internal(e.to_string()))
}

// ---------------------------------------------------------------------------
// Inspection handlers
// ---------------------------------------------------------------------------

/// `GET /pipelines/{id}/graph` — return annotated DOT (or SVG) for the pipeline graph.
pub async fn get_graph(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<GraphQuery>,
) -> Result<Json<GraphResponse>, ServerError> {
    let handle = state
        .get_pipeline(&id)
        .await
        .ok_or_else(|| ServerError::PipelineNotFound(id.clone()))?;

    let (completed_nodes, current_node, status) = {
        let inner = handle.inner.lock().await;
        (
            inner.completed_nodes.clone(),
            inner.current_node.clone(),
            inner.status.clone(),
        )
    };

    let failed = status == crate::state::PipelineStatus::Failed;
    let annotated = annotate_dot(
        &handle.dot_source,
        &handle.graph,
        &completed_nodes,
        current_node.as_deref(),
        failed,
    );

    let format = query.format.as_deref().unwrap_or("dot");

    match format {
        "svg" => {
            // render_svg shells out to `dot -Tsvg`, which uses blocking I/O.
            // Wrap in spawn_blocking so we don't stall a Tokio worker thread.
            let dot = annotated.clone();
            let svg = tokio::task::spawn_blocking(move || render_svg(&dot))
                .await
                .map_err(|e| ServerError::Internal(e.to_string()))??;
            Ok(Json(GraphResponse {
                dot: svg,
                format: "svg".to_string(),
            }))
        }
        _ => Ok(Json(GraphResponse {
            dot: annotated,
            format: "dot".to_string(),
        })),
    }
}

/// `GET /pipelines/{id}/checkpoint` — return the latest checkpoint or `{checkpoint: null}`.
pub async fn get_checkpoint(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let handle = state
        .get_pipeline(&id)
        .await
        .ok_or_else(|| ServerError::PipelineNotFound(id.clone()))?;

    let checkpoint_path = Checkpoint::default_path(&handle.logs_root);
    if !checkpoint_path.exists() {
        return Ok(Json(serde_json::json!({ "checkpoint": null })));
    }

    let checkpoint =
        Checkpoint::load(&checkpoint_path).map_err(|e| ServerError::Internal(e.to_string()))?;

    let checkpoint_json =
        serde_json::to_value(&checkpoint).map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(serde_json::json!({ "checkpoint": checkpoint_json })))
}

/// `GET /pipelines/{id}/events` — stream pipeline events as SSE.
///
/// Subscribes to the live broadcast channel **first** (before reading history)
/// to avoid a race window where events could be missed between the history read
/// and the subscription.
///
/// If `?since=N` is provided, events at indices `N..` from the in-memory
/// `event_log` are replayed as history before the live stream begins.
pub async fn get_events(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<EventsQuery>,
) -> Result<Sse<impl futures::Stream<Item = Result<Event, Infallible>>>, ServerError> {
    let handle = state
        .get_pipeline(&id)
        .await
        .ok_or_else(|| ServerError::PipelineNotFound(id.clone()))?;

    // Subscribe to live events FIRST (before reading history) to avoid missing
    // events that arrive between the history snapshot and the subscription.
    let live_rx = handle.event_tx.subscribe();

    // Read history from event_log if ?since=N is provided.
    let history_events: Vec<PipelineEvent> = if let Some(since) = query.since {
        let inner = handle.inner.lock().await;
        let start = since.min(inner.event_log.len()); // clamp: since > log length → empty history
        inner.event_log[start..].to_vec()
    } else {
        vec![]
    };

    // Convert history to a stream of SSE events.
    let history_stream = tokio_stream::iter(history_events).map(crate::sse::pipeline_event_to_sse);

    // Convert the live broadcast receiver to a stream; discard lag errors.
    let live_stream = BroadcastStream::new(live_rx)
        .filter_map(|result| async move { result.ok().map(crate::sse::pipeline_event_to_sse) });

    // Chain history replay then live stream.
    let combined = history_stream.chain(live_stream);

    Ok(Sse::new(combined))
}

/// `GET /pipelines/{id}/context` — return the context values from the latest checkpoint.
pub async fn get_context(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let handle = state
        .get_pipeline(&id)
        .await
        .ok_or_else(|| ServerError::PipelineNotFound(id.clone()))?;

    let checkpoint_path = Checkpoint::default_path(&handle.logs_root);
    if !checkpoint_path.exists() {
        return Ok(Json(serde_json::json!({})));
    }

    let checkpoint =
        Checkpoint::load(&checkpoint_path).map_err(|e| ServerError::Internal(e.to_string()))?;

    let context_json = serde_json::to_value(&checkpoint.context_values)
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(context_json))
}

// ---------------------------------------------------------------------------
// Tests — RED written first, then GREEN implemented above
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use axum::{
        body::Body,
        http::{self, Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    // ---------------------------------------------------------------------------
    // Helper: build a test Axum app with the pipeline router.
    // ---------------------------------------------------------------------------
    fn test_app() -> axum::Router {
        let state = AppState::with_temp_dir();
        router().with_state(state)
    }

    // ---------------------------------------------------------------------------
    // Helper: POST JSON body and return the response.
    // ---------------------------------------------------------------------------
    async fn post_json(
        app: axum::Router,
        uri: &str,
        body: serde_json::Value,
    ) -> axum::http::Response<Body> {
        app.oneshot(
            Request::builder()
                .method(http::Method::POST)
                .uri(uri)
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap()
    }

    // ---------------------------------------------------------------------------
    // Helper: GET and return the response.
    // ---------------------------------------------------------------------------
    async fn get_req(app: axum::Router, uri: &str) -> axum::http::Response<Body> {
        app.oneshot(
            Request::builder()
                .method(http::Method::GET)
                .uri(uri)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
    }

    // ---------------------------------------------------------------------------
    // Helper: consume response body as parsed JSON.
    // ---------------------------------------------------------------------------
    async fn body_json(resp: axum::http::Response<Body>) -> serde_json::Value {
        let bytes = resp
            .into_body()
            .collect()
            .await
            .expect("collect body")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("valid JSON body")
    }

    // Minimal valid DOT graph used in tests.
    fn valid_dot() -> serde_json::Value {
        serde_json::json!({
            "dot": "digraph test {\n  start [shape=Mdiamond]\n  exit  [shape=Msquare]\n  start -> exit\n}",
            "context": {}
        })
    }

    // ---------------------------------------------------------------------------
    // Test 1: create_pipeline_returns_id_and_running
    //
    // POST /pipelines with a valid DOT must return 200 with an `id` string
    // and `status == "running"`.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn create_pipeline_returns_id_and_running() {
        let app = test_app();
        let resp = post_json(app, "/pipelines", valid_dot()).await;

        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_json(resp).await;
        assert!(
            json["id"].as_str().is_some(),
            "response must include a string `id`; got: {json}"
        );
        assert_eq!(
            json["status"], "running",
            "status must be 'running' immediately after creation; got: {json}"
        );
    }

    // ---------------------------------------------------------------------------
    // Test 2: create_pipeline_invalid_dot_returns_400
    //
    // POST /pipelines with syntactically invalid DOT must return 400 with
    // error code PARSE_ERROR.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn create_pipeline_invalid_dot_returns_400() {
        let app = test_app();
        let body = serde_json::json!({
            "dot": "this is absolutely not valid graphviz dot syntax !!!",
            "context": {}
        });
        let resp = post_json(app, "/pipelines", body).await;

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

    // ---------------------------------------------------------------------------
    // Test 3: get_pipeline_not_found_returns_404
    //
    // GET /pipelines/{nonexistent-id} must return 404 with PIPELINE_NOT_FOUND.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn get_pipeline_not_found_returns_404() {
        let app = test_app();
        let resp = get_req(app, "/pipelines/nonexistent-pipeline-id-xyz").await;

        assert_eq!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "unknown pipeline must return 404"
        );

        let json = body_json(resp).await;
        assert_eq!(
            json["error"]["code"], "PIPELINE_NOT_FOUND",
            "error code must be PIPELINE_NOT_FOUND; got: {json}"
        );
    }

    // ---------------------------------------------------------------------------
    // Test: get_questions_empty_for_new_pipeline
    //
    // GET /pipelines/{id}/questions on a freshly-created pipeline must return
    // 200 with an empty `questions` array (no human-in-the-loop questions yet).
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn get_questions_empty_for_new_pipeline() {
        let state = AppState::with_temp_dir();
        let app = router().with_state(state);

        // Create a pipeline first.
        let create_resp = post_json(app.clone(), "/pipelines", valid_dot()).await;
        assert_eq!(create_resp.status(), StatusCode::OK);
        let create_json = body_json(create_resp).await;
        let id = create_json["id"].as_str().expect("id string").to_string();

        // GET /pipelines/{id}/questions — should return 200 with empty list.
        let resp = get_req(app, &format!("/pipelines/{id}/questions")).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "GET /pipelines/{{id}}/questions must return 200"
        );
        let json = body_json(resp).await;
        assert!(
            json["questions"].is_array(),
            "response must have a `questions` array; got: {json}"
        );
        assert_eq!(
            json["questions"].as_array().unwrap().len(),
            0,
            "new pipeline must have no pending questions; got: {json}"
        );
    }

    // ---------------------------------------------------------------------------
    // Test: post_answer_not_found
    //
    // POST /pipelines/{id}/questions/{qid}/answer with an unknown qid must
    // return 404 with error code QUESTION_NOT_FOUND.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn post_answer_not_found() {
        let state = AppState::with_temp_dir();
        let app = router().with_state(state);

        // Create a pipeline first.
        let create_resp = post_json(app.clone(), "/pipelines", valid_dot()).await;
        assert_eq!(create_resp.status(), StatusCode::OK);
        let create_json = body_json(create_resp).await;
        let id = create_json["id"].as_str().expect("id string").to_string();

        // POST an answer for a non-existent qid.
        let resp = post_json(
            app,
            &format!("/pipelines/{id}/questions/nonexistent-qid/answer"),
            serde_json::json!({ "answer": "yes" }),
        )
        .await;

        assert_eq!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "unknown qid must return 404"
        );
        let json = body_json(resp).await;
        assert_eq!(
            json["error"]["code"], "QUESTION_NOT_FOUND",
            "error code must be QUESTION_NOT_FOUND; got: {json}"
        );
    }

    // ---------------------------------------------------------------------------
    // Test: get_graph_returns_dot_json
    //
    // GET /pipelines/{id}/graph on a freshly-created pipeline must return 200
    // with a JSON body containing `dot` (string) and `format: "dot"`.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn get_graph_returns_dot_json() {
        let state = AppState::with_temp_dir();
        let app = router().with_state(state);

        // Create a pipeline first.
        let create_resp = post_json(app.clone(), "/pipelines", valid_dot()).await;
        assert_eq!(create_resp.status(), StatusCode::OK);
        let create_json = body_json(create_resp).await;
        let id = create_json["id"].as_str().expect("id string").to_string();

        // GET /pipelines/{id}/graph — must return 200 with dot and format fields.
        let resp = get_req(app, &format!("/pipelines/{id}/graph")).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "GET /pipelines/{{id}}/graph must return 200"
        );
        let json = body_json(resp).await;
        assert!(
            json["dot"].as_str().is_some(),
            "response must include a string `dot` field; got: {json}"
        );
        assert_eq!(
            json["format"], "dot",
            "response `format` must be 'dot' by default; got: {json}"
        );
    }

    // ---------------------------------------------------------------------------
    // Test: get_checkpoint_before_any_written
    //
    // GET /pipelines/{id}/checkpoint before the pipeline has written a checkpoint
    // must return 200 with `{checkpoint: null}`.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn get_checkpoint_before_any_written() {
        let state = AppState::with_temp_dir();
        let app = router().with_state(state);

        // Create a pipeline first.
        let create_resp = post_json(app.clone(), "/pipelines", valid_dot()).await;
        assert_eq!(create_resp.status(), StatusCode::OK);
        let create_json = body_json(create_resp).await;
        let id = create_json["id"].as_str().expect("id string").to_string();

        // GET /pipelines/{id}/checkpoint — should return 200 with {checkpoint: null}.
        let resp = get_req(app, &format!("/pipelines/{id}/checkpoint")).await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "GET /pipelines/{{id}}/checkpoint must return 200 even when no checkpoint written"
        );
        let json = body_json(resp).await;
        assert!(
            json["checkpoint"].is_null(),
            "checkpoint must be null before any checkpoint is written; got: {json}"
        );
    }

    // ---------------------------------------------------------------------------
    // Test: get_events_returns_event_stream_content_type
    //
    // GET /pipelines/{id}/events on an existing pipeline must return 200 with
    // Content-Type: text/event-stream (SSE wire format).
    //
    // NOTE: We intentionally do NOT collect the body because the SSE live
    // stream never closes — checking headers is sufficient for this test.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn get_events_returns_event_stream_content_type() {
        let state = AppState::with_temp_dir();
        let app = router().with_state(state);

        // Create a pipeline first.
        let create_resp = post_json(app.clone(), "/pipelines", valid_dot()).await;
        assert_eq!(create_resp.status(), StatusCode::OK);
        let create_json = body_json(create_resp).await;
        let id = create_json["id"].as_str().expect("id string").to_string();

        // GET /pipelines/{id}/events — must return 200 with text/event-stream.
        let resp = get_req(app, &format!("/pipelines/{id}/events")).await;

        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "GET /pipelines/{{id}}/events must return 200"
        );

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

    // ---------------------------------------------------------------------------
    // Test: list_pipelines_returns_array
    //
    // POST /pipelines twice, then GET /pipelines must return 200 with a JSON
    // array of length 2 where each entry has id, status, and started_at fields.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn list_pipelines_returns_array() {
        let state = AppState::with_temp_dir();
        let app = router().with_state(state);

        // Create two pipelines.
        let resp1 = post_json(app.clone(), "/pipelines", valid_dot()).await;
        assert_eq!(resp1.status(), StatusCode::OK, "first POST must return 200");

        let resp2 = post_json(app.clone(), "/pipelines", valid_dot()).await;
        assert_eq!(
            resp2.status(),
            StatusCode::OK,
            "second POST must return 200"
        );

        // GET /pipelines — must return 200 with a JSON array of length 2.
        let resp = get_req(app, "/pipelines").await;
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "GET /pipelines must return 200"
        );
        let json = body_json(resp).await;
        assert!(
            json.is_array(),
            "GET /pipelines must return a JSON array; got: {json}"
        );
        let arr = json.as_array().unwrap();
        assert_eq!(
            arr.len(),
            2,
            "GET /pipelines must return 2 pipelines; got: {arr:?}"
        );
        for entry in arr {
            assert!(
                entry["id"].as_str().is_some(),
                "each pipeline entry must have an `id` string; got: {entry}"
            );
            assert!(
                entry["status"].as_str().is_some(),
                "each pipeline entry must have a `status` string; got: {entry}"
            );
            assert!(
                entry["started_at"].as_str().is_some(),
                "each pipeline entry must have a `started_at` string; got: {entry}"
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Test 4: create_then_get_pipeline
    //
    // POST /pipelines followed by GET /pipelines/{id} must return the same
    // `id` and a valid `started_at`.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn create_then_get_pipeline() {
        // Each test gets its own AppState so state is isolated.
        let state = AppState::with_temp_dir();
        let app = router().with_state(state);

        // Create the pipeline.
        let create_resp = post_json(app.clone(), "/pipelines", valid_dot()).await;
        assert_eq!(create_resp.status(), StatusCode::OK);
        let create_json = body_json(create_resp).await;

        let id = create_json["id"]
            .as_str()
            .expect("create response must have an `id`")
            .to_string();

        // Retrieve the pipeline status.
        let get_resp = get_req(app, &format!("/pipelines/{id}")).await;
        assert_eq!(
            get_resp.status(),
            StatusCode::OK,
            "GET /pipelines/{id} must return 200"
        );

        let get_json = body_json(get_resp).await;
        assert_eq!(
            get_json["id"], id,
            "get response id must match create response id"
        );
        assert!(
            get_json["started_at"].as_str().is_some(),
            "get response must include a string `started_at`; got: {get_json}"
        );
        // Status is either running or completed (small graph finishes fast).
        let status = get_json["status"]
            .as_str()
            .expect("status must be a string");
        assert!(
            status == "running" || status == "completed",
            "status must be running or completed shortly after creation; got: {status}"
        );
    }
}
