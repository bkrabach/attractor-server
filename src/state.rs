//! Shared server state for managing concurrent pipeline runs.
//!
//! [`AppState`] is the top-level handle shared across all Axum handlers.
//! It holds a map of active (and recently completed) [`PipelineHandle`]s.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio::sync::{Mutex, RwLock, broadcast};
use tokio::task::AbortHandle;

use attractor::{Graph, PipelineEvent};

use crate::interviewer::InterviewerState;

// ---------------------------------------------------------------------------
// PipelineStatus
// ---------------------------------------------------------------------------

/// The lifecycle state of a pipeline run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStatus {
    /// The pipeline is currently executing.
    Running,
    /// The pipeline finished successfully.
    Completed,
    /// The pipeline terminated with an error.
    Failed,
    /// The pipeline was cancelled by an operator.
    Cancelled,
}

// ---------------------------------------------------------------------------
// PipelineInner — mutable interior protected by a Mutex
// ---------------------------------------------------------------------------

/// The mutable interior state of a pipeline run.
///
/// All access is protected by the [`Mutex`] on [`PipelineHandle::inner`].
pub struct PipelineInner {
    /// Current lifecycle status.
    pub status: PipelineStatus,
    /// Ordered log of pipeline events (capped at [`EVENT_LOG_CAPACITY`]).
    pub event_log: Vec<PipelineEvent>,
    /// Node IDs that have completed execution.
    pub completed_nodes: Vec<String>,
    /// The node currently being executed, if any.
    pub current_node: Option<String>,
}

/// Maximum number of events retained in the in-memory event log.
const EVENT_LOG_CAPACITY: usize = 10_000;

impl PipelineInner {
    /// Create a new [`PipelineInner`] starting in the [`PipelineStatus::Running`] state.
    pub fn new() -> Self {
        PipelineInner {
            status: PipelineStatus::Running,
            event_log: Vec::new(),
            completed_nodes: Vec::new(),
            current_node: None,
        }
    }

    /// Append `event` to the log.
    ///
    /// Once the log reaches [`EVENT_LOG_CAPACITY`] entries, additional events
    /// are silently dropped to prevent unbounded memory growth.
    pub fn push_event(&mut self, event: PipelineEvent) {
        if self.event_log.len() < EVENT_LOG_CAPACITY {
            self.event_log.push(event);
        }
    }
}

impl Default for PipelineInner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PipelineHandle
// ---------------------------------------------------------------------------

/// A handle to a running (or recently completed) pipeline.
///
/// Immutable fields are written once at spawn time; mutable state lives in
/// [`PipelineHandle::inner`] behind an async [`Mutex`].
pub struct PipelineHandle {
    // -- Immutable identity --
    /// Unique pipeline run identifier (UUID v4 string).
    pub id: String,
    /// The raw DOT source that was submitted.
    pub dot_source: String,
    /// Parsed graph representation of the DOT source.
    pub graph: Graph,

    // -- Channels / state --
    /// Broadcast sender for live [`PipelineEvent`] streaming.
    pub event_tx: broadcast::Sender<PipelineEvent>,
    /// Human-in-the-loop state for this pipeline.
    pub interviewer_state: Arc<InterviewerState>,

    // -- Filesystem --
    /// Root directory for this pipeline's log files.
    pub logs_root: PathBuf,

    // -- Metadata --
    /// When the pipeline was spawned.
    pub started_at: DateTime<Utc>,

    // -- Lifecycle control --
    /// Handle used to abort the pipeline's spawned task.
    pub abort_handle: AbortHandle,

    // -- Mutable interior --
    /// Mutable runtime state (status, event log, node tracking).
    ///
    /// Wrapped in `Arc` so spawned tasks can hold a reference without
    /// borrowing the `PipelineHandle` itself.
    pub inner: Arc<Mutex<PipelineInner>>,
}

// ---------------------------------------------------------------------------
// AppState
// ---------------------------------------------------------------------------

/// Top-level shared server state, cheap to clone via `Arc` interiors.
#[derive(Clone)]
pub struct AppState {
    /// Active and recently completed pipeline runs, keyed by pipeline ID.
    pub pipelines: Arc<RwLock<HashMap<String, Arc<PipelineHandle>>>>,
    /// Root directory for server data (logs, checkpoints, etc.).
    pub data_dir: PathBuf,
}

impl AppState {
    /// Create a new [`AppState`] backed by `data_dir`.
    pub fn new(data_dir: PathBuf) -> Self {
        AppState {
            pipelines: Arc::new(RwLock::new(HashMap::new())),
            data_dir,
        }
    }

    /// Create an [`AppState`] backed by a fresh temporary directory.
    ///
    /// Useful in tests and local development.  The directory persists until
    /// the OS cleans up `/tmp` (no automatic removal on drop).
    pub fn with_temp_dir() -> Self {
        let id = uuid::Uuid::new_v4();
        let data_dir = std::env::temp_dir().join(format!("attractor-server-{id}"));
        std::fs::create_dir_all(&data_dir).expect("create temp data dir");
        Self::new(data_dir)
    }

    /// Insert a [`PipelineHandle`] into the map.
    pub async fn insert_pipeline(&self, handle: Arc<PipelineHandle>) {
        self.pipelines
            .write()
            .await
            .insert(handle.id.clone(), handle);
    }

    /// Look up a pipeline by ID.  Returns `None` if not found.
    pub async fn get_pipeline(&self, id: &str) -> Option<Arc<PipelineHandle>> {
        self.pipelines.read().await.get(id).cloned()
    }

    /// Return the logs root directory for `pipeline_id`.
    ///
    /// This is `<data_dir>/logs/<pipeline_id>`.  The directory is **not**
    /// created by this method.
    pub fn logs_root_for(&self, pipeline_id: &str) -> PathBuf {
        self.data_dir.join("logs").join(pipeline_id)
    }
}

// ---------------------------------------------------------------------------
// Tests — RED written first, then GREEN implemented above
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------------------
    // Helper: build a minimal PipelineHandle for testing
    // ---------------------------------------------------------------------------
    fn make_handle(id: &str, app_state: &AppState) -> Arc<PipelineHandle> {
        use crate::interviewer::InterviewerState;
        use attractor::events::event_channel;

        let (event_tx, _rx) = event_channel();
        let interviewer_state = Arc::new(InterviewerState::new());
        let logs_root = app_state.logs_root_for(id);
        let abort_handle = tokio::spawn(async {
            tokio::time::sleep(std::time::Duration::MAX).await;
        })
        .abort_handle();

        Arc::new(PipelineHandle {
            id: id.to_string(),
            dot_source: "digraph {}".to_string(),
            graph: Graph::new(id.to_string()),
            event_tx,
            interviewer_state,
            logs_root,
            started_at: Utc::now(),
            abort_handle,
            inner: Arc::new(Mutex::new(PipelineInner::new())),
        })
    }

    // ---------------------------------------------------------------------------
    // Test 1: insert_and_get_pipeline
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn insert_and_get_pipeline() {
        let state = AppState::with_temp_dir();
        let handle = make_handle("pipe-abc", &state);
        state.insert_pipeline(handle.clone()).await;

        let got = state.get_pipeline("pipe-abc").await;
        assert!(got.is_some(), "expected to retrieve inserted pipeline");
        assert_eq!(got.unwrap().id, "pipe-abc");
    }

    // ---------------------------------------------------------------------------
    // Test 2: get_missing_pipeline_returns_none
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn get_missing_pipeline_returns_none() {
        let state = AppState::with_temp_dir();
        let got = state.get_pipeline("does-not-exist").await;
        assert!(got.is_none(), "missing pipeline must return None");
    }

    // ---------------------------------------------------------------------------
    // Test 3: pipeline_inner_starts_as_running
    // ---------------------------------------------------------------------------
    #[test]
    fn pipeline_inner_starts_as_running() {
        let inner = PipelineInner::new();
        assert_eq!(inner.status, PipelineStatus::Running);
        assert!(inner.event_log.is_empty());
        assert!(inner.completed_nodes.is_empty());
        assert!(inner.current_node.is_none());
    }

    // ---------------------------------------------------------------------------
    // Test 4: event_log_bounded_at_10000
    // ---------------------------------------------------------------------------
    #[test]
    fn event_log_bounded_at_10000() {
        let mut inner = PipelineInner::new();
        let event = PipelineEvent::PipelineStarted {
            name: "x".to_string(),
            id: "y".to_string(),
        };

        // Push 10,001 events — the log must cap at exactly 10,000.
        for _ in 0..=10_000 {
            inner.push_event(event.clone());
        }
        assert_eq!(
            inner.event_log.len(),
            10_000,
            "event_log must be bounded at exactly 10,000 entries"
        );
    }

    // ---------------------------------------------------------------------------
    // Test 5: pipeline_status_serializes_snake_case
    // ---------------------------------------------------------------------------
    #[test]
    fn pipeline_status_serializes_snake_case() {
        let cases = [
            (PipelineStatus::Running, "\"running\""),
            (PipelineStatus::Completed, "\"completed\""),
            (PipelineStatus::Failed, "\"failed\""),
            (PipelineStatus::Cancelled, "\"cancelled\""),
        ];
        for (status, expected) in cases {
            let json = serde_json::to_string(&status).expect("serialize PipelineStatus");
            assert_eq!(
                json, expected,
                "PipelineStatus::{:?} must serialize as {}",
                status, expected
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Test 6: logs_root_for_pipeline
    // ---------------------------------------------------------------------------
    #[test]
    fn logs_root_for_pipeline() {
        let data_dir = std::path::PathBuf::from("/tmp/test-data");
        let state = AppState::new(data_dir.clone());
        let logs_root = state.logs_root_for("pipe-xyz");
        assert_eq!(logs_root, data_dir.join("logs").join("pipe-xyz"));
    }
}
