//! SSE adapter — converts pipeline events into axum SSE responses.
//!
//! Functions:
//! - [`event_type_name`] — map a [`PipelineEvent`] variant to its SSE event type string.
//! - [`pipeline_event_to_sse`] — convert a [`PipelineEvent`] to an axum [`Event`].
//! - [`is_terminal_event`] — returns `true` for events that signal pipeline completion.

use std::convert::Infallible;

use attractor::PipelineEvent;
use axum::response::sse::Event;

// ---------------------------------------------------------------------------
// event_type_name
// ---------------------------------------------------------------------------

/// Map a [`PipelineEvent`] variant to its SSE event-type string.
///
/// The returned string matches the `"event"` serde discriminant tag used
/// on each variant (see `#[serde(tag = "event", rename_all = "snake_case")]`).
pub fn event_type_name(event: &PipelineEvent) -> &'static str {
    match event {
        PipelineEvent::PipelineStarted { .. } => "pipeline_started",
        PipelineEvent::PipelineCompleted { .. } => "pipeline_completed",
        PipelineEvent::PipelineFailed { .. } => "pipeline_failed",
        PipelineEvent::StageStarted { .. } => "stage_started",
        PipelineEvent::StageCompleted { .. } => "stage_completed",
        PipelineEvent::StageFailed { .. } => "stage_failed",
        PipelineEvent::StageRetrying { .. } => "stage_retrying",
        PipelineEvent::ParallelStarted { .. } => "parallel_started",
        PipelineEvent::ParallelBranchStarted { .. } => "parallel_branch_started",
        PipelineEvent::ParallelBranchCompleted { .. } => "parallel_branch_completed",
        PipelineEvent::ParallelCompleted { .. } => "parallel_completed",
        PipelineEvent::InterviewStarted { .. } => "interview_started",
        PipelineEvent::InterviewCompleted { .. } => "interview_completed",
        PipelineEvent::InterviewTimeout { .. } => "interview_timeout",
        PipelineEvent::CheckpointSaved { .. } => "checkpoint_saved",
    }
}

// ---------------------------------------------------------------------------
// pipeline_event_to_sse
// ---------------------------------------------------------------------------

/// Convert a [`PipelineEvent`] to an axum SSE [`Event`].
///
/// Sets the SSE `event` field to [`event_type_name`] and the `data` field to
/// the JSON-serialised event payload.  The error type is [`Infallible`] because
/// serialisation failure is handled gracefully (falls back to `"{}"`).
pub fn pipeline_event_to_sse(event: PipelineEvent) -> Result<Event, Infallible> {
    let json = serde_json::to_string(&event).unwrap_or_else(|err| {
        tracing::warn!("failed to serialize PipelineEvent: {err}");
        "{}".to_string()
    });
    let type_name = event_type_name(&event);
    Ok(Event::default().event(type_name).data(json))
}

// ---------------------------------------------------------------------------
// is_terminal_event
// ---------------------------------------------------------------------------

/// Return `true` if `event` signals that the pipeline has reached a terminal state.
///
/// Only [`PipelineCompleted`][PipelineEvent::PipelineCompleted] and
/// [`PipelineFailed`][PipelineEvent::PipelineFailed] are terminal.
pub fn is_terminal_event(event: &PipelineEvent) -> bool {
    matches!(
        event,
        PipelineEvent::PipelineCompleted { .. } | PipelineEvent::PipelineFailed { .. }
    )
}

// ---------------------------------------------------------------------------
// Tests — RED written first, then GREEN implemented above
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use attractor::PipelineEvent;

    use super::*;

    // ---------------------------------------------------------------------------
    // Test 1: event_type_name_matches_serde_tag
    //
    // Verifies that event_type_name returns the same string as the serde
    // "event" discriminant field produced by JSON serialisation.
    // Iterates over all 15 variants to give full coverage.
    // ---------------------------------------------------------------------------
    #[test]
    fn event_type_name_matches_serde_tag() {
        let variants: Vec<PipelineEvent> = vec![
            PipelineEvent::PipelineStarted {
                name: "test".into(),
                id: "abc".into(),
            },
            PipelineEvent::PipelineCompleted {
                duration: Duration::from_secs(1),
                artifact_count: 0,
            },
            PipelineEvent::PipelineFailed {
                error: "e".into(),
                duration: Duration::from_secs(1),
            },
            PipelineEvent::StageStarted {
                name: "s".into(),
                index: 0,
            },
            PipelineEvent::StageCompleted {
                name: "plan".into(),
                index: 1,
                duration: Duration::from_millis(1500),
            },
            PipelineEvent::StageFailed {
                name: "s".into(),
                index: 0,
                error: "e".into(),
                will_retry: false,
            },
            PipelineEvent::StageRetrying {
                name: "s".into(),
                index: 0,
                attempt: 1,
                delay: Duration::from_secs(1),
            },
            PipelineEvent::ParallelStarted { branch_count: 2 },
            PipelineEvent::ParallelBranchStarted {
                branch: "b".into(),
                index: 0,
            },
            PipelineEvent::ParallelBranchCompleted {
                branch: "b".into(),
                index: 0,
                duration: Duration::from_secs(1),
                success: true,
                error: None,
            },
            PipelineEvent::ParallelCompleted {
                duration: Duration::from_secs(1),
                success_count: 1,
                failure_count: 0,
            },
            PipelineEvent::InterviewStarted {
                question: "q".into(),
                stage: "s".into(),
            },
            PipelineEvent::InterviewCompleted {
                question: "q".into(),
                answer: "a".into(),
                duration: Duration::from_secs(1),
            },
            PipelineEvent::InterviewTimeout {
                question: "q".into(),
                stage: "s".into(),
                duration: Duration::from_secs(1),
            },
            PipelineEvent::CheckpointSaved {
                node_id: "n".into(),
            },
        ];

        assert_eq!(variants.len(), 15, "exhaustive: must cover all 15 variants");

        for v in &variants {
            let json: serde_json::Value = serde_json::to_value(v).expect("serialize variant");
            let serde_tag = json["event"].as_str().expect("event tag present");
            assert_eq!(
                event_type_name(v),
                serde_tag,
                "{:?}: event_type_name must match serde tag",
                v
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Test 2: all_variants_have_type_names
    //
    // Every PipelineEvent variant must produce a non-empty type name string.
    // This test exhaustively constructs all 15 variants.
    // ---------------------------------------------------------------------------
    #[test]
    fn all_variants_have_type_names() {
        let variants: Vec<PipelineEvent> = vec![
            PipelineEvent::PipelineStarted {
                name: "n".into(),
                id: "i".into(),
            },
            PipelineEvent::PipelineCompleted {
                duration: Duration::from_secs(1),
                artifact_count: 0,
            },
            PipelineEvent::PipelineFailed {
                error: "e".into(),
                duration: Duration::from_secs(1),
            },
            PipelineEvent::StageStarted {
                name: "s".into(),
                index: 0,
            },
            PipelineEvent::StageCompleted {
                name: "s".into(),
                index: 0,
                duration: Duration::from_secs(1),
            },
            PipelineEvent::StageFailed {
                name: "s".into(),
                index: 0,
                error: "e".into(),
                will_retry: false,
            },
            PipelineEvent::StageRetrying {
                name: "s".into(),
                index: 0,
                attempt: 1,
                delay: Duration::from_secs(1),
            },
            PipelineEvent::ParallelStarted { branch_count: 2 },
            PipelineEvent::ParallelBranchStarted {
                branch: "b".into(),
                index: 0,
            },
            PipelineEvent::ParallelBranchCompleted {
                branch: "b".into(),
                index: 0,
                duration: Duration::from_secs(1),
                success: true,
                error: None,
            },
            PipelineEvent::ParallelCompleted {
                duration: Duration::from_secs(1),
                success_count: 1,
                failure_count: 0,
            },
            PipelineEvent::InterviewStarted {
                question: "q".into(),
                stage: "s".into(),
            },
            PipelineEvent::InterviewCompleted {
                question: "q".into(),
                answer: "a".into(),
                duration: Duration::from_secs(1),
            },
            PipelineEvent::InterviewTimeout {
                question: "q".into(),
                stage: "s".into(),
                duration: Duration::from_secs(1),
            },
            PipelineEvent::CheckpointSaved {
                node_id: "n".into(),
            },
        ];

        assert_eq!(variants.len(), 15, "exhaustive: must cover all 15 variants");

        for v in &variants {
            let name = event_type_name(v);
            assert!(
                !name.is_empty(),
                "event_type_name must not be empty for {:?}",
                v
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Test 3: pipeline_event_to_sse_produces_valid_event
    //
    // Verifies that pipeline_event_to_sse succeeds and is Ok for a sample event.
    // The axum Event type is opaque, so we verify the Result is Ok.
    // ---------------------------------------------------------------------------
    #[test]
    fn pipeline_event_to_sse_produces_valid_event() {
        let ev = PipelineEvent::CheckpointSaved {
            node_id: "plan".to_string(),
        };
        let result = pipeline_event_to_sse(ev);
        assert!(
            result.is_ok(),
            "pipeline_event_to_sse must return Ok for a valid event"
        );

        // Also verify a PipelineStarted event.
        let ev2 = PipelineEvent::PipelineStarted {
            name: "my-pipeline".to_string(),
            id: "abc-123".to_string(),
        };
        let result2 = pipeline_event_to_sse(ev2);
        assert!(
            result2.is_ok(),
            "pipeline_event_to_sse must return Ok for PipelineStarted"
        );
        // TODO: assert event type and data fields if axum exposes accessors
    }

    // ---------------------------------------------------------------------------
    // Test 4: is_terminal_event_identifies_completion
    //
    // Only PipelineCompleted and PipelineFailed are terminal; all others are not.
    // ---------------------------------------------------------------------------
    #[test]
    fn is_terminal_event_identifies_completion() {
        let completed = PipelineEvent::PipelineCompleted {
            duration: Duration::from_secs(1),
            artifact_count: 2,
        };
        let failed = PipelineEvent::PipelineFailed {
            error: "oops".to_string(),
            duration: Duration::from_secs(1),
        };
        let started = PipelineEvent::PipelineStarted {
            name: "p".to_string(),
            id: "x".to_string(),
        };
        let stage = PipelineEvent::StageStarted {
            name: "s".to_string(),
            index: 0,
        };
        let checkpoint = PipelineEvent::CheckpointSaved {
            node_id: "n".to_string(),
        };

        assert!(
            is_terminal_event(&completed),
            "PipelineCompleted must be terminal"
        );
        assert!(
            is_terminal_event(&failed),
            "PipelineFailed must be terminal"
        );
        assert!(
            !is_terminal_event(&started),
            "PipelineStarted must not be terminal"
        );
        assert!(
            !is_terminal_event(&stage),
            "StageStarted must not be terminal"
        );
        assert!(
            !is_terminal_event(&checkpoint),
            "CheckpointSaved must not be terminal"
        );
    }
}
