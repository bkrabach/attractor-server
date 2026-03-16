//! Server-side state for human-in-the-loop question/answer flows.
//!
//! [`InterviewerState`] holds pending questions and the one-shot senders
//! used to deliver answers back to the waiting pipeline handler.
//!
//! [`PendingQuestion`] lives here (rather than in `state`) to avoid a
//! circular module dependency: `state` imports `InterviewerState`, so
//! `PendingQuestion` must be defined before `state` can reference it.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::Serialize;
use tokio::sync::oneshot;
use tokio::time::{Duration, timeout};
use uuid::Uuid;

use attractor::{Answer, Interviewer, Question};

// ---------------------------------------------------------------------------
// PendingQuestion
// ---------------------------------------------------------------------------

/// A human-in-the-loop question that has been submitted to the pipeline
/// but has not yet received an answer.
#[derive(Debug, Clone, Serialize)]
pub struct PendingQuestion {
    /// Unique identifier for this question (UUID v4 string).
    pub id: String,
    /// The question presented to the human.
    pub question: Question,
    /// When the question was created.
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// InterviewerState
// ---------------------------------------------------------------------------

/// Shared state for the human interview flow.
///
/// When a pipeline node calls `WaitForHumanHandler`, it inserts a
/// [`PendingQuestion`] here and waits on a [`oneshot::Receiver<Answer>`].
/// The HTTP endpoint that receives the human's answer sends on the
/// corresponding [`oneshot::Sender<Answer>`] to unblock the pipeline.
pub struct InterviewerState {
    /// Questions waiting for a human answer, keyed by question ID.
    pub pending: Mutex<HashMap<String, PendingQuestion>>,
    /// One-shot senders to deliver answers, keyed by question ID.
    pub answer_senders: Mutex<HashMap<String, oneshot::Sender<Answer>>>,
    /// IDs of questions that have already been answered (sender consumed).
    answered_ids: Mutex<HashSet<String>>,
}

impl InterviewerState {
    /// Create a new, empty [`InterviewerState`].
    pub fn new() -> Self {
        InterviewerState {
            pending: Mutex::new(HashMap::new()),
            answer_senders: Mutex::new(HashMap::new()),
            answered_ids: Mutex::new(HashSet::new()),
        }
    }

    /// Return a snapshot of all pending questions.
    ///
    /// The returned `Vec` is a point-in-time copy; it will not reflect
    /// subsequent insertions or removals.
    pub fn pending_questions(&self) -> Vec<PendingQuestion> {
        self.pending
            .lock()
            .expect("pending lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    /// Submit an answer for a pending question.
    ///
    /// On success:
    /// - removes the sender (consumes it),
    /// - removes the question from `pending`,
    /// - delivers the answer through the oneshot channel.
    ///
    /// # Errors
    ///
    /// Returns `Err("not_found")` if `question_id` is unknown.
    /// Returns `Err("already_answered")` if the sender was already consumed
    /// (this can happen if a concurrent call wins the race).
    pub fn submit_answer(&self, question_id: &str, answer: Answer) -> Result<(), &'static str> {
        // Remove the sender — this is the "claim" step.
        let sender = self
            .answer_senders
            .lock()
            .expect("answer_senders lock poisoned")
            .remove(question_id);

        let sender = match sender {
            Some(s) => s,
            None => {
                // Distinguish "never existed" from "already answered".
                let already = self
                    .answered_ids
                    .lock()
                    .expect("answered_ids lock poisoned")
                    .contains(question_id);
                return if already {
                    Err("already_answered")
                } else {
                    Err("not_found")
                };
            }
        };

        // Record that this question has been answered before releasing the
        // pending entry, so a concurrent caller sees "already_answered".
        self.answered_ids
            .lock()
            .expect("answered_ids lock poisoned")
            .insert(question_id.to_string());

        // Remove from the pending snapshot map.
        self.pending
            .lock()
            .expect("pending lock poisoned")
            .remove(question_id);

        // Deliver the answer. If the receiver is gone (pipeline timed out and
        // cleaned up), we silently discard — it's fine, the pipeline already
        // returned Answer::timeout().
        let _ = sender.send(answer);

        Ok(())
    }
}

impl Default for InterviewerState {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HttpInterviewer
// ---------------------------------------------------------------------------

/// Maximum time the pipeline will wait for a human answer (1 hour).
const ASK_TIMEOUT: Duration = Duration::from_secs(3_600);

/// HTTP-backed human-in-the-loop interviewer.
///
/// When `ask()` is called it:
/// 1. Generates a UUID question ID.
/// 2. Creates a [`oneshot`] channel.
/// 3. Stores the [`PendingQuestion`] and [`oneshot::Sender`] in
///    [`InterviewerState`].
/// 4. Awaits the answer with a 1-hour timeout.
///
/// Flow:
/// ```text
/// pipeline blocks on ask()
///     → browser polls  GET  /pipelines/{id}/questions
///     → browser submits POST /pipelines/{id}/questions/{qid}/answer
///     → ask() returns
/// ```
///
/// On timeout or channel close the method cleans up its state entries and
/// returns [`Answer::timeout()`].
#[derive(Clone)]
pub struct HttpInterviewer {
    state: Arc<InterviewerState>,
}

impl HttpInterviewer {
    /// Create a new `HttpInterviewer` backed by `state`.
    pub fn new(state: Arc<InterviewerState>) -> Self {
        HttpInterviewer { state }
    }

    /// Return a clone of the shared [`InterviewerState`].
    pub fn state(&self) -> Arc<InterviewerState> {
        self.state.clone()
    }
}

#[async_trait]
impl Interviewer for HttpInterviewer {
    async fn ask(&self, question: Question) -> Answer {
        let question_id = Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel::<Answer>();

        let pending = PendingQuestion {
            id: question_id.clone(),
            question,
            created_at: Utc::now(),
        };

        // Register the question and its answer channel.
        self.state
            .pending
            .lock()
            .expect("pending lock poisoned")
            .insert(question_id.clone(), pending);

        self.state
            .answer_senders
            .lock()
            .expect("answer_senders lock poisoned")
            .insert(question_id.clone(), tx);

        // Block until answered or timed out.
        match timeout(ASK_TIMEOUT, rx).await {
            Ok(Ok(answer)) => answer,
            _ => {
                // Timeout or receiver error — clean up and signal timeout.
                self.state
                    .pending
                    .lock()
                    .expect("pending lock poisoned")
                    .remove(&question_id);

                self.state
                    .answer_senders
                    .lock()
                    .expect("answer_senders lock poisoned")
                    .remove(&question_id);

                Answer::timeout()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use attractor::{AnswerValue, QuestionType};
    use std::collections::HashMap;

    fn make_question() -> Question {
        Question {
            text: "Are you ready?".to_string(),
            question_type: QuestionType::Confirmation,
            options: vec![],
            default: None,
            timeout: None,
            stage: "test-stage".to_string(),
            metadata: HashMap::new(),
        }
    }

    // ---------------------------------------------------------------------------
    // Test 1: ask_and_answer_flow
    //
    // Spawn ask() in a background task, verify the question appears in
    // pending_questions(), submit the answer, and verify ask() returns it.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn ask_and_answer_flow() {
        let state = Arc::new(InterviewerState::new());
        let interviewer = HttpInterviewer::new(state.clone());

        // Spawn ask() in the background — it will block until answered.
        let ask_handle = tokio::spawn({
            let interviewer = interviewer.clone();
            async move { interviewer.ask(make_question()).await }
        });

        // Give the background task a moment to register the question.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Verify the question is visible in pending_questions().
        let pending = state.pending_questions();
        assert_eq!(pending.len(), 1, "one question should be pending");
        let question_id = pending[0].id.clone();

        // Submit the answer.
        let result = state.submit_answer(&question_id, Answer::yes());
        assert!(result.is_ok(), "submit_answer should succeed");

        // The background ask() should return the answer we sent.
        let answer = ask_handle.await.expect("ask task panicked");
        assert_eq!(answer.value, AnswerValue::Yes);

        // Pending map must be empty now.
        assert!(
            state.pending_questions().is_empty(),
            "pending should be empty after answer"
        );
    }

    // ---------------------------------------------------------------------------
    // Test 2: submit_answer_for_nonexistent_question
    //
    // Submitting an answer for an unknown question_id must return Err("not_found").
    // ---------------------------------------------------------------------------
    #[test]
    fn submit_answer_for_nonexistent_question() {
        let state = InterviewerState::new();
        let result = state.submit_answer("nonexistent-id", Answer::yes());
        assert_eq!(result, Err("not_found"));
    }

    // ---------------------------------------------------------------------------
    // Test: submit_answer_for_already_answered_question
    //
    // Answering a question a second time must return Err("already_answered"),
    // not Err("not_found").
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn submit_answer_for_already_answered_question() {
        let state = Arc::new(InterviewerState::new());
        let interviewer = HttpInterviewer::new(state.clone());

        // Spawn ask() so a question is registered.
        let ask_handle = tokio::spawn({
            let interviewer = interviewer.clone();
            async move { interviewer.ask(make_question()).await }
        });

        tokio::time::sleep(Duration::from_millis(50)).await;

        let pending = state.pending_questions();
        assert_eq!(pending.len(), 1, "one question should be pending");
        let question_id = pending[0].id.clone();

        // First answer succeeds.
        let result = state.submit_answer(&question_id, Answer::yes());
        assert!(result.is_ok(), "first submit_answer should succeed");

        let _ = ask_handle.await;

        // Second answer must return Err("already_answered"), not Err("not_found").
        let result = state.submit_answer(&question_id, Answer::yes());
        assert_eq!(result, Err("already_answered"));
    }

    // ---------------------------------------------------------------------------
    // Test 3: multiple_concurrent_questions
    //
    // Multiple simultaneous ask() calls each block independently.
    // Answering each one by ID releases only that question.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn multiple_concurrent_questions() {
        let state = Arc::new(InterviewerState::new());
        let interviewer = HttpInterviewer::new(state.clone());

        // Spawn three concurrent ask() calls.
        let handle1 = tokio::spawn({
            let iv = interviewer.clone();
            async move { iv.ask(make_question()).await }
        });
        let handle2 = tokio::spawn({
            let iv = interviewer.clone();
            async move { iv.ask(make_question()).await }
        });
        let handle3 = tokio::spawn({
            let iv = interviewer.clone();
            async move { iv.ask(make_question()).await }
        });

        // Wait for all three to register.
        tokio::time::sleep(Duration::from_millis(100)).await;

        let pending = state.pending_questions();
        assert_eq!(pending.len(), 3, "three questions should be pending");

        let ids: Vec<String> = pending.iter().map(|p| p.id.clone()).collect();

        // Answer them in reverse order with distinct answers.
        state
            .submit_answer(&ids[2], Answer::yes())
            .expect("submit q3");
        state
            .submit_answer(&ids[1], Answer::no())
            .expect("submit q2");
        state
            .submit_answer(&ids[0], Answer::yes())
            .expect("submit q1");

        // All three tasks should complete and return distinct answers.
        let a1 = handle1.await.expect("ask1 panicked");
        let a2 = handle2.await.expect("ask2 panicked");
        let a3 = handle3.await.expect("ask3 panicked");

        // Each ask() returned an answer (not timeout).
        assert_ne!(a1.value, AnswerValue::Timeout, "q1 must not timeout");
        assert_ne!(a2.value, AnswerValue::Timeout, "q2 must not timeout");
        assert_ne!(a3.value, AnswerValue::Timeout, "q3 must not timeout");

        // Pending map must be empty.
        assert!(
            state.pending_questions().is_empty(),
            "pending should be empty after all answers"
        );
    }
}
