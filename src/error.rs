//! Server error types with consistent JSON HTTP responses.
//!
//! Every error condition surfaces as `{"error": {"code": "...", "message": "...", "status": N}}`
//! with an optional `diagnostics` field for validation errors.

use attractor::validation::Diagnostic;
use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

// ---------------------------------------------------------------------------
// Helper structs
// ---------------------------------------------------------------------------

/// Top-level JSON error response body.
#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

/// Detail fields within an error response.
#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub code: &'static str,
    pub message: String,
    pub status: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<Diagnostic>>,
}

// ---------------------------------------------------------------------------
// ServerError enum
// ---------------------------------------------------------------------------

/// All error conditions the server can surface.
///
/// Each variant maps to a consistent JSON response body and HTTP status code.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ServerError {
    /// A named pipeline was not found. → 404
    #[error("pipeline not found: {0}")]
    PipelineNotFound(String),

    /// A question ID was not found in the pipeline. → 404
    #[error("question not found: {0}")]
    QuestionNotFound(String),

    /// Submitted DOT source or expression failed to parse. → 400
    #[error("parse error: {0}")]
    ParseError(String),

    /// Submitted DOT graph failed validation lint rules. → 422
    #[error("validation error")]
    ValidationError(Vec<Diagnostic>),

    /// The pipeline exists but is not currently running. → 409
    #[error("pipeline not running: {0}")]
    PipelineNotRunning(String),

    /// An answer was submitted for a question that is already answered. → 409
    #[error("question already answered: {0}")]
    QuestionAlreadyAnswered(String),

    /// The submitted answer value is not valid. → 400
    #[error("invalid answer: {0}")]
    InvalidAnswer(String),

    /// The `dot` / `graphviz` binary is not available on this host. → 501
    #[error("graphviz not available")]
    GraphvizNotAvailable,

    /// A file was not found in the pipeline's working directory. → 404
    #[error("file not found: {0}")]
    FileNotFound(String),

    /// A file path attempted directory traversal (contains `..`). → 400
    #[error("path traversal not allowed")]
    PathTraversal,

    /// A file exceeds the maximum displayable size. → 400
    #[error("file too large: {0} bytes")]
    FileTooLarge(u64),

    /// An unexpected internal server error. → 500
    #[error("internal error: {0}")]
    Internal(String),
}

impl ServerError {
    /// HTTP status code for this error variant.
    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::PipelineNotFound(_) | Self::QuestionNotFound(_) | Self::FileNotFound(_) => {
                StatusCode::NOT_FOUND
            }
            Self::ParseError(_)
            | Self::InvalidAnswer(_)
            | Self::PathTraversal
            | Self::FileTooLarge(_) => StatusCode::BAD_REQUEST,
            Self::ValidationError(_) => StatusCode::UNPROCESSABLE_ENTITY,
            Self::PipelineNotRunning(_) | Self::QuestionAlreadyAnswered(_) => StatusCode::CONFLICT,
            Self::GraphvizNotAvailable => StatusCode::NOT_IMPLEMENTED,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Machine-readable error code string.
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::PipelineNotFound(_) => "PIPELINE_NOT_FOUND",
            Self::QuestionNotFound(_) => "QUESTION_NOT_FOUND",
            Self::ParseError(_) => "PARSE_ERROR",
            Self::ValidationError(_) => "VALIDATION_ERROR",
            Self::PipelineNotRunning(_) => "PIPELINE_NOT_RUNNING",
            Self::QuestionAlreadyAnswered(_) => "QUESTION_ALREADY_ANSWERED",
            Self::InvalidAnswer(_) => "INVALID_ANSWER",
            Self::FileNotFound(_) => "FILE_NOT_FOUND",
            Self::PathTraversal => "PATH_TRAVERSAL",
            Self::FileTooLarge(_) => "FILE_TOO_LARGE",
            Self::GraphvizNotAvailable => "GRAPHVIZ_NOT_AVAILABLE",
            Self::Internal(_) => "INTERNAL_ERROR",
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let code = self.error_code();
        // Extract diagnostics before moving self into to_string().
        let diagnostics = if let Self::ValidationError(ref diags) = self {
            Some(diags.clone())
        } else {
            None
        };
        let message = self.to_string();
        let body = ErrorBody {
            error: ErrorDetail {
                code,
                message,
                status: status.as_u16(),
                diagnostics,
            },
        };
        (status, Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use attractor::validation::{Diagnostic, Severity};
    use http_body_util::BodyExt;

    /// Consume an axum `Response`, parse the body as JSON.
    async fn body_json(err: ServerError) -> serde_json::Value {
        let response = err.into_response();
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("body collect")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("valid JSON body")
    }

    #[tokio::test]
    async fn pipeline_not_found_produces_404_json() {
        let err = ServerError::PipelineNotFound("pipe-1".into());
        // status_code() helper
        assert_eq!(err.status_code(), StatusCode::NOT_FOUND);

        // Full round-trip through IntoResponse
        let json = body_json(ServerError::PipelineNotFound("pipe-1".into())).await;
        assert_eq!(json["error"]["code"], "PIPELINE_NOT_FOUND");
        assert_eq!(json["error"]["status"], 404);
        assert!(json["error"]["message"].as_str().is_some());
    }

    #[tokio::test]
    async fn parse_error_produces_400_json() {
        let json = body_json(ServerError::ParseError("unexpected `}`".into())).await;
        assert_eq!(json["error"]["code"], "PARSE_ERROR");
        assert_eq!(json["error"]["status"], 400);
    }

    #[tokio::test]
    async fn validation_error_includes_diagnostics() {
        let diag = Diagnostic {
            rule: "start_node".into(),
            severity: Severity::Error,
            message: "no start node".into(),
            node_id: None,
            edge: None,
            fix: None,
        };
        let json = body_json(ServerError::ValidationError(vec![diag])).await;
        assert_eq!(json["error"]["code"], "VALIDATION_ERROR");
        assert_eq!(json["error"]["status"], 422);
        let diags = json["error"]["diagnostics"]
            .as_array()
            .expect("diagnostics array");
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0]["rule"], "start_node");
    }

    #[tokio::test]
    async fn conflict_errors_produce_409() {
        let r1 = ServerError::PipelineNotRunning("pipe-1".into());
        let r2 = ServerError::QuestionAlreadyAnswered("q-1".into());
        assert_eq!(r1.status_code(), StatusCode::CONFLICT);
        assert_eq!(r2.status_code(), StatusCode::CONFLICT);

        // Also verify via IntoResponse
        let j1 = body_json(ServerError::PipelineNotRunning("pipe-1".into())).await;
        let j2 = body_json(ServerError::QuestionAlreadyAnswered("q-1".into())).await;
        assert_eq!(j1["error"]["status"], 409);
        assert_eq!(j2["error"]["status"], 409);
        assert_eq!(j1["error"]["code"], "PIPELINE_NOT_RUNNING");
        assert_eq!(j2["error"]["code"], "QUESTION_ALREADY_ANSWERED");
    }

    #[tokio::test]
    async fn graphviz_not_available_produces_501() {
        let json = body_json(ServerError::GraphvizNotAvailable).await;
        assert_eq!(json["error"]["code"], "GRAPHVIZ_NOT_AVAILABLE");
        assert_eq!(json["error"]["status"], 501);
    }

    #[tokio::test]
    async fn internal_error_produces_500() {
        let json = body_json(ServerError::Internal("db connection refused".into())).await;
        assert_eq!(json["error"]["code"], "INTERNAL_ERROR");
        assert_eq!(json["error"]["status"], 500);
    }

    #[tokio::test]
    async fn file_not_found_produces_404_json() {
        let json = body_json(ServerError::FileNotFound("docs/plan.md".into())).await;
        assert_eq!(json["error"]["code"], "FILE_NOT_FOUND");
        assert_eq!(json["error"]["status"], 404);
    }

    #[tokio::test]
    async fn path_traversal_produces_400_json() {
        let json = body_json(ServerError::PathTraversal).await;
        assert_eq!(json["error"]["code"], "PATH_TRAVERSAL");
        assert_eq!(json["error"]["status"], 400);
    }

    #[tokio::test]
    async fn file_too_large_produces_400_json() {
        let json = body_json(ServerError::FileTooLarge(2_000_000)).await;
        assert_eq!(json["error"]["code"], "FILE_TOO_LARGE");
        assert_eq!(json["error"]["status"], 400);
    }

    #[tokio::test]
    async fn no_diagnostics_field_for_non_validation_errors() {
        let json = body_json(ServerError::PipelineNotFound("x".into())).await;
        // With skip_serializing_if, the `diagnostics` key must be absent entirely.
        let obj = json["error"].as_object().expect("error object");
        assert!(
            !obj.contains_key("diagnostics"),
            "non-validation errors must not include a diagnostics field"
        );
    }
}
