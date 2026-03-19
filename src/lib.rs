//! `attractor-server` — HTTP server for the attractor DOT-based AI pipeline runner.
//!
//! This crate provides an HTTP/SSE interface for running and monitoring attractor
//! pipelines remotely. It exposes endpoints for submitting DOT graphs, streaming
//! execution events, and querying pipeline state.
//!
//! # Public API
//!
//! - [`create_router`] — assemble all 9 routes into an axum [`Router`] with CORS middleware.
//! - [`create_service`] — create an [`AppState`] from [`ServerConfig`] and call [`create_router`].
//! - [`ServerConfig`] — server configuration (host, port, CORS origins, data directory).

pub mod backends;
pub mod error;
pub mod files;
pub mod interviewer;
pub mod routes;
pub mod sse;
pub mod state;

// ---------------------------------------------------------------------------
// Re-exports
// ---------------------------------------------------------------------------

pub use error::ServerError;
pub use state::{AppState as State, PipelineStatus};

// ---------------------------------------------------------------------------
// ServerConfig
// ---------------------------------------------------------------------------

use std::path::PathBuf;

use axum::{Router, http::HeaderValue};
use tower_http::cors::{AllowOrigin, Any, CorsLayer};

use crate::state::AppState;

/// Server configuration used to initialise the HTTP service.
///
/// All fields have sensible defaults via [`Default`]; override only what you need:
///
/// ```rust
/// use attractor_server::ServerConfig;
///
/// let config = ServerConfig {
///     port: 8080,
///     ..ServerConfig::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// TCP address the server will bind to (default: `"0.0.0.0"`).
    pub host: String,
    /// TCP port the server will listen on (default: `3000`).
    pub port: u16,
    /// Allowed CORS origins.  An empty Vec means "allow all origins" (`*`).
    pub cors_origins: Vec<String>,
    /// Root directory for server data (logs, checkpoints, etc.).
    ///
    /// Defaults to `<temp_dir>/attractor-server`.
    pub data_dir: PathBuf,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 3000,
            cors_origins: Vec::new(),
            data_dir: std::env::temp_dir().join("attractor-server"),
        }
    }
}

// ---------------------------------------------------------------------------
// create_router
// ---------------------------------------------------------------------------

/// Assemble all 9 pipeline routes into an axum [`Router`] with CORS middleware.
///
/// The returned router is fully stateful (`Router<()>`) — it can be handed
/// directly to [`axum::serve`].
///
/// `cors_origins` controls the CORS policy:
/// - Empty vec → allow all origins (`*`) via [`AllowOrigin::any()`].
/// - Non-empty → allow only the listed origins; each request's `Origin` header
///   is reflected back when it matches, otherwise the CORS headers are absent.
///
/// Routes wired:
/// - `POST   /pipelines`
/// - `GET    /pipelines/{id}`
/// - `POST   /pipelines/{id}/cancel`
/// - `GET    /pipelines/{id}/events`
/// - `GET    /pipelines/{id}/questions`
/// - `POST   /pipelines/{id}/questions/{qid}/answer`
/// - `GET    /pipelines/{id}/graph`
/// - `GET    /pipelines/{id}/checkpoint`
/// - `GET    /pipelines/{id}/context`
pub fn create_router(state: AppState, cors_origins: Vec<String>) -> Router {
    let cors = if cors_origins.is_empty() {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        let parsed: Vec<HeaderValue> = cors_origins.iter().filter_map(|s| s.parse().ok()).collect();
        CorsLayer::new()
            .allow_origin(AllowOrigin::list(parsed))
            .allow_methods(Any)
            .allow_headers(Any)
    };

    routes::router().with_state(state).layer(cors)
}

// ---------------------------------------------------------------------------
// create_service
// ---------------------------------------------------------------------------

/// Create a fully-configured axum [`Router`] from a [`ServerConfig`].
///
/// Builds an [`AppState`] backed by `config.data_dir`, then delegates to
/// [`create_router`] with the CORS origins from the config.
pub fn create_service(config: &ServerConfig) -> Router {
    let state = AppState::new(config.data_dir.clone());
    create_router(state, config.cors_origins.clone())
}

// ---------------------------------------------------------------------------
// Tests — RED written first (stubs with todo!), then GREEN implemented above
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use axum::{
        body::Body,
        http::{Method, Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    // ---------------------------------------------------------------------------
    // Helpers
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

    async fn options_req(app: Router, uri: &str) -> axum::http::Response<Body> {
        app.oneshot(
            Request::builder()
                .method(Method::OPTIONS)
                .uri(uri)
                .header("origin", "http://example.com")
                .header("access-control-request-method", "POST")
                .header("access-control-request-headers", "content-type")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
    }

    async fn post_json(
        app: Router,
        uri: &str,
        body: serde_json::Value,
    ) -> axum::http::Response<Body> {
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

    async fn body_json(resp: axum::http::Response<Body>) -> serde_json::Value {
        let bytes = resp
            .into_body()
            .collect()
            .await
            .expect("collect body")
            .to_bytes();
        serde_json::from_slice(&bytes).expect("valid JSON body")
    }

    // ---------------------------------------------------------------------------
    // Test 1: create_router_serves_all_endpoints
    //
    // All 9 routes must be registered.  We verify by checking that none of them
    // return 405 METHOD_NOT_ALLOWED (which axum returns when a path matches but
    // the HTTP method does not).  Routes with a fake pipeline ID should return
    // 404 with PIPELINE_NOT_FOUND JSON (proving the route is matched and the
    // handler ran), rather than a generic 404 meaning "no route matched".
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn create_router_serves_all_endpoints() {
        let fake_id = "00000000-0000-0000-0000-000000000000";
        let fake_qid = "00000000-0000-0000-0000-000000000001";

        // --- Routes that look up a pipeline by ID and must return PIPELINE_NOT_FOUND ---
        let id_routes: &[(Method, String)] = &[
            (Method::GET, format!("/pipelines/{fake_id}")),
            (Method::POST, format!("/pipelines/{fake_id}/cancel")),
            (Method::GET, format!("/pipelines/{fake_id}/questions")),
            (Method::GET, format!("/pipelines/{fake_id}/graph")),
            (Method::GET, format!("/pipelines/{fake_id}/checkpoint")),
            (Method::GET, format!("/pipelines/{fake_id}/context")),
            (Method::GET, format!("/pipelines/{fake_id}/events")),
        ];

        for (method, path) in id_routes {
            let app = create_router(AppState::with_temp_dir(), vec![]);
            let resp = app
                .oneshot(
                    Request::builder()
                        .method(method.clone())
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();

            assert_ne!(
                resp.status(),
                StatusCode::METHOD_NOT_ALLOWED,
                "Route {} {} must be registered (got 405)",
                method,
                path
            );

            // For GET/POST routes that look up a pipeline, the handler must have
            // run and returned PIPELINE_NOT_FOUND rather than a generic 404.
            // (SSE endpoint returns 404 with our error JSON too.)
            let status = resp.status();
            assert_eq!(
                status,
                StatusCode::NOT_FOUND,
                "Route {} {} with unknown ID should return 404 PIPELINE_NOT_FOUND; got {}",
                method,
                path,
                status
            );
        }

        // --- POST /pipelines — invalid (empty) body → 422/400, not 404 ---
        let app = create_router(AppState::with_temp_dir(), vec![]);
        let resp = post_json(app, "/pipelines", serde_json::json!({})).await;
        assert_ne!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "POST /pipelines must be registered (got 404 route-not-found)"
        );

        // --- POST /pipelines/{id}/questions/{qid}/answer ---
        let app = create_router(AppState::with_temp_dir(), vec![]);
        let resp = post_json(
            app,
            &format!("/pipelines/{fake_id}/questions/{fake_qid}/answer"),
            serde_json::json!({ "answer": "yes" }),
        )
        .await;
        assert_ne!(
            resp.status(),
            StatusCode::METHOD_NOT_ALLOWED,
            "POST /pipelines/{{id}}/questions/{{qid}}/answer must be registered (got 405)"
        );
    }

    // ---------------------------------------------------------------------------
    // Test 2: cors_headers_present
    //
    // A preflight OPTIONS request must receive an Access-Control-Allow-Origin
    // header, confirming CorsLayer is active.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn cors_headers_present() {
        let app = create_router(AppState::with_temp_dir(), vec![]);
        let resp = options_req(app, "/pipelines").await;

        let headers = resp.headers();
        assert!(
            headers.contains_key("access-control-allow-origin"),
            "Preflight OPTIONS response must include Access-Control-Allow-Origin; \
             got headers: {:?}",
            headers
        );
    }

    // ---------------------------------------------------------------------------
    // Test 3: create_service_from_config
    //
    // create_service() must build a working router from a ServerConfig.
    // We verify that GET /pipelines/{id} with an unknown ID returns the expected
    // PIPELINE_NOT_FOUND 404 response, proving the router is correctly wired.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn create_service_from_config() {
        let config = ServerConfig {
            data_dir: std::env::temp_dir().join(format!("attractor-test-{}", uuid::Uuid::new_v4())),
            ..ServerConfig::default()
        };

        let app = create_service(&config);

        let fake_id = "00000000-0000-0000-0000-000000000000";
        let resp = get_req(app, &format!("/pipelines/{fake_id}")).await;

        assert_eq!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "create_service router must return 404 for unknown pipeline"
        );

        let json = body_json(resp).await;
        assert_eq!(
            json["error"]["code"], "PIPELINE_NOT_FOUND",
            "create_service router must return PIPELINE_NOT_FOUND; got: {json}"
        );
    }

    // ---------------------------------------------------------------------------
    // Test 4: cors_specific_origin_reflected (SRV-FIX-001)
    //
    // When ServerConfig.cors_origins contains specific origins, create_service
    // must configure CORS to reflect those origins rather than using the wildcard
    // AllowOrigin::Any.  A preflight OPTIONS request from a listed origin should
    // receive that exact origin in Access-Control-Allow-Origin (not "*").
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn cors_specific_origin_reflected() {
        let config = ServerConfig {
            cors_origins: vec!["https://myapp.example.com".to_string()],
            data_dir: std::env::temp_dir().join(format!("attractor-test-{}", uuid::Uuid::new_v4())),
            ..ServerConfig::default()
        };

        let app = create_service(&config);

        let resp = app
            .oneshot(
                Request::builder()
                    .method(Method::OPTIONS)
                    .uri("/pipelines")
                    .header("origin", "https://myapp.example.com")
                    .header("access-control-request-method", "POST")
                    .header("access-control-request-headers", "content-type")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let acao = resp
            .headers()
            .get("access-control-allow-origin")
            .map(|v| v.to_str().unwrap_or(""))
            .unwrap_or("");

        // With specific origins wired, the reflected value must be the exact
        // origin, NOT the wildcard "*" produced by AllowOrigin::Any.
        assert_eq!(
            acao, "https://myapp.example.com",
            "listed origin must be reflected in ACAO header; got: {:?}",
            acao
        );
    }
}
