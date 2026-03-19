//! attractor-server binary entry point.
//!
//! Provides a clap-based CLI for running the attractor HTTP server.

use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::{EnvFilter, fmt};

use attractor_server::{create_router, state::AppState};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// HTTP server for the attractor DOT-based AI pipeline runner.
#[derive(Debug, Parser)]
#[command(
    name = "attractor-server",
    about = "HTTP server for the attractor DOT-based AI pipeline runner",
    version
)]
pub struct Cli {
    /// TCP port to listen on.
    #[arg(long, default_value_t = 3000)]
    pub port: u16,

    /// TCP host address to bind to.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Comma-separated list of allowed CORS origins.
    /// An empty value means allow all origins (`*`).
    #[arg(long, default_value = "")]
    pub cors_origins: String,

    /// Directory for persistent server data (logs, checkpoints, etc.).
    /// Defaults to `<temp_dir>/attractor-server`.
    #[arg(long)]
    pub data_dir: Option<PathBuf>,

    /// Path to a single DOT pipeline file (single-pipeline mode).
    /// When provided the server is pre-loaded with that pipeline.
    #[arg(long)]
    pub file: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Helpers (public so tests can reach them from within the module)
// ---------------------------------------------------------------------------

/// Parse a comma-separated CORS origins string into a `Vec<String>`.
///
/// Returns an empty vec when `raw` is blank.  Trims whitespace around each
/// entry and drops empty tokens.
pub fn parse_cors_origins(raw: &str) -> Vec<String> {
    if raw.is_empty() {
        return Vec::new();
    }
    raw.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Resolve the data directory: use the provided path, or fall back to
/// `<temp_dir>/attractor-server`.
pub fn resolve_data_dir(data_dir: Option<PathBuf>) -> PathBuf {
    data_dir.unwrap_or_else(|| std::env::temp_dir().join("attractor-server"))
}

// ---------------------------------------------------------------------------
// Async entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    // Initialise tracing — respects RUST_LOG, defaults to "info".
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let cors_origins = parse_cors_origins(&cli.cors_origins);
    let data_dir = resolve_data_dir(cli.data_dir);

    // Create AppState directly so we can pre-load a pipeline before the router
    // is assembled (required for single-pipeline mode via --file).
    let state = AppState::new(data_dir);

    // Single-pipeline mode: pre-load the DOT file into AppState now so it is
    // available immediately when the server starts accepting requests.
    if let Some(ref file_path) = cli.file {
        tracing::info!(
            file = %file_path.display(),
            "single-pipeline mode: loading DOT file"
        );
        preload_pipeline_from_file(&state, file_path)
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "failed to pre-load pipeline from {}: {e:?}",
                    file_path.display()
                )
            });
    }

    let app = create_router(state, cors_origins);

    let bind_addr = format!("{}:{}", cli.host, cli.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .unwrap_or_else(|e| panic!("failed to bind {bind_addr}: {e}"));

    let local_addr = listener.local_addr().expect("local_addr");

    if cli.file.is_some() {
        tracing::info!(addr = %local_addr, "attractor-server listening (single-pipeline mode)");
    } else {
        // Persistent server mode — accepts arbitrary pipelines via the API.
        tracing::info!(addr = %local_addr, "attractor-server listening (persistent mode)");
    }

    axum::serve(listener, app).await.expect("server error");
}

// ---------------------------------------------------------------------------
// Pre-load helper (public so tests can call it directly)
// ---------------------------------------------------------------------------

/// Read a DOT file from `file_path`, parse and validate it, then spawn it as
/// a running pipeline inside `state`.
///
/// Returns the new pipeline ID on success.  On I/O error the function returns
/// `Err(ServerError::Internal(...))` so the caller can decide whether to
/// panic (startup) or surface the error via HTTP (future use).
pub async fn preload_pipeline_from_file(
    state: &AppState,
    file_path: &std::path::Path,
) -> Result<String, attractor_server::ServerError> {
    let dot = tokio::fs::read_to_string(file_path).await.map_err(|e| {
        attractor_server::ServerError::Internal(format!(
            "failed to read DOT file {}: {e}",
            file_path.display()
        ))
    })?;
    attractor_server::routes::spawn_pipeline(state, dot, std::collections::HashMap::new(), None).await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    // ----- Cli struct defaults -----

    #[test]
    fn cli_defaults() {
        let cli = Cli::try_parse_from(["attractor-server"]).unwrap();
        assert_eq!(cli.port, 3000);
        assert_eq!(cli.host, "0.0.0.0");
        assert_eq!(cli.cors_origins, "");
        assert!(cli.data_dir.is_none());
        assert!(cli.file.is_none());
    }

    #[test]
    fn cli_port_override() {
        let cli = Cli::try_parse_from(["attractor-server", "--port", "8080"]).unwrap();
        assert_eq!(cli.port, 8080);
    }

    #[test]
    fn cli_host_override() {
        let cli = Cli::try_parse_from(["attractor-server", "--host", "127.0.0.1"]).unwrap();
        assert_eq!(cli.host, "127.0.0.1");
    }

    #[test]
    fn cli_cors_origins_stored_raw() {
        let cli = Cli::try_parse_from([
            "attractor-server",
            "--cors-origins",
            "http://localhost:3000,http://example.com",
        ])
        .unwrap();
        assert_eq!(cli.cors_origins, "http://localhost:3000,http://example.com");
    }

    #[test]
    fn cli_data_dir() {
        let cli = Cli::try_parse_from(["attractor-server", "--data-dir", "/tmp/mydata"]).unwrap();
        assert_eq!(cli.data_dir, Some(PathBuf::from("/tmp/mydata")));
    }

    #[test]
    fn cli_file() {
        let cli =
            Cli::try_parse_from(["attractor-server", "--file", "/path/to/pipeline.dot"]).unwrap();
        assert_eq!(cli.file, Some(PathBuf::from("/path/to/pipeline.dot")));
    }

    // ----- parse_cors_origins -----

    #[test]
    fn cors_origins_parsing_empty_yields_empty_vec() {
        let cli = Cli::try_parse_from(["attractor-server"]).unwrap();
        let origins = parse_cors_origins(&cli.cors_origins);
        assert!(origins.is_empty());
    }

    #[test]
    fn cors_origins_parsing_multiple_trimmed() {
        let cli = Cli::try_parse_from([
            "attractor-server",
            "--cors-origins",
            "http://localhost:3000, http://example.com",
        ])
        .unwrap();
        let origins = parse_cors_origins(&cli.cors_origins);
        assert_eq!(origins, vec!["http://localhost:3000", "http://example.com"]);
    }

    #[test]
    fn cors_origins_single_entry() {
        let origins = parse_cors_origins("https://myapp.example.com");
        assert_eq!(origins, vec!["https://myapp.example.com"]);
    }

    // ----- resolve_data_dir -----

    #[test]
    fn data_dir_defaults_to_temp_attractor_server() {
        let cli = Cli::try_parse_from(["attractor-server"]).unwrap();
        let data_dir = resolve_data_dir(cli.data_dir);
        let expected = std::env::temp_dir().join("attractor-server");
        assert_eq!(data_dir, expected);
    }

    #[test]
    fn data_dir_respects_override() {
        let cli = Cli::try_parse_from(["attractor-server", "--data-dir", "/custom/path"]).unwrap();
        let data_dir = resolve_data_dir(cli.data_dir);
        assert_eq!(data_dir, PathBuf::from("/custom/path"));
    }

    // ---------------------------------------------------------------------------
    // SRV-FIX-002: preload_pipeline_from_file submits pipeline to state
    //
    // When a DOT file is passed to preload_pipeline_from_file(), the resulting
    // pipeline must appear in the AppState so the server starts in
    // single-pipeline mode with the pipeline already running.
    // ---------------------------------------------------------------------------
    #[tokio::test]
    async fn file_preload_submits_pipeline_to_state() {
        let dot =
            "digraph test {\n  start [shape=Mdiamond]\n  exit  [shape=Msquare]\n  start -> exit\n}";

        let tmp_dir =
            std::env::temp_dir().join(format!("attractor-preload-test-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&tmp_dir)
            .await
            .expect("create temp dir");

        let file_path = tmp_dir.join("pipeline.dot");
        tokio::fs::write(&file_path, dot)
            .await
            .expect("write dot file");

        let state = attractor_server::state::AppState::new(tmp_dir.clone());
        preload_pipeline_from_file(&state, &file_path)
            .await
            .expect("preload should succeed");

        let count = state.pipelines.read().await.len();
        assert_eq!(
            count, 1,
            "preloaded pipeline must appear in AppState; got {count}"
        );
    }
}
