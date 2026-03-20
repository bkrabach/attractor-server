//! File listing and content handlers for pipeline working directories.
//!
//! Exposes two handler functions:
//! - `list_files`      — returns the recursive directory tree as JSON
//! - `get_file_content` — returns a single file's contents as plain text

use std::path::{Path, PathBuf};

use axum::Json;
use axum::extract::{Path as AxumPath, State};
use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::error::ServerError;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A node in the file tree — either a file or a directory with children.
#[derive(Debug, Clone, Serialize)]
pub struct FileNode {
    /// File or directory name (e.g. "plan.md").
    pub name: String,
    /// Path relative to the pipeline's working directory (e.g. "docs/plans/plan.md").
    pub path: String,
    /// `"file"` or `"directory"`.
    #[serde(rename = "type")]
    pub node_type: String,
    /// File size in bytes (files only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// ISO 8601 last-modified timestamp.
    #[serde(rename = "modifiedAt", skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,
    /// Child entries (directories only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub children: Option<Vec<FileNode>>,
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum file size the server will return (1 MB).
const MAX_FILE_SIZE: u64 = 1_048_576;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Recursively build a sorted tree of [`FileNode`]s from a directory.
fn build_tree(dir: &Path, base: &Path) -> std::io::Result<Vec<FileNode>> {
    let mut entries = Vec::new();

    if !dir.exists() {
        return Ok(entries);
    }

    let mut dir_entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    dir_entries.sort_by_key(|e| e.file_name());

    for entry in dir_entries {
        let metadata = entry.metadata()?;
        let name = entry.file_name().to_string_lossy().to_string();
        let rel_path = entry
            .path()
            .strip_prefix(base)
            .unwrap_or(&entry.path())
            .to_string_lossy()
            .to_string();

        let modified_at = metadata
            .modified()
            .ok()
            .map(|t| DateTime::<Utc>::from(t).to_rfc3339());

        if metadata.is_dir() {
            let children = build_tree(&entry.path(), base)?;
            entries.push(FileNode {
                name,
                path: rel_path,
                node_type: "directory".to_string(),
                size: None,
                modified_at,
                children: Some(children),
            });
        } else {
            entries.push(FileNode {
                name,
                path: rel_path,
                node_type: "file".to_string(),
                size: Some(metadata.len()),
                modified_at,
                children: None,
            });
        }
    }

    Ok(entries)
}

/// Validate that a relative path does not escape the working directory.
///
/// Rejects:
/// - Paths containing `..` components (directory traversal)
/// - Absolute paths (leading `/` bypasses `PathBuf::join` containment)
/// - Empty paths
fn validate_path(file_path: &str) -> Result<(), ServerError> {
    // Reject empty paths.
    if file_path.is_empty() {
        return Err(ServerError::FileNotFound(String::new()));
    }

    // Reject absolute paths — PathBuf::join replaces the base when the
    // argument is absolute, which would allow reading arbitrary files.
    if file_path.starts_with('/') || file_path.starts_with('\\') {
        return Err(ServerError::PathTraversal);
    }

    // Reject any path component that is ".." (directory traversal).
    // Check both the raw string and each path component to handle
    // URL-decoded paths and platform differences.
    for component in std::path::Path::new(file_path).components() {
        if let std::path::Component::ParentDir = component {
            return Err(ServerError::PathTraversal);
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /pipelines/{id}/files` — return the recursive directory tree as JSON.
pub async fn list_files(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Vec<FileNode>>, ServerError> {
    let handle = state
        .get_pipeline(&id)
        .await
        .ok_or_else(|| ServerError::PipelineNotFound(id.clone()))?;

    let working_dir = handle.working_dir.clone();

    let tree = tokio::task::spawn_blocking(move || build_tree(&working_dir, &working_dir))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(Json(tree))
}

/// `GET /pipelines/{id}/files/{*path}` — return a single file's contents.
pub async fn get_file_content(
    State(state): State<AppState>,
    AxumPath((id, file_path)): AxumPath<(String, String)>,
) -> Result<String, ServerError> {
    let handle = state
        .get_pipeline(&id)
        .await
        .ok_or_else(|| ServerError::PipelineNotFound(id.clone()))?;

    // Security: reject path traversal and absolute paths.
    validate_path(&file_path)?;

    let full_path: PathBuf = handle.working_dir.join(&file_path);

    // Use async metadata to check existence and type (no blocking std::fs).
    let metadata = match tokio::fs::metadata(&full_path).await {
        Ok(m) => m,
        Err(_) => return Err(ServerError::FileNotFound(file_path)),
    };

    if !metadata.is_file() {
        return Err(ServerError::FileNotFound(file_path));
    }

    if metadata.len() > MAX_FILE_SIZE {
        return Err(ServerError::FileTooLarge(metadata.len()));
    }

    let content = tokio::fs::read_to_string(&full_path)
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))?;

    Ok(content)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // build_tree tests
    // -----------------------------------------------------------------------

    #[test]
    fn build_tree_empty_dir() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let tree = build_tree(tmp.path(), tmp.path()).expect("build_tree");
        assert!(tree.is_empty(), "empty dir should produce empty tree");
    }

    #[test]
    fn build_tree_nonexistent_dir() {
        let path = std::path::PathBuf::from("/tmp/does-not-exist-attractor-test");
        let tree = build_tree(&path, &path).expect("build_tree");
        assert!(tree.is_empty(), "nonexistent dir should produce empty tree");
    }

    #[test]
    fn build_tree_single_file() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        std::fs::write(tmp.path().join("hello.md"), "# Hello").expect("write file");

        let tree = build_tree(tmp.path(), tmp.path()).expect("build_tree");
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].name, "hello.md");
        assert_eq!(tree[0].path, "hello.md");
        assert_eq!(tree[0].node_type, "file");
        assert!(tree[0].size.is_some());
        assert!(tree[0].children.is_none());
    }

    #[test]
    fn build_tree_nested_directory() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        std::fs::create_dir_all(tmp.path().join("docs/plans")).expect("create dirs");
        std::fs::write(tmp.path().join("docs/plans/plan.md"), "# Plan").expect("write file");

        let tree = build_tree(tmp.path(), tmp.path()).expect("build_tree");
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].name, "docs");
        assert_eq!(tree[0].node_type, "directory");

        let docs_children = tree[0].children.as_ref().expect("docs children");
        assert_eq!(docs_children.len(), 1);
        assert_eq!(docs_children[0].name, "plans");

        let plans_children = docs_children[0].children.as_ref().expect("plans children");
        assert_eq!(plans_children.len(), 1);
        assert_eq!(plans_children[0].name, "plan.md");
        assert_eq!(plans_children[0].path, "docs/plans/plan.md");
    }

    #[test]
    fn build_tree_entries_sorted_alphabetically() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        std::fs::write(tmp.path().join("charlie.txt"), "c").unwrap();
        std::fs::write(tmp.path().join("alpha.txt"), "a").unwrap();
        std::fs::write(tmp.path().join("bravo.txt"), "b").unwrap();

        let tree = build_tree(tmp.path(), tmp.path()).expect("build_tree");
        let names: Vec<&str> = tree.iter().map(|n| n.name.as_str()).collect();
        assert_eq!(names, vec!["alpha.txt", "bravo.txt", "charlie.txt"]);
    }

    // -----------------------------------------------------------------------
    // validate_path tests
    // -----------------------------------------------------------------------

    #[test]
    fn validate_path_rejects_dot_dot() {
        let result = validate_path("../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn validate_path_rejects_embedded_dot_dot() {
        let result = validate_path("docs/../../../etc/passwd");
        assert!(result.is_err());
    }

    #[test]
    fn validate_path_rejects_absolute_path() {
        let result = validate_path("/etc/passwd");
        assert!(result.is_err(), "absolute paths must be rejected");
    }

    #[test]
    fn validate_path_rejects_empty_path() {
        let result = validate_path("");
        assert!(result.is_err(), "empty paths must be rejected");
    }

    #[test]
    fn validate_path_allows_normal_paths() {
        assert!(validate_path("docs/plans/plan.md").is_ok());
        assert!(validate_path("src/main.rs").is_ok());
        assert!(validate_path("README.md").is_ok());
    }

    #[test]
    fn validate_path_allows_filenames_with_dots() {
        // Filenames like "backup..tar" should not be rejected.
        assert!(validate_path("backup..tar").is_ok());
        assert!(validate_path("file...name").is_ok());
        assert!(validate_path(".hidden").is_ok());
    }
}
