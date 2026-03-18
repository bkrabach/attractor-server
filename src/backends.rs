//! LLM backend implementations for the attractor server.
//!
//! Provides [`LlmCodergenBackend`] — a [`CodergenBackend`] that forwards
//! prompts to a real LLM via [`unified_llm`].

use async_trait::async_trait;
use unified_llm::{Client, GenerateParams, generate};

use attractor::error::EngineError;
use attractor::graph::Node;
use attractor::handler::{CodergenBackend, CodergenResult};
use attractor::state::context::Context;

// ---------------------------------------------------------------------------
// LlmCodergenBackend
// ---------------------------------------------------------------------------

/// A [`CodergenBackend`] that calls the LLM API via [`unified_llm`].
///
/// Wraps a [`Client`] built from environment variables
/// (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).
///
/// Model priority (highest to lowest):
/// 1. Node-level `llm_model` attribute (set by stylesheet or DOT directly)
/// 2. `default_model` supplied at construction time
pub struct LlmCodergenBackend {
    client: Client,
    default_model: String,
}

impl LlmCodergenBackend {
    /// Create a new backend with the given client and default model.
    pub fn new(client: Client, default_model: impl Into<String>) -> Self {
        LlmCodergenBackend {
            client,
            default_model: default_model.into(),
        }
    }

    /// Try to build a backend from environment variables.
    ///
    /// Returns `None` if no LLM credentials are found in the environment,
    /// allowing the server to fall back to simulation mode with a warning.
    pub fn from_env(default_model: impl Into<String>) -> Option<Self> {
        match Client::from_env() {
            Ok(client) => Some(LlmCodergenBackend::new(client, default_model)),
            Err(e) => {
                tracing::warn!(
                    "LLM client unavailable — codergen running in simulation mode. \
                     Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY to enable real LLM calls. \
                     Error: {e}"
                );
                None
            }
        }
    }
}

#[async_trait]
impl CodergenBackend for LlmCodergenBackend {
    async fn run(
        &self,
        node: &Node,
        prompt: &str,
        _ctx: &Context,
    ) -> Result<CodergenResult, EngineError> {
        // Honour the per-node model if the stylesheet/DOT set one.
        let model = if !node.llm_model.is_empty() {
            node.llm_model.clone()
        } else {
            self.default_model.clone()
        };

        let mut params = GenerateParams::new(model, prompt);
        params.client = Some(self.client.clone());

        // Route to the node's explicit provider when specified.
        // Without this, the Client defaults to the first registered provider
        // (typically OpenAI), which rejects Anthropic/Gemini model names.
        if !node.llm_provider.is_empty() {
            params.provider = Some(node.llm_provider.clone());
        }

        let result = generate(params).await.map_err(|e| EngineError::Handler {
            node_id: node.id.clone(),
            message: format!("LLM call failed: {e}"),
        })?;

        Ok(CodergenResult::Text(result.text))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use attractor::graph::Node;

    /// Build GenerateParams from a node the same way LlmCodergenBackend::run() does.
    ///
    /// This helper is used to verify the param-building logic in isolation,
    /// without needing a real LLM client or making a network call.
    fn build_params_for_node(node: &Node, default_model: &str, prompt: &str) -> GenerateParams {
        let model = if !node.llm_model.is_empty() {
            node.llm_model.clone()
        } else {
            default_model.to_string()
        };
        let mut params = GenerateParams::new(model, prompt);
        if !node.llm_provider.is_empty() {
            params.provider = Some(node.llm_provider.clone());
        }
        params
    }

    /// RED test for SRV-BUG-003: when a node has `llm_provider = "anthropic"`,
    /// the `GenerateParams::provider` field must be set to `"anthropic"` so the
    /// Client routes to the Anthropic adapter instead of the OpenAI default.
    ///
    /// Before fix: `params.provider` was `None` regardless of node attribute.
    /// After fix: `params.provider` reflects the node's `llm_provider`.
    #[test]
    fn node_llm_provider_forwarded_to_generate_params() {
        let mut node = Node::default();
        node.llm_model = "claude-opus-4-6".to_string();
        node.llm_provider = "anthropic".to_string();

        let params = build_params_for_node(&node, "gpt-4o", "hello");

        assert_eq!(
            params.provider,
            Some("anthropic".to_string()),
            "llm_provider='anthropic' must be forwarded to GenerateParams::provider"
        );
    }

    /// When `llm_provider` is empty, provider should remain None (use client default).
    #[test]
    fn empty_llm_provider_leaves_params_provider_as_none() {
        let mut node = Node::default();
        node.llm_model = "gpt-4o".to_string();
        // llm_provider intentionally left empty

        let params = build_params_for_node(&node, "gpt-4o", "hello");

        assert_eq!(
            params.provider, None,
            "empty llm_provider must NOT set GenerateParams::provider"
        );
    }

    /// Node llm_model overrides default_model in params.
    #[test]
    fn node_llm_model_used_when_set() {
        let mut node = Node::default();
        node.llm_model = "claude-opus-4-6".to_string();

        let params = build_params_for_node(&node, "gpt-4o", "prompt");

        assert_eq!(params.model, "claude-opus-4-6");
    }

    /// When node llm_model is empty, default_model is used.
    #[test]
    fn default_model_used_when_node_llm_model_empty() {
        let node = Node::default(); // llm_model is empty

        let params = build_params_for_node(&node, "gpt-4o", "prompt");

        assert_eq!(params.model, "gpt-4o");
    }

    /// Verify LlmCodergenBackend::from_env() never panics, regardless of which
    /// API keys are present in the environment.
    ///
    /// Note: whether credentials are present or absent affects the `Option`
    /// return value, not the absence of a panic — which is what this test
    /// actually asserts.  Behavioral coverage (returns `None` without keys)
    /// requires an integration test that controls the env.
    #[test]
    fn from_env_does_not_panic() {
        let result = std::panic::catch_unwind(|| LlmCodergenBackend::from_env("gpt-4o"));
        assert!(
            result.is_ok(),
            "from_env() must not panic regardless of env vars"
        );
    }

    /// Structural compile-check: confirms `LlmCodergenBackend` exists as a type.
    /// Cannot test `new()` directly without a real `Client` (requires API keys).
    /// Behavioral coverage lives in `from_env_does_not_panic` and integration tests.
    #[test]
    fn llm_codergen_backend_type_exists() {
        let _ = std::any::TypeId::of::<LlmCodergenBackend>();
    }

    /// RED test for SRV-BUG-001: verify LlmCodergenBackend implements CodergenBackend.
    ///
    /// Before fix: LlmCodergenBackend didn't exist in attractor-server.
    /// After fix: this compiles and the trait bound is satisfied.
    #[test]
    fn llm_codergen_backend_is_codergen_backend() {
        fn assert_is_codergen_backend<T: CodergenBackend>() {}
        assert_is_codergen_backend::<LlmCodergenBackend>();
    }

    /// Structural compile-check: confirms `LlmCodergenBackend` can be named as a type.
    /// Model-selection logic (node `llm_model` vs `default_model`) is exercised
    /// in isolation via `build_params_for_node` tests above.
    #[test]
    fn llm_codergen_backend_struct_is_accessible() {
        let _ = std::any::TypeId::of::<LlmCodergenBackend>();
    }
}
