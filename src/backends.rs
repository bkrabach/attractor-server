//! LLM backend implementations for the attractor server.
//!
//! Provides [`LlmCodergenBackend`] — a [`CodergenBackend`] that runs a full
//! agentic tool loop via [`coding_agent_loop::Session`], executing tool calls
//! (write_file, shell, read_file, etc.) on disk.

use std::path::PathBuf;

use async_trait::async_trait;
use unified_llm::Client;

use attractor::error::EngineError;
use attractor::graph::Node;
use attractor::handler::{CodergenBackend, CodergenResult};
use attractor::state::context::Context;

use coding_agent_loop::profile::{anthropic_profile, gemini_profile, openai_profile};
use coding_agent_loop::turns::Turn;
use coding_agent_loop::{LocalExecutionEnvironment, Session, SessionConfig};

// ---------------------------------------------------------------------------
// LlmCodergenBackend
// ---------------------------------------------------------------------------

/// A [`CodergenBackend`] that runs a full agentic tool loop via
/// [`coding_agent_loop::Session`].
///
/// Each `run()` call creates a fresh `Session` with the appropriate
/// [`ProviderProfile`](coding_agent_loop::profile::ProviderProfile), submits
/// the prompt, and collects the final assistant text after all tool calls
/// have been executed on disk.
///
/// Uses [`Client::from_env()`] so it picks up API keys from environment
/// variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).
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

/// Infer the LLM provider from a model name string.
///
/// Returns `"anthropic"`, `"gemini"`, or `"openai"` (the default fallback).
fn infer_provider(model: &str) -> &'static str {
    let m = model.trim().to_lowercase();
    if m.starts_with("claude") {
        "anthropic"
    } else if m.starts_with("gemini") {
        "gemini"
    } else {
        // Covers gpt-*, o1-*, o3-*, codex-*, and anything else.
        "openai"
    }
}

#[async_trait]
impl CodergenBackend for LlmCodergenBackend {
    async fn run(
        &self,
        node: &Node,
        prompt: &str,
        ctx: &Context,
    ) -> Result<CodergenResult, EngineError> {
        // 1. Resolve model: node-level override > default.
        let model = if !node.llm_model.is_empty() {
            node.llm_model.clone()
        } else {
            self.default_model.clone()
        };

        // 2. Resolve provider: node-level override > infer from model name.
        let provider_id = if !node.llm_provider.is_empty() {
            node.llm_provider.as_str()
        } else {
            infer_provider(&model)
        };

        // 3. Select the appropriate provider profile (tool set + system prompt).
        let profile = match provider_id {
            "anthropic" => anthropic_profile(&model),
            "gemini" => gemini_profile(&model),
            "openai" => openai_profile(&model),
            other => {
                tracing::warn!(
                    node_id = %node.id,
                    provider = %other,
                    "unrecognised llm_provider; falling back to openai profile"
                );
                openai_profile(&model)
            }
        };

        // 4. Resolve working directory from pipeline context.
        let working_dir_str = ctx.get_string("_working_dir");
        let working_dir = if working_dir_str.is_empty() {
            std::env::temp_dir()
        } else {
            PathBuf::from(&working_dir_str)
        };

        tracing::debug!(
            node_id = %node.id,
            model = %model,
            provider = %provider_id,
            working_dir = %working_dir.display(),
            "codergen: resolved session parameters"
        );

        let env = Box::new(LocalExecutionEnvironment::new(&working_dir));

        // 5. Configure the session with reasonable limits for a pipeline node.
        //
        // Only override reasoning_effort when the DOT author explicitly set a
        // non-default value.  The parser defaults every node to "high", so
        // forwarding it unconditionally would inject reasoning_effort into
        // requests for models that don't support it (e.g. gpt-4o, gemini).
        // SessionConfig::default() leaves it as None, which is correct for
        // all models.  Only explicit "low"/"medium" overrides take effect.
        let config = SessionConfig {
            max_tool_rounds_per_input: 50, // prevent runaway tool loops
            reasoning_effort: if !node.reasoning_effort.is_empty()
                && node.reasoning_effort != "high"
            {
                Some(node.reasoning_effort.clone())
            } else {
                None
            },
            ..Default::default()
        };

        // 6. Create a fresh Session and run the agentic loop.
        let mut session = Session::new(config, profile, env, self.client.clone());

        let submit_result = session.submit(prompt).await;

        // 7. Always shut down the session — even on error — to avoid leaking
        //    subagent resources in a long-running server process.
        if submit_result.is_err() {
            session.shutdown().await;
            return submit_result
                .map_err(|e| EngineError::Handler {
                    node_id: node.id.clone(),
                    message: format!("agent session failed: {e}"),
                })
                .map(|()| CodergenResult::Text(String::new()));
        }

        // 8. Extract the last non-empty assistant text from the session history.
        let text = session
            .history()
            .turns()
            .iter()
            .rev()
            .find_map(|t| match t {
                Turn::Assistant(a) if !a.content.is_empty() => Some(a.content.clone()),
                _ => None,
            })
            .unwrap_or_default();

        if text.is_empty() {
            tracing::warn!(
                node_id = %node.id,
                "codergen: session completed but no non-empty assistant text found in history"
            );
        }

        // 9. Graceful shutdown (flush events, clean up subagents).
        session.shutdown().await;

        Ok(CodergenResult::Text(text))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use attractor::graph::Node;

    // -- infer_provider -----------------------------------------------------------

    #[test]
    fn infer_provider_openai_models() {
        assert_eq!(infer_provider("gpt-4o"), "openai");
        assert_eq!(infer_provider("gpt-4o-mini"), "openai");
        assert_eq!(infer_provider("o1-preview"), "openai");
        assert_eq!(infer_provider("o3-mini"), "openai");
        assert_eq!(infer_provider("codex-4o"), "openai");
    }

    #[test]
    fn infer_provider_anthropic_models() {
        assert_eq!(infer_provider("claude-opus-4-5"), "anthropic");
        assert_eq!(infer_provider("claude-haiku-4-5-20251001"), "anthropic");
        assert_eq!(infer_provider("claude-sonnet-4-20250514"), "anthropic");
        // Case-insensitive: model strings from DOT may vary.
        assert_eq!(infer_provider("Claude-3-Opus"), "anthropic");
    }

    #[test]
    fn infer_provider_gemini_models() {
        assert_eq!(infer_provider("gemini-2.5-pro"), "gemini");
        assert_eq!(infer_provider("gemini-2.0-flash"), "gemini");
        assert_eq!(infer_provider("Gemini-Pro"), "gemini");
    }

    #[test]
    fn infer_provider_unknown_defaults_to_openai() {
        assert_eq!(infer_provider("some-custom-model"), "openai");
        assert_eq!(infer_provider("deepseek-r1"), "openai");
        assert_eq!(infer_provider(""), "openai");
    }

    #[test]
    fn infer_provider_trims_whitespace() {
        // DOT attribute values may have leading/trailing whitespace.
        assert_eq!(infer_provider(" claude-opus-4-5"), "anthropic");
        assert_eq!(infer_provider("  gemini-2.5-pro  "), "gemini");
        assert_eq!(infer_provider(" gpt-4o "), "openai");
    }

    // -- Model / provider resolution logic ----------------------------------------

    /// Replicate the model resolution logic from run() without a real Client.
    fn resolve_model(node: &Node, default_model: &str) -> String {
        if !node.llm_model.is_empty() {
            node.llm_model.clone()
        } else {
            default_model.to_string()
        }
    }

    /// Replicate the provider resolution logic from run() without a real Client.
    /// Returns a String (not &str) so we can return node.llm_provider values.
    fn resolve_provider(node: &Node, model: &str) -> String {
        if !node.llm_provider.is_empty() {
            node.llm_provider.clone()
        } else {
            infer_provider(model).to_string()
        }
    }

    #[test]
    fn node_llm_model_overrides_default() {
        let mut node = Node::default();
        node.llm_model = "claude-opus-4-6".to_string();

        let model = resolve_model(&node, "gpt-4o");
        let provider = resolve_provider(&node, &model);
        assert_eq!(model, "claude-opus-4-6");
        assert_eq!(provider, "anthropic");
    }

    #[test]
    fn default_model_used_when_node_llm_model_empty() {
        let node = Node::default(); // llm_model is empty
        let model = resolve_model(&node, "gpt-4o");
        let provider = resolve_provider(&node, &model);
        assert_eq!(model, "gpt-4o");
        assert_eq!(provider, "openai");
    }

    /// When `llm_provider` is explicitly set on the node, the backend must
    /// use that value directly rather than inferring from the model name.
    /// This is the critical path for custom/third-party models where
    /// infer_provider would guess wrong.
    #[test]
    fn node_llm_provider_overrides_inference() {
        let mut node = Node::default();
        node.llm_model = "my-custom-model".to_string();
        node.llm_provider = "anthropic".to_string();

        let model = resolve_model(&node, "gpt-4o");
        let provider = resolve_provider(&node, &model);

        // Without the override, infer_provider would return "openai":
        assert_eq!(infer_provider(&model), "openai");
        // But resolve_provider must honour the explicit node.llm_provider:
        assert_eq!(provider, "anthropic");
    }

    #[test]
    fn empty_llm_provider_falls_back_to_inference() {
        let mut node = Node::default();
        node.llm_model = "gemini-2.5-pro".to_string();
        // llm_provider intentionally left empty

        let model = resolve_model(&node, "gpt-4o");
        let provider = resolve_provider(&node, &model);
        assert_eq!(provider, "gemini");
    }

    // -- Trait bounds / structural checks -----------------------------------------

    /// Verify LlmCodergenBackend::from_env() never panics, regardless of which
    /// API keys are present in the environment.
    #[test]
    fn from_env_does_not_panic() {
        let result = std::panic::catch_unwind(|| LlmCodergenBackend::from_env("gpt-4o"));
        assert!(
            result.is_ok(),
            "from_env() must not panic regardless of env vars"
        );
    }

    /// Compile-check: confirms `LlmCodergenBackend` implements CodergenBackend.
    #[test]
    fn llm_codergen_backend_is_codergen_backend() {
        fn assert_is_codergen_backend<T: CodergenBackend>() {}
        assert_is_codergen_backend::<LlmCodergenBackend>();
    }

    /// Reasoning effort: "high" (parser default) should NOT be forwarded.
    #[test]
    fn reasoning_effort_high_not_forwarded() {
        let mut node = Node::default();
        node.reasoning_effort = "high".to_string();

        // The run() logic: if !node.reasoning_effort.is_empty() && node.reasoning_effort != "high"
        let should_forward = !node.reasoning_effort.is_empty() && node.reasoning_effort != "high";
        assert!(!should_forward, "\"high\" must not be forwarded to config");
    }

    /// Reasoning effort: "low" should be forwarded.
    #[test]
    fn reasoning_effort_low_forwarded() {
        let mut node = Node::default();
        node.reasoning_effort = "low".to_string();

        let should_forward = !node.reasoning_effort.is_empty() && node.reasoning_effort != "high";
        assert!(should_forward, "\"low\" must be forwarded to config");
    }

    /// Reasoning effort: empty should NOT be forwarded.
    #[test]
    fn reasoning_effort_empty_not_forwarded() {
        let node = Node::default();

        let should_forward = !node.reasoning_effort.is_empty() && node.reasoning_effort != "high";
        assert!(!should_forward, "empty must not be forwarded to config");
    }
}
