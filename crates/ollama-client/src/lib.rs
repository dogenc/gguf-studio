//! Async-Client für die lokale Ollama-REST-API (Standard: http://127.0.0.1:11434).
//! Unterstützt: Modellliste, Details, Generate (Streaming), Create/Rebuild,
//! Laden/Entladen und einfache Benchmarks.

pub mod store;

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone)]
pub struct OllamaClient {
    base: String,
    http: reqwest::Client,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelSummary {
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(default)]
    pub modified_at: String,
    #[serde(default)]
    pub details: ModelDetails,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelDetails {
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ShowResponse {
    #[serde(default)]
    pub modelfile: String,
    #[serde(default)]
    pub parameters: String,
    #[serde(default)]
    pub template: String,
    #[serde(default)]
    pub details: ModelDetails,
    /// Rohmetadaten (Architektur, Kontextfenster, Embedding-Länge …)
    #[serde(default)]
    pub model_info: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct GenerateOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<i32>,
}

#[derive(Debug, Clone, Default)]
pub struct BenchResult {
    pub total_secs: f64,
    pub tokens: u64,
    pub tokens_per_sec: f64,
}

impl OllamaClient {
    pub fn new(base: impl Into<String>) -> Self {
        Self { base: base.into(), http: reqwest::Client::new() }
    }

    pub fn local() -> Self {
        Self::new("http://127.0.0.1:11434")
    }

    pub async fn list_models(&self) -> Result<Vec<ModelSummary>> {
        #[derive(Deserialize)]
        struct Tags { models: Vec<ModelSummary> }
        let r: Tags = self.http.get(format!("{}/api/tags", self.base))
            .send().await.context("Ollama nicht erreichbar")?
            .error_for_status()?.json().await?;
        Ok(r.models)
    }

    pub async fn show(&self, name: &str) -> Result<ShowResponse> {
        Ok(self.http.post(format!("{}/api/show", self.base))
            .json(&serde_json::json!({ "name": name }))
            .send().await?.error_for_status()?.json().await?)
    }

    /// Streamt eine Antwort; `on_token` wird pro Chunk aufgerufen.
    pub async fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        options: Option<GenerateOptions>,
        mut on_token: impl FnMut(&str),
    ) -> Result<BenchResult> {
        let start = Instant::now();
        let mut body = serde_json::json!({ "model": model, "prompt": prompt, "stream": true });
        if let Some(o) = options {
            body["options"] = serde_json::to_value(o)?;
        }
        let resp = self.http.post(format!("{}/api/generate", self.base))
            .json(&body).send().await?.error_for_status()?;

        let mut tokens = 0u64;
        let mut stream = resp.bytes_stream();
        let mut buf = Vec::new();
        while let Some(chunk) = stream.next().await {
            buf.extend_from_slice(&chunk?);
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&line) {
                    if let Some(t) = v.get("response").and_then(|r| r.as_str()) {
                        tokens += 1;
                        on_token(t);
                    }
                    if let Some(n) = v.get("eval_count").and_then(|n| n.as_u64()) {
                        tokens = n;
                    }
                }
            }
        }
        let total_secs = start.elapsed().as_secs_f64();
        Ok(BenchResult { total_secs, tokens, tokens_per_sec: tokens as f64 / total_secs.max(1e-9) })
    }

    /// Modell aus einem Modelfile (neu) erstellen — Rebuild nach Editor-Änderungen.
    pub async fn create(&self, name: &str, modelfile: &str) -> Result<()> {
        self.http.post(format!("{}/api/create", self.base))
            .json(&serde_json::json!({ "name": name, "modelfile": modelfile, "stream": false }))
            .send().await?.error_for_status()?;
        Ok(())
    }

    /// Modell entladen (keep_alive: 0).
    pub async fn unload(&self, name: &str) -> Result<()> {
        self.http.post(format!("{}/api/generate", self.base))
            .json(&serde_json::json!({ "model": name, "keep_alive": 0 }))
            .send().await?.error_for_status()?;
        Ok(())
    }
}

/// Minimaler Modelfile-Validator für den Editor.
pub fn validate_modelfile(src: &str) -> Vec<String> {
    const DIRECTIVES: &[&str] = &["FROM", "PARAMETER", "TEMPLATE", "SYSTEM", "ADAPTER", "LICENSE", "MESSAGE"];
    let mut errors = Vec::new();
    let mut has_from = false;
    for (i, line) in src.lines().enumerate() {
        let l = line.trim();
        if l.is_empty() || l.starts_with('#') { continue; }
        let word = l.split_whitespace().next().unwrap_or("");
        let upper = word.to_ascii_uppercase();
        if upper == "FROM" { has_from = true; }
        if !DIRECTIVES.contains(&upper.as_str()) && !l.starts_with('"') {
            errors.push(format!("Zeile {}: unbekannte Direktive '{}'", i + 1, word));
        }
        if upper == "PARAMETER" && l.split_whitespace().count() < 3 {
            errors.push(format!("Zeile {}: PARAMETER benötigt Name und Wert", i + 1));
        }
    }
    if !has_from {
        errors.push("Fehlende FROM-Direktive".into());
    }
    errors
}
