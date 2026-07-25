//! Liest die Architektur-Hyperparameter aus den GGUF-Metadaten (llama.cpp-
//! Konvention: `<arch>.<feld>`, z. B. `llama.attention.head_count`).

use anyhow::{anyhow, Result};
use gguf_core::value::GgufValue;
use gguf_core::GgufFile;

#[derive(Debug, Clone)]
pub struct ArchConfig {
    pub name: String,
    pub n_layer: u64,
    pub n_embd: u64,
    pub n_head: u64,
    pub n_head_kv: u64,
    pub n_ff: u64,
    pub n_vocab: u64,
    pub rms_eps: f32,
    pub rope_theta: f32,
    pub head_dim: u64,
}

fn meta_u64(file: &GgufFile, arch: &str, key: &str) -> Option<u64> {
    file.metadata.get(&format!("{arch}.{key}")).and_then(GgufValue::as_u64)
}

fn meta_f64(file: &GgufFile, arch: &str, key: &str) -> Option<f64> {
    file.metadata.get(&format!("{arch}.{key}")).and_then(GgufValue::as_f64)
}

impl ArchConfig {
    /// Liest die für einen Forward-Pass nötigen Hyperparameter. Unterstützt
    /// das llama.cpp-Metadatenschema, das die meisten GGUF-Architekturen
    /// (Llama, Mistral, Qwen, Gemma …) gemeinsam verwenden.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let name = file.architecture().ok_or_else(|| anyhow!("general.architecture fehlt"))?.to_string();
        let n_layer = meta_u64(file, &name, "block_count")
            .ok_or_else(|| anyhow!("{name}.block_count fehlt"))?;
        let n_embd = meta_u64(file, &name, "embedding_length")
            .ok_or_else(|| anyhow!("{name}.embedding_length fehlt"))?;
        let n_head = meta_u64(file, &name, "attention.head_count")
            .ok_or_else(|| anyhow!("{name}.attention.head_count fehlt"))?;
        let n_head_kv = meta_u64(file, &name, "attention.head_count_kv").unwrap_or(n_head);
        let n_ff = meta_u64(file, &name, "feed_forward_length").unwrap_or(n_embd * 4);
        let n_vocab = file.metadata.get("tokenizer.ggml.tokens")
            .and_then(|v| match v { GgufValue::Array(a) => Some(a.len() as u64), _ => None })
            .unwrap_or(0);
        let rms_eps = meta_f64(file, &name, "attention.layer_norm_rms_epsilon").unwrap_or(1e-5) as f32;
        let rope_theta = meta_f64(file, &name, "rope.freq_base").unwrap_or(10000.0) as f32;
        let head_dim = n_embd / n_head.max(1);

        Ok(Self { name, n_layer, n_embd, n_head, n_head_kv, n_ff, n_vocab, rms_eps, rope_theta, head_dim })
    }

    pub fn is_gqa(&self) -> bool {
        self.n_head_kv < self.n_head
    }
}
