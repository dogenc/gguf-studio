//! Lädt einzelne Gewichtstensoren nach llama.cpp-Namenskonvention und
//! dequantisiert sie bei Bedarf. Jede Ladung greift nur auf den mmap-Slice
//! des jeweils benötigten Tensors zu (kein Volllesen der Datei).

use anyhow::{anyhow, Result};
use gguf_core::{dequant, GgufFile, TensorInfo};

pub struct Weights<'a> {
    file: &'a GgufFile,
}

impl<'a> Weights<'a> {
    pub fn new(file: &'a GgufFile) -> Self {
        Self { file }
    }

    fn find(&self, name: &str) -> Result<&TensorInfo> {
        self.file
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("Tensor '{name}' nicht in Datei gefunden"))
    }

    /// Lädt einen Tensor vollständig als f32-Vektor (dequantisiert).
    pub fn load(&self, name: &str) -> Result<Vec<f32>> {
        let t = self.find(name)?;
        let n = t.n_elements() as usize;
        let raw = self.file.tensor_bytes(t, 0, t.size_bytes as usize)?;
        Ok(dequant::dequantize_full(t.dtype, raw, n))
    }

    /// Wie `load`, liefert zusätzlich die Shape (Rust-Reihenfolge: GGUF
    /// speichert Shapes in umgekehrter (ne0 zuerst) Reihenfolge).
    pub fn load_with_shape(&self, name: &str) -> Result<(Vec<f32>, Vec<u64>)> {
        let t = self.find(name)?;
        let n = t.n_elements() as usize;
        let raw = self.file.tensor_bytes(t, 0, t.size_bytes as usize)?;
        Ok((dequant::dequantize_full(t.dtype, raw, n), t.shape.clone()))
    }

    pub fn has(&self, name: &str) -> bool {
        self.file.tensors.iter().any(|t| t.name == name)
    }

    /// Absoluter Datei-Offset (ab Dateianfang) und Byte-Länge des kompletten
    /// Tensors — Grundlage für den Sprung in den Hex-Viewer.
    pub fn file_range(&self, name: &str) -> Result<(u64, u64)> {
        let t = self.find(name)?;
        Ok((self.file.data_offset + t.offset, t.size_bytes))
    }

    /// Wie `file_range`, aber nur für eine einzelne Zeile einer
    /// [n_vocab, n_embd]-Matrix (siehe `embedding_row`).
    pub fn file_range_for_row(&self, name: &str, row: u64, n_embd: u64) -> Result<(u64, u64)> {
        let t = self.find(name)?;
        let (block_n, block_bytes) = t.dtype.block_layout();
        let row_bytes = (n_embd as usize).div_ceil(block_n) as u64 * block_bytes as u64;
        let offset = self.file.data_offset + t.offset + row * row_bytes;
        Ok((offset, row_bytes))
    }

    /// Einzelne Zeile (einen Token-Embedding-Vektor) aus einer
    /// [n_vocab, n_embd]-Matrix ohne die gesamte Matrix zu dequantisieren —
    /// wichtig, da `token_embd.weight` bei großen Modellen mehrere hundert
    /// MB groß ist.
    pub fn embedding_row(&self, name: &str, row: u64, n_embd: u64) -> Result<Vec<f32>> {
        let t = self.find(name)?;
        let (block_n, block_bytes) = t.dtype.block_layout();
        let row_bytes = (n_embd as usize).div_ceil(block_n) * block_bytes;
        let offset = row as u64 * row_bytes as u64;
        let raw = self.file.tensor_bytes(t, offset, row_bytes)?;
        Ok(dequant::dequantize_full(t.dtype, raw, n_embd as usize))
    }
}

/// Übliche llama.cpp-Tensornamen für Layer `i`.
pub struct LayerNames {
    pub attn_norm: String,
    pub attn_q: String,
    pub attn_k: String,
    pub attn_v: String,
    pub attn_output: String,
    pub ffn_norm: String,
    pub ffn_gate: String,
    pub ffn_up: String,
    pub ffn_down: String,
}

impl LayerNames {
    pub fn for_layer(i: u64) -> Self {
        Self {
            attn_norm: format!("blk.{i}.attn_norm.weight"),
            attn_q: format!("blk.{i}.attn_q.weight"),
            attn_k: format!("blk.{i}.attn_k.weight"),
            attn_v: format!("blk.{i}.attn_v.weight"),
            attn_output: format!("blk.{i}.attn_output.weight"),
            ffn_norm: format!("blk.{i}.ffn_norm.weight"),
            ffn_gate: format!("blk.{i}.ffn_gate.weight"),
            ffn_up: format!("blk.{i}.ffn_up.weight"),
            ffn_down: format!("blk.{i}.ffn_down.weight"),
        }
    }
}
