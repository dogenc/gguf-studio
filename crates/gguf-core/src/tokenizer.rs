//! Tokenizer-Explorer: liest `tokenizer.ggml.*`-Metadaten aus einer GGUF-Datei
//! und stellt Tokens, IDs, Byte-Repräsentationen und Special Tokens bereit.

use crate::value::GgufValue;
use crate::GgufFile;

#[derive(Debug, Clone, serde::Serialize)]
pub struct TokenEntry {
    pub id: u32,
    pub text: String,
    pub bytes: Vec<u8>,
    pub score: Option<f32>,
    pub kind: TokenKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum TokenKind {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl TokenKind {
    fn from_u32(v: u32) -> Self {
        match v {
            2 => TokenKind::Unknown,
            3 => TokenKind::Control,
            4 => TokenKind::UserDefined,
            5 => TokenKind::Unused,
            6 => TokenKind::Byte,
            _ => TokenKind::Normal,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TokenKind::Normal => "Normal",
            TokenKind::Unknown => "Unknown",
            TokenKind::Control => "Control",
            TokenKind::UserDefined => "User-Defined",
            TokenKind::Unused => "Unused",
            TokenKind::Byte => "Byte",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SpecialToken {
    pub name: &'static str,
    pub id: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TokenizerInfo {
    pub model: Option<String>,
    pub tokens: Vec<TokenEntry>,
    pub special: Vec<SpecialToken>,
}

impl TokenizerInfo {
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

/// Extrahiert die Tokenizer-Informationen aus den GGUF-Metadaten, sofern vorhanden.
pub fn extract(file: &GgufFile) -> Option<TokenizerInfo> {
    let tokens_val = file.metadata.get("tokenizer.ggml.tokens")?;
    let GgufValue::Array(tokens_arr) = tokens_val else { return None };

    let scores: Option<&Vec<GgufValue>> = file
        .metadata
        .get("tokenizer.ggml.scores")
        .and_then(|v| match v {
            GgufValue::Array(a) => Some(a),
            _ => None,
        });
    let types: Option<&Vec<GgufValue>> = file
        .metadata
        .get("tokenizer.ggml.token_type")
        .and_then(|v| match v {
            GgufValue::Array(a) => Some(a),
            _ => None,
        });

    let mut tokens = Vec::with_capacity(tokens_arr.len());
    for (i, t) in tokens_arr.iter().enumerate() {
        let text = t.as_str().unwrap_or_default().to_string();
        let score = scores
            .and_then(|s| s.get(i))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        let kind = types
            .and_then(|s| s.get(i))
            .and_then(|v| v.as_u64())
            .map(|v| TokenKind::from_u32(v as u32))
            .unwrap_or(TokenKind::Normal);
        tokens.push(TokenEntry {
            id: i as u32,
            bytes: text.as_bytes().to_vec(),
            text,
            score,
            kind,
        });
    }

    let model = file
        .metadata
        .get("tokenizer.ggml.model")
        .and_then(GgufValue::as_str)
        .map(|s| s.to_string());

    let mut special = Vec::new();
    const SPECIAL_KEYS: &[(&str, &str)] = &[
        ("tokenizer.ggml.bos_token_id", "BOS"),
        ("tokenizer.ggml.eos_token_id", "EOS"),
        ("tokenizer.ggml.unknown_token_id", "UNK"),
        ("tokenizer.ggml.padding_token_id", "PAD"),
        ("tokenizer.ggml.separator_token_id", "SEP"),
        ("tokenizer.ggml.cls_token_id", "CLS"),
        ("tokenizer.ggml.mask_token_id", "MASK"),
    ];
    for (key, label) in SPECIAL_KEYS {
        if let Some(id) = file.metadata.get(*key).and_then(GgufValue::as_u64) {
            special.push(SpecialToken { name: label, id });
        }
    }

    Some(TokenizerInfo { model, tokens, special })
}

/// Byte-Hex-Darstellung eines Tokens, z. B. "48 65 6C 6C 6F".
pub fn bytes_hex(entry: &TokenEntry) -> String {
    entry
        .bytes
        .iter()
        .map(|b| format!("{b:02X}"))
        .collect::<Vec<_>>()
        .join(" ")
}
