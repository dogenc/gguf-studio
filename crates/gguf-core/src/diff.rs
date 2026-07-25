//! Modellvergleich: stellt zwei GGUF-Dateien gegenüber und markiert
//! Unterschiede in Header, Metadaten, Tensoren und Tokenizer.

use crate::tokenizer::{self, TokenizerInfo};
use crate::value::GgufValue;
use crate::GgufFile;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum DiffStatus {
    Same,
    Changed,
    OnlyInA,
    OnlyInB,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MetadataDiff {
    pub key: String,
    pub a: Option<String>,
    pub b: Option<String>,
    pub status: DiffStatus,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorDiff {
    pub name: String,
    pub a: Option<(String, Vec<u64>, u64)>, // dtype, shape, size
    pub b: Option<(String, Vec<u64>, u64)>,
    pub status: DiffStatus,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct HeaderDiff {
    pub field: &'static str,
    pub a: String,
    pub b: String,
    pub status: DiffStatus,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelDiff {
    pub header: Vec<HeaderDiff>,
    pub metadata: Vec<MetadataDiff>,
    pub tensors: Vec<TensorDiff>,
    pub tokenizer_vocab_a: Option<usize>,
    pub tokenizer_vocab_b: Option<usize>,
    pub tokenizer_diff_sample: Vec<(u32, Option<String>, Option<String>)>,
}

fn status_of(a: Option<&String>, b: Option<&String>) -> DiffStatus {
    match (a, b) {
        (Some(x), Some(y)) if x == y => DiffStatus::Same,
        (Some(_), Some(_)) => DiffStatus::Changed,
        (Some(_), None) => DiffStatus::OnlyInA,
        (None, Some(_)) => DiffStatus::OnlyInB,
        (None, None) => DiffStatus::Same,
    }
}

fn fmt_val(v: &GgufValue) -> String {
    v.display_short(200)
}

pub fn compare(a: &GgufFile, b: &GgufFile) -> ModelDiff {
    let header = vec![
        header_field("Version", a.version.to_string(), b.version.to_string()),
        header_field("Architektur", a.architecture().unwrap_or("?").to_string(), b.architecture().unwrap_or("?").to_string()),
        header_field("Alignment", a.alignment.to_string(), b.alignment.to_string()),
        header_field("Dateigröße", a.file_size.to_string(), b.file_size.to_string()),
        header_field("Anzahl Tensoren", a.tensors.len().to_string(), b.tensors.len().to_string()),
        header_field("Parameteranzahl", a.param_count().to_string(), b.param_count().to_string()),
    ];

    let mut keys: Vec<&String> = a.metadata.keys().chain(b.metadata.keys()).collect();
    keys.sort();
    keys.dedup();
    let metadata = keys
        .into_iter()
        .map(|k| {
            let av = a.metadata.get(k).map(fmt_val);
            let bv = b.metadata.get(k).map(fmt_val);
            let status = status_of(av.as_ref(), bv.as_ref());
            MetadataDiff { key: k.clone(), a: av, b: bv, status }
        })
        .collect();

    let mut names: Vec<&String> = a.tensors.iter().map(|t| &t.name)
        .chain(b.tensors.iter().map(|t| &t.name))
        .collect();
    names.sort();
    names.dedup();
    let tensors = names
        .into_iter()
        .map(|name| {
            let ta = a.tensors.iter().find(|t| &t.name == name)
                .map(|t| (format!("{:?}", t.dtype), t.shape.clone(), t.size_bytes));
            let tb = b.tensors.iter().find(|t| &t.name == name)
                .map(|t| (format!("{:?}", t.dtype), t.shape.clone(), t.size_bytes));
            let status = match (&ta, &tb) {
                (Some(x), Some(y)) if x == y => DiffStatus::Same,
                (Some(_), Some(_)) => DiffStatus::Changed,
                (Some(_), None) => DiffStatus::OnlyInA,
                (None, Some(_)) => DiffStatus::OnlyInB,
                (None, None) => DiffStatus::Same,
            };
            TensorDiff { name: name.clone(), a: ta, b: tb, status }
        })
        .collect();

    let tok_a: Option<TokenizerInfo> = tokenizer::extract(a);
    let tok_b: Option<TokenizerInfo> = tokenizer::extract(b);
    let mut tokenizer_diff_sample = Vec::new();
    if let (Some(ta), Some(tb)) = (&tok_a, &tok_b) {
        let max_len = ta.tokens.len().max(tb.tokens.len());
        for id in 0..max_len {
            let at = ta.tokens.get(id).map(|t| t.text.clone());
            let bt = tb.tokens.get(id).map(|t| t.text.clone());
            if at != bt {
                tokenizer_diff_sample.push((id as u32, at, bt));
                if tokenizer_diff_sample.len() >= 200 {
                    break;
                }
            }
        }
    }

    ModelDiff {
        header,
        metadata,
        tensors,
        tokenizer_vocab_a: tok_a.map(|t| t.vocab_size()),
        tokenizer_vocab_b: tok_b.map(|t| t.vocab_size()),
        tokenizer_diff_sample,
    }
}

fn header_field(name: &'static str, a: String, b: String) -> HeaderDiff {
    let status = if a == b { DiffStatus::Same } else { DiffStatus::Changed };
    HeaderDiff { field: name, a, b, status }
}
