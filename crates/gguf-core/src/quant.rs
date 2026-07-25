//! Quantisierungs-Analyzer: Kennzahlen & Erklärtexte je GGML-Typ.

use crate::{GgmlType, GgufFile};
use std::collections::BTreeMap;

#[derive(Debug, Clone, serde::Serialize)]
pub struct QuantStats {
    pub dtype: GgmlType,
    pub tensor_count: usize,
    pub total_bytes: u64,
    pub total_elements: u64,
    pub bits_per_weight: f64,
}

/// Aggregiert die Quantisierungsverteilung einer Datei.
pub fn analyze(file: &GgufFile) -> Vec<QuantStats> {
    let mut map: BTreeMap<String, QuantStats> = BTreeMap::new();
    for t in &file.tensors {
        let key = format!("{:?}", t.dtype);
        let e = map.entry(key).or_insert_with(|| QuantStats {
            dtype: t.dtype,
            tensor_count: 0,
            total_bytes: 0,
            total_elements: 0,
            bits_per_weight: t.dtype.bits_per_weight(),
        });
        e.tensor_count += 1;
        e.total_bytes += t.size_bytes;
        e.total_elements += t.n_elements();
    }
    let mut v: Vec<_> = map.into_values().collect();
    v.sort_by(|a, b| b.total_bytes.cmp(&a.total_bytes));
    v
}

/// Kompressionsrate gegenüber F32.
pub fn compression_vs_f32(stats: &[QuantStats]) -> f64 {
    let bytes: u64 = stats.iter().map(|s| s.total_bytes).sum();
    let elems: u64 = stats.iter().map(|s| s.total_elements).sum();
    if bytes == 0 { return 1.0; }
    (elems as f64 * 4.0) / bytes as f64
}

/// Relative Qualitäts- und Geschwindigkeits-Schätzwerte (0.0–1.0) je
/// Quant-Typ, basierend auf bekannten Community-Benchmarks (llama.cpp).
/// Dient der groben visuellen Einordnung, nicht als exakte Messung.
pub fn relative_quality(t: GgmlType) -> f32 {
    use GgmlType::*;
    match t {
        F32 | F64 => 1.0,
        BF16 | F16 => 0.99,
        Q8_0 | Q8_1 | Q8K => 0.97,
        Q6K => 0.95,
        Q5K | Q5_1 => 0.92,
        Q5_0 => 0.90,
        IQ4NL | IQ4XS => 0.89,
        Q4K | Q4_1 => 0.87,
        Q4_0 => 0.84,
        IQ3S | IQ3XXS => 0.78,
        Q3K => 0.75,
        IQ2S | IQ2XS => 0.65,
        Q2K => 0.60,
        IQ2XXS => 0.55,
        IQ1S | IQ1M => 0.40,
        I8 | I16 | I32 | I64 => 0.99,
        Unknown(_) => 0.5,
    }
}

/// Relative Inferenzgeschwindigkeit (höher = schneller), grobe Einordnung.
pub fn relative_speed(t: GgmlType) -> f32 {
    use GgmlType::*;
    match t {
        F32 | F64 => 0.4,
        BF16 | F16 => 0.55,
        Q8_0 | Q8_1 | Q8K => 0.75,
        Q6K => 0.82,
        Q5K | Q5_0 | Q5_1 => 0.85,
        Q4K | Q4_0 | Q4_1 | IQ4NL | IQ4XS => 0.95,
        Q3K | IQ3S | IQ3XXS => 0.97,
        Q2K | IQ2S | IQ2XS | IQ2XXS => 1.0,
        IQ1S | IQ1M => 1.0,
        I8 | I16 | I32 | I64 => 0.9,
        Unknown(_) => 0.5,
    }
}

/// Ein benannter Byte-Abschnitt innerhalb eines einzelnen Quantisierungs-
/// Blocks (z. B. "Skala", "gepackte 4-Bit-Gewichte") — Grundlage für ein
/// visuelles Bit-Layout-Diagramm.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BitField {
    pub name: String,
    pub bytes: f32,
    pub description: String,
}

/// Beschreibt, wie die Bytes eines einzelnen Quantisierungs-Blocks
/// aufgeteilt sind — für die gängigsten Typen exakt nach dem
/// llama.cpp/ggml-Blocklayout, für seltenere Typen nur grob.
pub fn bit_layout(t: GgmlType) -> Vec<BitField> {
    use GgmlType::*;
    match t {
        F32 => vec![
            BitField { name: "f32-Wert".into(), bytes: 4.0, description: "Eine einzelne 32-Bit-Gleitkommazahl, volle Genauigkeit, kein Packen mit anderen Werten.".into() },
        ],
        F16 | BF16 => vec![
            BitField { name: "f16/bf16-Wert".into(), bytes: 2.0, description: "Eine 16-Bit-Gleitkommazahl (halbe Genauigkeit gegenüber F32).".into() },
        ],
        F64 => vec![
            BitField { name: "f64-Wert".into(), bytes: 8.0, description: "Eine 64-Bit-Gleitkommazahl, in LLMs selten verwendet.".into() },
        ],
        Q4_0 => vec![
            BitField { name: "Skala (f16)".into(), bytes: 2.0, description: "Ein gemeinsamer Skalierungsfaktor für den gesamten 32er-Block.".into() },
            BitField { name: "32× 4-Bit-Gewicht".into(), bytes: 16.0, description: "32 vorzeichenlose 4-Bit-Werte (0-15), zu je 2 pro Byte gepackt. Tatsächlicher Wert = (q − 8) × Skala.".into() },
        ],
        Q4_1 => vec![
            BitField { name: "Skala (f16)".into(), bytes: 2.0, description: "Skalierungsfaktor für den 32er-Block.".into() },
            BitField { name: "Minimum (f16)".into(), bytes: 2.0, description: "Zusätzlicher Offset-Wert (im Gegensatz zu Q4_0 nicht symmetrisch um 0).".into() },
            BitField { name: "32× 4-Bit-Gewicht".into(), bytes: 16.0, description: "32 gepackte 4-Bit-Werte. Tatsächlicher Wert = q × Skala + Minimum.".into() },
        ],
        Q5_0 => vec![
            BitField { name: "Skala (f16)".into(), bytes: 2.0, description: "Gemeinsamer Skalierungsfaktor für den 32er-Block.".into() },
            BitField { name: "High-Bit-Maske".into(), bytes: 4.0, description: "Das jeweils 5. (höchste) Bit aller 32 Gewichte, separat gepackt.".into() },
            BitField { name: "32× 4 Low-Bits".into(), bytes: 16.0, description: "Die unteren 4 Bit jedes Gewichts; zusammen mit der High-Bit-Maske ergibt sich ein 5-Bit-Wert.".into() },
        ],
        Q8_0 => vec![
            BitField { name: "Skala (f16)".into(), bytes: 2.0, description: "Gemeinsamer Skalierungsfaktor für den 32er-Block.".into() },
            BitField { name: "32× 8-Bit-Gewicht".into(), bytes: 32.0, description: "32 vorzeichenbehaftete Bytes (Int8). Tatsächlicher Wert = q × Skala.".into() },
        ],
        Q4K => vec![
            BitField { name: "d (f16)".into(), bytes: 2.0, description: "Globale Skala des 256er-Superblocks.".into() },
            BitField { name: "dmin (f16)".into(), bytes: 2.0, description: "Globaler Minimalwert-Offset des Superblocks.".into() },
            BitField { name: "Sub-Skalen (6-Bit)".into(), bytes: 12.0, description: "8 gepackte 6-Bit-Sub-Skalen, je eine für 32 Werte innerhalb des 256er-Blocks — feingranularer als eine einzige globale Skala.".into() },
            BitField { name: "256× 4-Bit-Gewicht".into(), bytes: 128.0, description: "256 4-Bit-Werte, zu je 2 pro Byte gepackt.".into() },
        ],
        Q5K => vec![
            BitField { name: "d / dmin (f16)".into(), bytes: 4.0, description: "Globale Skala und Offset des 256er-Superblocks.".into() },
            BitField { name: "Sub-Skalen (6-Bit)".into(), bytes: 12.0, description: "8 gepackte 6-Bit-Sub-Skalen für je 32 Werte.".into() },
            BitField { name: "High-Bit-Maske".into(), bytes: 32.0, description: "Das 5. Bit jedes der 256 Werte.".into() },
            BitField { name: "256× 4 Low-Bits".into(), bytes: 128.0, description: "Untere 4 Bit jedes Werts.".into() },
        ],
        Q6K => vec![
            BitField { name: "ql (Low-Bits)".into(), bytes: 128.0, description: "Untere 4 Bit von 256 Gewichten, zu je 2 pro Byte gepackt.".into() },
            BitField { name: "qh (High-Bits)".into(), bytes: 64.0, description: "Obere 2 Bit von 256 Gewichten, zu je 4 pro Byte gepackt.".into() },
            BitField { name: "Sub-Skalen (i8)".into(), bytes: 16.0, description: "16 vorzeichenbehaftete Sub-Skalen, je eine für 16 Werte.".into() },
            BitField { name: "d (f16)".into(), bytes: 2.0, description: "Globale Skala des 256er-Superblocks.".into() },
        ],
        Q8K => vec![
            BitField { name: "d (f32)".into(), bytes: 4.0, description: "Globale Skala des 256er-Superblocks.".into() },
            BitField { name: "256× 8-Bit-Gewicht".into(), bytes: 256.0, description: "256 vorzeichenbehaftete Bytes (Int8).".into() },
            BitField { name: "Sub-Summen (i16)".into(), bytes: 32.0, description: "Zwischensummen je 16er-Gruppe, beschleunigen die Dequantisierung.".into() },
        ],
        I8 => vec![BitField { name: "i8-Wert".into(), bytes: 1.0, description: "Ganzzahl, 1 Byte — meist ein Hilfstensor, keine Modellgewichte.".into() }],
        I16 => vec![BitField { name: "i16-Wert".into(), bytes: 2.0, description: "Ganzzahl, 2 Byte.".into() }],
        I32 => vec![BitField { name: "i32-Wert".into(), bytes: 4.0, description: "Ganzzahl, 4 Byte.".into() }],
        I64 => vec![BitField { name: "i64-Wert".into(), bytes: 8.0, description: "Ganzzahl, 8 Byte.".into() }],
        other => {
            let (n, b) = other.block_layout();
            vec![BitField {
                name: format!("{n} gepackte Werte"),
                bytes: b as f32,
                description: "Importance-/IQ-Quantisierung mit Codebook-basierter Kompression — das exakte \
                    Bit-Layout ist komplexer als ein einfaches Skala+Werte-Schema und hier nicht im \
                    Detail nachgebildet.".into(),
            }]
        }
    }
}

pub fn describe(t: GgmlType) -> &'static str {
    use GgmlType::*;
    match t {
        F32 => "32-Bit-Gleitkomma, unquantisiert. Referenzgenauigkeit, höchster Speicherbedarf.",
        F16 => "16-Bit-Gleitkomma. Halber Speicher von F32 bei minimalem Qualitätsverlust.",
        BF16 => "bfloat16 — F32-Exponentenbereich mit 8-Bit-Mantisse; trainingsnah.",
        Q8_0 | Q8_1 | Q8K => "8-Bit-Quantisierung. Nahezu verlustfrei, ~4x kleiner als F32.",
        Q6K => "6-Bit K-Quantisierung. Sehr gute Qualität, guter Kompromiss.",
        Q5_0 | Q5_1 | Q5K => "5-Bit. Gute Qualität bei deutlicher Kompression.",
        Q4_0 | Q4_1 | Q4K => "4-Bit. Standard für Consumer-Hardware; leichte Qualitätseinbußen.",
        Q3K => "3-Bit K-Quant. Stark komprimiert, merkbare Einbußen bei kleinen Modellen.",
        Q2K => "2-Bit K-Quant. Maximale Kompression, deutliche Qualitätsverluste.",
        IQ1S | IQ1M => "~1,5–1,75 Bit (importance-basiert). Experimentell, extreme Kompression.",
        IQ2XXS | IQ2XS | IQ2S => "~2-Bit importance-Quantisierung mit Codebooks.",
        IQ3XXS | IQ3S => "~3-Bit importance-Quantisierung; besser als klassisches Q3.",
        IQ4NL | IQ4XS => "~4-Bit nichtlinear; oft bessere Qualität als Q4_K bei gleicher Größe.",
        F64 => "64-Bit-Gleitkomma (selten in LLMs).",
        I8 | I16 | I32 | I64 => "Ganzzahl-Datentyp (Hilfstensoren).",
        Unknown(_) => "Unbekannter/zukünftiger Quantisierungstyp.",
    }
}
