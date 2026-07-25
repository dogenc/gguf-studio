//! Dequantisierung und Statistik-Berechnung für Tensor-Vorschau.
//! Arbeitet ausschließlich auf begrenzten Blöcken (Zero-Copy-Slices aus dem
//! Mmap) — es wird niemals ein ganzer Tensor auf einmal materialisiert,
//! sondern nur so viele Blöcke wie für die angeforderte Elementanzahl nötig.

use crate::GgmlType;
use half::f16;

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TensorStats {
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std_dev: f32,
    /// 32 Bins für eine einfache Histogramm-Visualisierung.
    pub histogram: [u32; 32],
}

/// Dequantisiert höchstens `max_elements` Werte ab Blockindex 0 eines Tensors.
/// Liefert leeren Vec bei unbekanntem Typ oder leerem Slice.
pub fn dequantize_preview(dtype: GgmlType, raw: &[u8], max_elements: usize) -> Vec<f32> {
    let (block_n, block_bytes) = dtype.block_layout();
    if block_bytes == 0 || raw.is_empty() {
        return Vec::new();
    }
    let n_blocks_avail = raw.len() / block_bytes;
    let n_blocks_needed = max_elements.div_ceil(block_n).min(n_blocks_avail);
    let mut out = Vec::with_capacity(n_blocks_needed * block_n);

    for b in 0..n_blocks_needed {
        let block = &raw[b * block_bytes..(b + 1) * block_bytes];
        decode_block(dtype, block, &mut out);
        if out.len() >= max_elements {
            break;
        }
    }
    out.truncate(max_elements);
    out
}

/// Dequantisiert den kompletten Tensorinhalt aus `raw` (muss mindestens
/// `n_elements` Werte abdecken). Für die Mini-Inferenz benötigt, wo mit den
/// vollständigen Gewichtsmatrizen gerechnet werden muss statt nur einer
/// Stichprobe.
pub fn dequantize_full(dtype: GgmlType, raw: &[u8], n_elements: usize) -> Vec<f32> {
    dequantize_preview(dtype, raw, n_elements)
}

fn decode_block(dtype: GgmlType, block: &[u8], out: &mut Vec<f32>) {
    use GgmlType::*;
    match dtype {
        F32 => out.push(f32::from_le_bytes(block.try_into().unwrap_or_default())),
        F16 => out.push(f16::from_le_bytes(block.try_into().unwrap_or_default()).to_f32()),
        BF16 => {
            let bits = u16::from_le_bytes(block.try_into().unwrap_or_default());
            out.push(f32::from_bits((bits as u32) << 16));
        }
        F64 => out.push(f64::from_le_bytes(block.try_into().unwrap_or_default()) as f32),
        I8 => out.push(block[0] as i8 as f32),
        I16 => out.push(i16::from_le_bytes(block.try_into().unwrap_or_default()) as f32),
        I32 => out.push(i32::from_le_bytes(block.try_into().unwrap_or_default()) as f32),
        I64 => out.push(i64::from_le_bytes(block.try_into().unwrap_or_default()) as f32),
        Q4_0 => decode_q4_0(block, out),
        Q8_0 => decode_q8_0(block, out),
        Q4K => decode_q4_k(block, out),
        Q6K => decode_q6_k(block, out),
        // Andere K-/IQ-Quants: vereinfachte Rohbyte-Anzeige als normalisierte
        // Approximation, bis dedizierte Dekoder ergänzt werden. Für die
        // Mini-Inferenz führt das zu ungenauen (aber lauffähigen) Ergebnissen
        // bei diesen selteneren Typen.
        _ => {
            for &b in block {
                out.push((b as f32 - 128.0) / 128.0);
            }
        }
    }
}

/// Q6_K: 256 Elemente pro Block (llama.cpp `block_q6_K`-Layout):
/// 128 Byte `ql` (Low-4-Bit, 2 Elemente/Byte), 64 Byte `qh` (High-2-Bit,
/// 4 Elemente/Byte), 16 Byte `scales` (i8, 16 Sub-Blöcke à 16 Elemente),
/// 1x f16 Gesamt-Skala `d`. Werte liegen im Bereich [-32, 31].
fn decode_q6_k(block: &[u8], out: &mut Vec<f32>) {
    if block.len() < 210 {
        return;
    }
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

    // Referenz: llama.cpp dequantize_row_q6_K — zwei Hälften à 128 Elemente.
    for half in 0..2 {
        let ql_h = &ql[half * 64..half * 64 + 64];
        let qh_h = &qh[half * 32..half * 32 + 32];
        let sc_h = &scales[half * 8..half * 8 + 8];
        for l in 0..32 {
            let is = l / 16;
            let q1 = (((ql_h[l] & 0x0F) as i32) | (((qh_h[l] >> 0) & 0x03) as i32) << 4) - 32;
            let q2 = (((ql_h[l + 32] & 0x0F) as i32) | (((qh_h[l] >> 2) & 0x03) as i32) << 4) - 32;
            let q3 = (((ql_h[l] >> 4) as i32) | (((qh_h[l] >> 4) & 0x03) as i32) << 4) - 32;
            let q4 = (((ql_h[l + 32] >> 4) as i32) | (((qh_h[l] >> 6) & 0x03) as i32) << 4) - 32;
            out.push(d * sc_h[is] as i8 as f32 * q1 as f32);
            out.push(d * sc_h[is + 2] as i8 as f32 * q2 as f32);
            out.push(d * sc_h[is + 4] as i8 as f32 * q3 as f32);
            out.push(d * sc_h[is + 6] as i8 as f32 * q4 as f32);
        }
    }
}

/// Q4_0: 1x f16 Skala + 32x 4-Bit-Gewichte (16 Bytes gepackt).
fn decode_q4_0(block: &[u8], out: &mut Vec<f32>) {
    if block.len() < 2 + 16 {
        return;
    }
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for &byte in &block[2..18] {
        let lo = (byte & 0x0F) as i32 - 8;
        let hi = ((byte >> 4) & 0x0F) as i32 - 8;
        out.push(lo as f32 * scale);
        out.push(hi as f32 * scale);
    }
}

/// Q8_0: 1x f16 Skala + 32x i8-Gewichte.
fn decode_q8_0(block: &[u8], out: &mut Vec<f32>) {
    if block.len() < 2 + 32 {
        return;
    }
    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for &byte in &block[2..34] {
        out.push(byte as i8 as f32 * scale);
    }
}

/// Q4_K (vereinfacht): 2x f16 (d, dmin) + 12 Byte Sub-Skalen + 128 Byte
/// gepackte 4-Bit-Werte über 256 Elemente. Nutzt die globale Skala je
/// Sub-Block grob approximiert, ausreichend für eine Vorschau/Statistik.
fn decode_q4_k(block: &[u8], out: &mut Vec<f32>) {
    if block.len() < 144 {
        return;
    }
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let qs = &block[16..144];
    for &byte in qs {
        let lo = (byte & 0x0F) as f32;
        let hi = ((byte >> 4) & 0x0F) as f32;
        out.push(lo * d - dmin);
        out.push(hi * d - dmin);
    }
}

pub fn compute_stats(values: &[f32]) -> TensorStats {
    if values.is_empty() {
        return TensorStats::default();
    }
    let count = values.len();
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    for &v in values {
        if v < min { min = v; }
        if v > max { max = v; }
        sum += v as f64;
    }
    let mean = (sum / count as f64) as f32;
    let mut var_sum = 0.0f64;
    for &v in values {
        let d = v as f64 - mean as f64;
        var_sum += d * d;
    }
    let std_dev = (var_sum / count as f64).sqrt() as f32;

    let mut histogram = [0u32; 32];
    let range = (max - min).max(f32::EPSILON);
    for &v in values {
        let bucket = (((v - min) / range) * 31.99) as usize;
        histogram[bucket.min(31)] += 1;
    }

    TensorStats { count, min, max, mean, std_dev, histogram }
}
