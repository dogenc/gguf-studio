//! Grundoperationen eines Transformer-Forward-Pass, in reinem f32,
//! implementiert genau wie in llama.cpp/ggml (RMSNorm, RoPE, GQA-Attention,
//! SwiGLU-FFN) — bewusst unoptimiert (keine SIMD/BLAS), da diese Engine der
//! Veranschaulichung dient, nicht der Produktions-Inferenz.

/// y = x / rms(x) * weight, mit rms(x) = sqrt(mean(x^2) + eps).
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(weight).map(|(&v, &w)| v * scale * w).collect()
}

/// y = W x, mit W als [out, in]-Matrix in row-major (GGUF-Speicherordnung
/// für 2D-Gewichte: `weight[o * n_in + i]`).
pub fn mat_vec(weight: &[f32], n_in: usize, n_out: usize, x: &[f32]) -> Vec<f32> {
    debug_assert_eq!(x.len(), n_in);
    let mut out = vec![0.0f32; n_out];
    for o in 0..n_out {
        let row = &weight[o * n_in..(o + 1) * n_in];
        out[o] = row.iter().zip(x).map(|(&w, &v)| w * v).sum();
    }
    out
}

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&x, &y)| x + y).collect()
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU: down( silu(gate(x)) * up(x) ).
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up).map(|(&g, &u)| silu(g) * u).collect()
}

/// RoPE (rotary position embedding), NEOX-Stil (Paare (i, i+d/2)), wie in
/// llama.cpp für Llama-artige Modelle Standard.
pub fn rope_inplace(vec: &mut [f32], pos: usize, head_dim: usize, theta_base: f32) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta_base.powf((2 * i) as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        let (sin, cos) = angle.sin_cos();
        let x0 = vec[i];
        let x1 = vec[i + half];
        vec[i] = x0 * cos - x1 * sin;
        vec[i + half] = x0 * sin + x1 * cos;
    }
}

pub fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}
