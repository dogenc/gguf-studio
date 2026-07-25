//! Mini-Forward-Pass-Engine für den Lern-Explorer: rechnet Embedding sowie
//! die ersten `n_layers_to_run` Transformer-Layer *tatsächlich* mit den
//! dequantisierten Gewichten der geladenen GGUF-Datei durch — keine
//! Simulation, sondern echte Matrixmultiplikationen, RMSNorm, RoPE, Softmax-
//! Attention (mit GQA) und SwiGLU-FFN, wie in llama.cpp.
//!
//! Bewusste Grenzen: nur 1–4 Layer, nur ein kurzer Prompt (Größenordnung
//! zwei- bis dreistellig an Tokens) — vollständige Modelle mit 30+ Layern
//! und großem Vokabular würden auf der CPU pro Prompt Sekunden bis Minuten
//! brauchen und sind nicht Ziel dieses Lern-Tools.
//!
//! Angenommenes Tensorlayout: Llama-artige Architekturen (llama.cpp-
//! Namenskonvention `blk.{i}.attn_q.weight` etc.), 2D-Gewichte als
//! `[n_out, n_in]` row-major im GGUF-Datenblock. Modelle mit abweichender
//! Namensgebung (z. B. manche Encoder-Decoder- oder MoE-Architekturen)
//! werden nicht erkannt und liefern einen Fehler statt falscher Zahlen.
//!
//! Jeder tatsächliche Datei-Lesezugriff (jeder dequantisierte Tensor, jede
//! Embedding-Zeile) wird zusätzlich über `on_event` als [`ReadEvent`]
//! gemeldet — unabhängig vom Eingabetext, das ist eine 1:1-Aufzeichnung der
//! echten Hex-Zugriffe, die der aufrufende Code live anzeigen oder als Log
//! exportieren kann.

pub mod arch;
pub mod ops;
pub mod tokenize;
pub mod weights;

use anyhow::{anyhow, Result};
use arch::ArchConfig;
use gguf_core::GgufFile;
use tokenize::TokenizedPiece;
use weights::{LayerNames, Weights};

/// Verweist auf den exakten Datei-Ausschnitt (absoluter Offset ab
/// Dateianfang, Länge in Bytes), aus dem die in diesem Schritt verwendeten
/// Gewichte gelesen wurden — Grundlage für den "Im Hex-Viewer zeigen"-Link.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HexRef {
    pub tensor_name: String,
    pub file_offset: u64,
    pub byte_len: u64,
}

/// Ein einzelner, tatsächlich stattgefundener Datei-Lesezugriff während des
/// Forward-Pass — die Rohdaten für das Live-Log/die Zusammenfassung.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReadEvent {
    pub step_label: String,
    pub tensor_name: String,
    pub file_offset: u64,
    pub byte_len: u64,
    pub reason: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StepTrace {
    pub label: String,
    pub explanation: String,
    /// Kurzer Zahlen-Ausschnitt zur Anzeige (erste paar Werte des Ergebnisvektors).
    pub sample: Vec<f32>,
    pub shape: String,
    /// Datei-Ausschnitte der in diesem Schritt gelesenen Gewichtstensoren.
    pub hex_refs: Vec<HexRef>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AttentionTrace {
    pub layer: u64,
    pub head: u64,
    /// Softmax-Attention-Gewichte für die zuletzt verarbeitete Position,
    /// über alle vorherigen Positionen (inkl. sich selbst).
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ForwardResult {
    pub steps: Vec<StepTrace>,
    pub attention: Vec<AttentionTrace>,
    /// Top-Kandidaten für das nächste Token (Token-ID, Logit, Wahrscheinlichkeit).
    pub top_next_tokens: Vec<(u32, f32, f32)>,
}

fn sample(v: &[f32]) -> Vec<f32> {
    v.iter().take(8).copied().collect()
}

/// Lädt einen Tensor vollständig, meldet den Lesezugriff über `on_event` und
/// liefert die dequantisierten Werte.
fn load_and_trace<F: FnMut(ReadEvent)>(
    w: &Weights,
    name: &str,
    step_label: &str,
    reason: &str,
    on_event: &mut F,
) -> Result<Vec<f32>> {
    let v = w.load(name)?;
    if let Ok((file_offset, byte_len)) = w.file_range(name) {
        on_event(ReadEvent {
            step_label: step_label.to_string(),
            tensor_name: name.to_string(),
            file_offset,
            byte_len,
            reason: reason.to_string(),
        });
    }
    Ok(v)
}

fn load_shape_and_trace<F: FnMut(ReadEvent)>(
    w: &Weights,
    name: &str,
    step_label: &str,
    reason: &str,
    on_event: &mut F,
) -> Result<(Vec<f32>, Vec<u64>)> {
    let v = w.load_with_shape(name)?;
    if let Ok((file_offset, byte_len)) = w.file_range(name) {
        on_event(ReadEvent {
            step_label: step_label.to_string(),
            tensor_name: name.to_string(),
            file_offset,
            byte_len,
            reason: reason.to_string(),
        });
    }
    Ok(v)
}

/// Führt Embedding + die ersten `n_layers` Transformer-Layer für die
/// gegebene, bereits tokenisierte Eingabe aus. Liefert den Hidden-State der
/// letzten Position nach jedem Layer als Trace sowie (falls `n_layers` alle
/// Layer abdeckt) die finalen Logits über das Vokabular. `on_event` wird für
/// JEDEN tatsächlichen Tensor-Lesezugriff aufgerufen (alle Layer, nicht nur
/// Layer 0) — Grundlage für ein vollständiges Live-Log.
pub fn run_forward<F: FnMut(ReadEvent)>(
    file: &GgufFile,
    pieces: &[TokenizedPiece],
    n_layers: u64,
    mut on_event: F,
) -> Result<ForwardResult> {
    if pieces.is_empty() {
        return Err(anyhow!("Keine Tokens übergeben"));
    }
    let token_ids: Vec<u32> = pieces.iter().map(|p| p.id).collect();
    let cfg = ArchConfig::from_gguf(file)?;
    let w = Weights::new(file);
    let n_layers = n_layers.min(cfg.n_layer).max(1);

    let embd_name = if w.has("token_embd.weight") { "token_embd.weight" } else { "tok_embeddings.weight" };
    if !w.has(embd_name) {
        return Err(anyhow!("Kein Token-Embedding-Tensor gefunden (token_embd.weight)"));
    }

    let mut steps = Vec::new();
    let mut attention = Vec::new();

    // Schritt 0: Erklärung der Tokenisierung selbst — bevor überhaupt
    // gerechnet wird, muss der Text erst in die Bausteine zerlegt werden,
    // die das Modell versteht.
    let piece_list: String = pieces.iter()
        .map(|p| format!("'{}'→{}", p.text.replace('\u{2581}', "␣"), p.id))
        .collect::<Vec<_>>()
        .join("  ");
    steps.push(StepTrace {
        label: format!("Tokenisierung ({} Tokens)", pieces.len()),
        explanation: format!(
            "Bevor irgendetwas gerechnet wird, muss der eingegebene Text in Tokens zerlegt werden — \
             Wort- oder Wortteil-Bausteine, die im Vokabular der Datei vorkommen (␣ markiert einen \
             Wortanfang, SentencePiece-Konvention). Jedes Token bekommt seine feste Nummer (ID) aus \
             dem Vokabular zugewiesen: {piece_list}. Diese Zahlen sind der EINZIGE Input, den das \
             Modell versteht — es sieht nie den Text selbst, nur diese IDs. Deshalb entscheidet schon \
             diese Zerlegung mit, wie 'vertraut' dem Modell eine Formulierung vorkommt: sehr häufige \
             Wörter sind oft ein einzelnes Token, seltene Wörter werden in mehrere Teil-Tokens zerlegt."
        ),
        sample: pieces.iter().map(|p| p.id as f32).collect(),
        shape: format!("[{} Tokens]", pieces.len()),
        hex_refs: Vec::new(),
    });

    // Embedding-Lookup für jede Position (echte Zeilen aus der Embedding-Matrix).
    let mut hidden_states: Vec<Vec<f32>> = Vec::with_capacity(token_ids.len());
    for (pos, &tok) in token_ids.iter().enumerate() {
        let emb = w.embedding_row(embd_name, tok as u64, cfg.n_embd)?;
        let (row_offset, row_len) = w.file_range_for_row(embd_name, tok as u64, cfg.n_embd)?;
        on_event(ReadEvent {
            step_label: format!("Embedding-Lookup Position {pos}"),
            tensor_name: embd_name.to_string(),
            file_offset: row_offset,
            byte_len: row_len,
            reason: format!("Zeile für Token-ID {tok}"),
        });
        if pos == token_ids.len() - 1 {
            steps.push(StepTrace {
                label: format!("Embedding-Lookup (Token-ID {tok})"),
                explanation: format!(
                    "Jede Token-ID indiziert eine Zeile in der Embedding-Matrix ({} × {}). Das ist \
                     wörtlich zu nehmen: Token-ID {tok} ist einfach eine Zeilennummer, und die Zeile \
                     an genau dieser Nummer wird aus der Datei gelesen (siehe Hex-Link unten — das \
                     sind die rohen Bytes dieser einen Zeile). Das Ergebnis ist ein Vektor mit {} \
                     Dimensionen — die 'Bedeutung' des Tokens als Zahlen, bevor der Kontext (andere \
                     Tokens) berücksichtigt wird. Diese Zahlen selbst kommen aus dem Training: sie \
                     wurden so lange angepasst, bis Tokens, die in ähnlichen Zusammenhängen auftraten, \
                     ähnliche Vektoren bekamen.",
                    cfg.n_vocab, cfg.n_embd, cfg.n_embd
                ),
                sample: sample(&emb),
                shape: format!("[{}]", cfg.n_embd),
                hex_refs: vec![HexRef { tensor_name: embd_name.to_string(), file_offset: row_offset, byte_len: row_len }],
            });
        }
        hidden_states.push(emb);
    }

    let head_dim = cfg.head_dim as usize;
    let n_head = cfg.n_head as usize;
    let n_head_kv = cfg.n_head_kv as usize;
    let group = n_head / n_head_kv.max(1);

    for layer in 0..n_layers {
        let ln = LayerNames::for_layer(layer);
        let attn_norm_w = load_and_trace(&w, &ln.attn_norm, &format!("Layer {layer}: RMSNorm-Gewicht"), "Skalierungsvektor vor der Attention", &mut on_event)?;
        let (wq, _) = load_shape_and_trace(&w, &ln.attn_q, &format!("Layer {layer}: Query-Projektion"), "Gewichtsmatrix für die Query-Vektoren", &mut on_event)?;
        let (wk, _) = load_shape_and_trace(&w, &ln.attn_k, &format!("Layer {layer}: Key-Projektion"), "Gewichtsmatrix für die Key-Vektoren", &mut on_event)?;
        let (wv, _) = load_shape_and_trace(&w, &ln.attn_v, &format!("Layer {layer}: Value-Projektion"), "Gewichtsmatrix für die Value-Vektoren", &mut on_event)?;
        let (wo, _) = load_shape_and_trace(&w, &ln.attn_output, &format!("Layer {layer}: Attention-Ausgabeprojektion"), "Mischt die Attention-Heads zurück in die Modell-Dimension", &mut on_event)?;

        let n_embd = cfg.n_embd as usize;
        let kv_dim = n_head_kv * head_dim;

        // 1) Pre-Attention RMSNorm + Q/K/V-Projektion + RoPE für alle Positionen.
        let mut qs = Vec::with_capacity(hidden_states.len());
        let mut ks = Vec::with_capacity(hidden_states.len());
        let mut vs = Vec::with_capacity(hidden_states.len());
        let mut normed_last = Vec::new();
        for (pos, h) in hidden_states.iter().enumerate() {
            let normed = ops::rms_norm(h, &attn_norm_w, cfg.rms_eps);
            if pos == hidden_states.len() - 1 {
                normed_last = normed.clone();
            }
            let mut q = ops::mat_vec(&wq, n_embd, n_head * head_dim, &normed);
            let mut k = ops::mat_vec(&wk, n_embd, kv_dim, &normed);
            let v = ops::mat_vec(&wv, n_embd, kv_dim, &normed);
            for hh in 0..n_head {
                ops::rope_inplace(&mut q[hh * head_dim..(hh + 1) * head_dim], pos, head_dim, cfg.rope_theta);
            }
            for hh in 0..n_head_kv {
                ops::rope_inplace(&mut k[hh * head_dim..(hh + 1) * head_dim], pos, head_dim, cfg.rope_theta);
            }
            qs.push(q);
            ks.push(k);
            vs.push(v);
        }

        if layer == 0 {
            let (off, len) = w.file_range(&ln.attn_norm)?;
            steps.push(StepTrace {
                label: format!("Layer {layer}: RMSNorm vor Attention"),
                explanation: "RMSNorm skaliert den Vektor auf eine konstante Größe (Root-Mean-Square), \
                    bevor er in die Attention geht — das stabilisiert das Training/die Inferenz, \
                    ähnlich einer Normalisierung von Messwerten vor einem Vergleich.".into(),
                sample: sample(&normed_last),
                shape: format!("[{}]", cfg.n_embd),
                hex_refs: vec![HexRef { tensor_name: ln.attn_norm.clone(), file_offset: off, byte_len: len }],
            });
        }

        // 2) Attention für JEDE Position (nicht nur die letzte) — jede
        // Position braucht ihren eigenen aktualisierten Hidden-State, damit
        // sie als korrekter Key/Value-Kontext für den nächsten Layer dient.
        let last_pos = hidden_states.len() - 1;
        let mut after_attn_all: Vec<Vec<f32>> = Vec::with_capacity(hidden_states.len());
        let mut after_attn_last_trace = Vec::new();
        for pos in 0..hidden_states.len() {
            let mut attn_out = vec![0.0f32; n_head * head_dim];
            for hh in 0..n_head {
                let kv_head = hh / group.max(1);
                let q_h = &qs[pos][hh * head_dim..(hh + 1) * head_dim];
                let mut scores: Vec<f32> = (0..=pos)
                    .map(|p| {
                        let k_h = &ks[p][kv_head * head_dim..(kv_head + 1) * head_dim];
                        ops::dot(q_h, k_h) / (head_dim as f32).sqrt()
                    })
                    .collect();
                ops::softmax_inplace(&mut scores);
                if layer == 0 && pos == last_pos {
                    attention.push(AttentionTrace { layer, head: hh as u64, weights: scores.clone() });
                }
                let mut out_h = vec![0.0f32; head_dim];
                for (p, &weight) in scores.iter().enumerate() {
                    let v_h = &vs[p][kv_head * head_dim..(kv_head + 1) * head_dim];
                    for d in 0..head_dim {
                        out_h[d] += weight * v_h[d];
                    }
                }
                attn_out[hh * head_dim..(hh + 1) * head_dim].copy_from_slice(&out_h);
            }
            let attn_proj = ops::mat_vec(&wo, n_head * head_dim, n_embd, &attn_out);
            let after_attn = ops::add(&hidden_states[pos], &attn_proj);
            if layer == 0 && pos == last_pos {
                after_attn_last_trace = after_attn.clone();
            }
            after_attn_all.push(after_attn);
        }

        if layer == 0 {
            let refs = [&ln.attn_q, &ln.attn_k, &ln.attn_v, &ln.attn_output]
                .iter()
                .filter_map(|n| w.file_range(n).ok().map(|(off, len)| HexRef { tensor_name: (*n).clone(), file_offset: off, byte_len: len }))
                .collect();
            steps.push(StepTrace {
                label: format!("Layer {layer}: Self-Attention ({} Heads, {} KV-Heads{})",
                    n_head, n_head_kv, if cfg.is_gqa() { ", Grouped-Query-Attention" } else { "" }),
                explanation: format!(
                    "Jeder der {n_head} Attention-Heads berechnet, wie stark die aktuelle Position \
                     auf jede vorherige Position 'achten' soll (Softmax über Query·Key-Skalarprodukte), \
                     und mischt dann die Value-Vektoren entsprechend gewichtet zusammen. \
                     {} Das Ergebnis wird über eine Ausgabe-Projektion zurück auf {} Dimensionen gebracht \
                     und per Residual-Verbindung auf den ursprünglichen Vektor addiert.",
                    if cfg.is_gqa() {
                        format!("Da nur {n_head_kv} statt {n_head} Key/Value-Heads existieren (GQA), \
                                 teilen sich jeweils {group} Query-Heads dieselben Keys/Values — spart Speicher \
                                 bei kaum spürbarem Qualitätsverlust.")
                    } else { String::new() },
                    cfg.n_embd
                ),
                sample: sample(&after_attn_last_trace),
                shape: format!("[{}]", cfg.n_embd),
                hex_refs: refs,
            });
        }

        // 3) FFN (SwiGLU) mit Residual — ebenfalls für jede Position.
        let ffn_norm_w = load_and_trace(&w, &ln.ffn_norm, &format!("Layer {layer}: RMSNorm-Gewicht (FFN)"), "Skalierungsvektor vor dem Feed-Forward-Netz", &mut on_event)?;
        let (w_gate, _) = load_shape_and_trace(&w, &ln.ffn_gate, &format!("Layer {layer}: FFN-Gate-Projektion"), "Gate-Gewichtsmatrix (SwiGLU)", &mut on_event)?;
        let (w_up, _) = load_shape_and_trace(&w, &ln.ffn_up, &format!("Layer {layer}: FFN-Up-Projektion"), "Up-Gewichtsmatrix (SwiGLU)", &mut on_event)?;
        let (w_down, _) = load_shape_and_trace(&w, &ln.ffn_down, &format!("Layer {layer}: FFN-Down-Projektion"), "Down-Gewichtsmatrix (zurück auf Modell-Dimension)", &mut on_event)?;
        let n_ff = cfg.n_ff as usize;

        let mut after_ffn_last_trace = Vec::new();
        for (pos, after_attn) in after_attn_all.iter().enumerate() {
            let normed2 = ops::rms_norm(after_attn, &ffn_norm_w, cfg.rms_eps);
            let gate = ops::mat_vec(&w_gate, n_embd, n_ff, &normed2);
            let up = ops::mat_vec(&w_up, n_embd, n_ff, &normed2);
            let activated = ops::swiglu(&gate, &up);
            let ffn_out = ops::mat_vec(&w_down, n_ff, n_embd, &activated);
            let after_ffn = ops::add(after_attn, &ffn_out);
            if layer == 0 && pos == last_pos {
                after_ffn_last_trace = after_ffn.clone();
            }
            hidden_states[pos] = after_ffn;
        }

        if layer == 0 {
            let refs = [&ln.ffn_gate, &ln.ffn_up, &ln.ffn_down]
                .iter()
                .filter_map(|n| w.file_range(n).ok().map(|(off, len)| HexRef { tensor_name: (*n).clone(), file_offset: off, byte_len: len }))
                .collect();
            steps.push(StepTrace {
                label: format!("Layer {layer}: Feed-Forward-Netz (SwiGLU, {n_ff} versteckte Dimensionen)"),
                explanation: format!(
                    "Nach der Attention durchläuft der Vektor ein kleines 2-schichtiges Netz: \
                     er wird auf {n_ff} Dimensionen hochprojiziert (per zwei getrennten Gewichtsmatrizen \
                     'gate' und 'up'), mit der SiLU-Aktivierung nichtlinear verknüpft (SwiGLU: \
                     silu(gate(x)) · up(x)) und wieder auf {} Dimensionen heruntergerechnet. \
                     Hier — nicht in der Attention — passiert der Großteil des 'Wissens' im Modell.",
                    cfg.n_embd
                ),
                sample: sample(&after_ffn_last_trace),
                shape: format!("[{}]", cfg.n_embd),
                hex_refs: refs,
            });
        }
    }

    // Output-Projektion → Logits, nur wenn ein Output-Tensor vorhanden ist.
    let mut top_next_tokens = Vec::new();
    let out_norm_name = "output_norm.weight";
    let out_name = if w.has("output.weight") { "output.weight" } else { embd_name };
    if w.has(out_norm_name) && w.has(out_name) {
        let final_norm_w = load_and_trace(&w, out_norm_name, "Output-RMSNorm", "Letzte Normalisierung vor der Vorhersage", &mut on_event)?;
        let last_hidden = hidden_states.last().unwrap();
        let normed_final = ops::rms_norm(last_hidden, &final_norm_w, cfg.rms_eps);
        let (out_w, out_shape) = load_shape_and_trace(&w, out_name, "Output-Projektion", "Projiziert auf die volle Vokabulargröße", &mut on_event)?;
        let n_embd = cfg.n_embd as usize;
        let n_vocab = (out_shape.iter().product::<u64>() / cfg.n_embd) as usize;
        let logits = ops::mat_vec(&out_w, n_embd, n_vocab, &normed_final);

        let mut probs = logits.clone();
        ops::softmax_inplace(&mut probs);
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        top_next_tokens = idx.into_iter().take(10)
            .map(|i| (i as u32, logits[i], probs[i]))
            .collect();

        let (out_off, out_len) = w.file_range(out_name)?;
        steps.push(StepTrace {
            label: "Output-Projektion → Logits".into(),
            explanation: format!(
                "Nach dem letzten Layer wird der Vektor noch einmal normiert und auf die volle \
                 Vokabulargröße ({n_vocab}) projiziert — jede der {n_vocab} mögliche nächsten Tokens \
                 bekommt einen Zahlenwert (Logit) zugewiesen, wie gut sie an dieser Stelle passt. Diese \
                 Logits laufen durch Softmax zu Wahrscheinlichkeiten, die sich zu 100% aufsummieren — \
                 das höchste ist die Vorhersage für das nächste Token. GENAU DAS ist die Antwort auf \
                 die Frage 'woher weiß das Modell, wie es antworten muss': es weiß es nicht im Sinne \
                 von Fakten-Nachschlagen, sondern diese eine Zahlenreihe (die 'Logits') ist das Ergebnis \
                 aller vorherigen Rechenschritte mit den trainierten Gewichten — und die Gewichte wurden \
                 beim Training so eingestellt, dass bei ähnlichen bisherigen Texten genau die Tokens \
                 hohe Werte bekommen, die in den Trainingsdaten an dieser Stelle üblich waren."
            ),
            sample: sample(&logits),
            shape: format!("[{n_vocab}]"),
            hex_refs: vec![HexRef { tensor_name: out_name.to_string(), file_offset: out_off, byte_len: out_len }],
        });
    }

    Ok(ForwardResult { steps, attention, top_next_tokens })
}
