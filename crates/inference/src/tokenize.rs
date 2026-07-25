//! Einfacher Greedy-Longest-Match-Tokenizer gegen das GGUF-Vokabular.
//!
//! Kein vollständiger BPE-Encoder (der eigentliche Merge-Algorithmus von
//! llama.cpp ist deutlich komplexer und nicht Ziel dieses Lern-Tools) —
//! stattdessen wird für jede Position im Text das längste Vokabeltoken
//! gesucht, das dort passt. Für kurze, einfache Demo-Prompts liefert das in
//! der Regel eine sinnvolle, nachvollziehbare Zerlegung; bei Sonderzeichen
//! oder seltenen Wortformen kann das Ergebnis vom "echten" Tokenizer
//! (SentencePiece/BPE) abweichen.

use gguf_core::tokenizer::TokenizerInfo;

#[derive(Debug, Clone, serde::Serialize)]
pub struct TokenizedPiece {
    pub id: u32,
    pub text: String,
}

pub fn greedy_tokenize(info: &TokenizerInfo, text: &str) -> Vec<TokenizedPiece> {
    // Viele llama.cpp-Vokabulare markieren Wortanfänge mit '▁' (U+2581)
    // statt einem Leerzeichen (SentencePiece-Konvention).
    let normalized = text.replace(' ', "\u{2581}");
    let chars: Vec<char> = normalized.chars().collect();
    let mut out = Vec::new();
    let mut pos = 0;

    while pos < chars.len() {
        let mut best: Option<(usize, u32)> = None;
        // Längstes passendes Token ab `pos` suchen (begrenzt auf 24 Zeichen
        // Kandidatenlänge, ausreichend für alle üblichen Subword-Tokens).
        let max_len = (chars.len() - pos).min(24);
        for len in (1..=max_len).rev() {
            let candidate: String = chars[pos..pos + len].iter().collect();
            if let Some(tok) = info.tokens.iter().find(|t| t.text == candidate) {
                best = Some((len, tok.id));
                break;
            }
        }
        match best {
            Some((len, id)) => {
                let piece: String = chars[pos..pos + len].iter().collect();
                out.push(TokenizedPiece { id, text: piece });
                pos += len;
            }
            None => {
                // Kein Treffer: einzelnes Zeichen überspringen (als
                // Unknown-Marker ohne gültige ID ausgeben, id = u32::MAX).
                out.push(TokenizedPiece { id: u32::MAX, text: chars[pos].to_string() });
                pos += 1;
            }
        }
    }
    out
}
