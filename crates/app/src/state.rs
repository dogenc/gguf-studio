//! Zentraler App-Zustand. Async-Arbeit (Ollama, Datei-Parsing) läuft auf einer
//! Tokio-Runtime; Ergebnisse kommen über mpsc-Kanäle zurück in den UI-Thread.

use crate::backup;
use gguf_core::diff::ModelDiff;
use gguf_core::{GgmlType, GgufFile};
use hexview::HexSource;
use inference::{ForwardResult, ReadEvent};
use ollama_client::{ModelSummary, OllamaClient};
use std::{path::PathBuf, sync::mpsc, sync::Arc, time::{Duration, Instant}};

pub enum AsyncMsg {
    Models(anyhow::Result<Vec<ModelSummary>>),
    FileOpened(anyhow::Result<(Arc<GgufFile>, HexSource)>),
    CompareFileOpened(anyhow::Result<Arc<GgufFile>>),
    ChatToken(String),
    ChatDone(anyhow::Result<ollama_client::BenchResult>),
    Status(String),
    ForwardDone(anyhow::Result<ForwardResult>),
    ForwardReadEvent(ReadEvent),
}

pub struct AppState {
    pub rt: tokio::runtime::Runtime,
    pub tx: mpsc::Sender<AsyncMsg>,
    rx: mpsc::Receiver<AsyncMsg>,
    pub ollama: OllamaClient,

    pub models: Vec<ModelSummary>,
    pub gguf: Option<Arc<GgufFile>>,
    pub hex: Option<HexSource>,
    pub status: String,

    // UI-State
    pub metadata_filter: String,
    pub tensor_filter: String,
    pub hex_goto: String,
    pub hex_edit_mode: bool,
    /// true, solange ein programmatischer Sprung zu `hex_goto` noch vom
    /// Hex-Panel ausgeführt werden muss (siehe `jump_to_hex`).
    pub hex_pending_jump: bool,
    pub modelfile_src: String,
    pub modelfile_errors: Vec<String>,
    pub modelfile_target: String,
    pub chat_model: String,
    pub chat_prompt: String,
    pub chat_output: String,
    pub chat_running: bool,
    /// Der zuletzt tatsächlich an Ollama gesendete Prompt — Grundlage für
    /// den "Erklären"-Button, auch wenn `chat_prompt` inzwischen weiter
    /// bearbeitet wurde.
    pub chat_last_prompt: String,

    // Tokenizer-Explorer
    pub tokenizer_filter: String,

    // Tensor-Browser: aktuell ausgewählter Tensor für Vorschau/Statistik
    pub selected_tensor: Option<usize>,

    // Quantisierungs-Analyzer: aktuell für das Bit-Layout-Diagramm gewählter Typ
    pub quant_selected: Option<GgmlType>,

    // Modellvergleich
    pub compare_gguf: Option<Arc<GgufFile>>,
    pub compare_diff: Option<ModelDiff>,

    // Modelfile-Versionsverwaltung
    pub modelfile_versions: Vec<PathBuf>,

    // Forward-Pass-Explorer
    pub forward_prompt: String,
    pub forward_n_layers: u64,
    pub forward_running: bool,
    pub forward_result: Option<ForwardResult>,
    pub forward_error: Option<String>,

    // Live-Ausführungslog: jeder tatsächliche Datei-Lesezugriff während des
    // letzten Forward-Pass-Laufs, in echter Reihenfolge.
    pub forward_log: Vec<ReadEvent>,
    pub forward_log_bytes: u64,
    pub forward_started_at: Option<Instant>,
    pub forward_duration: Option<Duration>,

    /// Von einem Panel gesetzt, wenn ein anderes Panel (z. B. der Hex-
    /// Viewer) fokussiert und dorthin gesprungen werden soll. Wird von
    /// `main.rs` einmal pro Frame konsumiert.
    pub jump_to_panel: Option<&'static str>,
}

impl AppState {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let rt = tokio::runtime::Runtime::new().expect("Tokio-Runtime");
        let mut s = Self {
            rt, tx, rx,
            ollama: OllamaClient::local(),
            models: Vec::new(),
            gguf: None,
            hex: None,
            status: "Bereit".into(),
            metadata_filter: String::new(),
            tensor_filter: String::new(),
            hex_goto: String::new(),
            hex_edit_mode: false,
            hex_pending_jump: false,
            modelfile_src: "FROM llama3\nPARAMETER temperature 0.7\n".into(),
            modelfile_errors: Vec::new(),
            modelfile_target: "mein-modell".into(),
            chat_model: String::new(),
            chat_prompt: String::new(),
            chat_output: String::new(),
            chat_running: false,
            chat_last_prompt: String::new(),
            tokenizer_filter: String::new(),
            selected_tensor: None,
            quant_selected: None,
            compare_gguf: None,
            compare_diff: None,
            modelfile_versions: Vec::new(),
            forward_prompt: "Die Katze sitzt".into(),
            forward_n_layers: 2,
            forward_running: false,
            forward_result: None,
            forward_error: None,
            forward_log: Vec::new(),
            forward_log_bytes: 0,
            forward_started_at: None,
            forward_duration: None,
            jump_to_panel: None,
        };
        s.refresh_models();
        s
    }

    pub fn status_line(&self) -> String {
        match &self.gguf {
            Some(g) => format!(
                "{} · {} Tensoren · {:.2} GB · {}",
                g.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
                g.tensors.len(),
                g.file_size as f64 / 1e9,
                self.status
            ),
            None => self.status.clone(),
        }
    }

    pub fn refresh_models(&mut self) {
        let tx = self.tx.clone();
        let client = self.ollama.clone();
        self.status = "Lade Ollama-Modelle…".into();
        self.rt.spawn(async move {
            let _ = tx.send(AsyncMsg::Models(client.list_models().await));
        });
    }

    pub fn open_file_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("GGUF", &["gguf"])
            .add_filter("Alle Dateien", &["*"])
            .pick_file()
        {
            self.open_path(path);
        }
    }

    pub fn open_path(&mut self, path: PathBuf) {
        let tx = self.tx.clone();
        self.status = format!("Parse {}…", path.display());
        self.rt.spawn(async move {
            let res = tokio::task::spawn_blocking(move || {
                let g = GgufFile::open(&path)?;
                let h = HexSource::open(&path)?;
                Ok::<_, anyhow::Error>((Arc::new(g), h))
            })
            .await
            .unwrap_or_else(|e| Err(e.into()));
            let _ = tx.send(AsyncMsg::FileOpened(res));
        });
    }

    /// Löst einen Ollama-Modellnamen zum lokalen GGUF-Blob auf (ohne
    /// `.gguf`-Endung, gespeichert unter `~/.ollama/models/blobs/…`) und
    /// öffnet ihn wie eine normal ausgewählte Datei.
    pub fn open_ollama_model(&mut self, name: &str) {
        match ollama_client::store::resolve_model_path(name) {
            Ok(path) => self.open_path(path),
            Err(e) => self.status = format!("Konnte '{name}' nicht auflösen: {e}"),
        }
    }

    pub fn open_compare_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("GGUF", &["gguf"])
            .add_filter("Alle Dateien", &["*"])
            .pick_file()
        {
            let tx = self.tx.clone();
            self.status = format!("Lade Vergleichsdatei {}…", path.display());
            self.rt.spawn(async move {
                let res = tokio::task::spawn_blocking(move || {
                    GgufFile::open(&path).map(Arc::new).map_err(anyhow::Error::from)
                })
                .await
                .unwrap_or_else(|e| Err(e.into()));
                let _ = tx.send(AsyncMsg::CompareFileOpened(res));
            });
        }
    }

    pub fn recompute_diff(&mut self) {
        if let (Some(a), Some(b)) = (&self.gguf, &self.compare_gguf) {
            self.compare_diff = Some(gguf_core::diff::compare(a, b));
        }
    }

    pub fn refresh_modelfile_versions(&mut self) {
        self.modelfile_versions = backup::list_versions(&self.modelfile_target);
    }

    pub fn restore_modelfile_version(&mut self, path: &std::path::Path) {
        if let Ok(content) = backup::read_version(&path.to_path_buf()) {
            self.modelfile_src = content;
            self.status = format!("Version wiederhergestellt: {}", path.display());
        }
    }

    /// Übernimmt den zuletzt an Ollama gesendeten Chat-Prompt in den
    /// Forward-Pass-Explorer, startet dort eine echte Berechnung mit den
    /// Gewichten der aktuell geöffneten GGUF-Datei und springt zum Panel —
    /// so lässt sich nachvollziehen, welche Tensor-Zeilen ein Modell für
    /// genau diesen Prompt benutzt. Wichtig: das ist eine Annäherung mit
    /// denselben Gewichten, aber nur den ersten Layern — NICHT dieselbe
    /// Berechnung, die Ollama intern für die tatsächliche Chat-Antwort
    /// durchgeführt hat (Ollama läuft als eigener Prozess und liefert keine
    /// internen Logits/Gewichts-Zugriffe über die REST-API zurück).
    pub fn explain_last_chat(&mut self) {
        if self.chat_last_prompt.is_empty() || self.gguf.is_none() {
            return;
        }
        self.forward_prompt = self.chat_last_prompt.clone();
        self.run_forward_pass();
        self.jump_to_panel = Some("forward");
    }

    /// Springt im Hex-Viewer zu einem absoluten Datei-Offset und öffnet/
    /// fokussiert das Hex-Panel (ausgeführt von `main.rs` im nächsten Frame).
    pub fn jump_to_hex(&mut self, offset: u64) {
        self.hex_goto = format!("{offset:x}");
        self.hex_pending_jump = true;
        self.jump_to_panel = Some("hex");
    }

    pub fn run_forward_pass(&mut self) {
        let Some(g) = self.gguf.clone() else { return };
        if self.forward_running || self.forward_prompt.trim().is_empty() {
            return;
        }
        self.forward_running = true;
        self.forward_result = None;
        self.forward_error = None;
        self.forward_log.clear();
        self.forward_log_bytes = 0;
        self.forward_started_at = Some(Instant::now());
        self.forward_duration = None;
        let tx = self.tx.clone();
        let tx_events = self.tx.clone();
        let prompt = self.forward_prompt.clone();
        let n_layers = self.forward_n_layers;
        self.rt.spawn(async move {
            let res = tokio::task::spawn_blocking(move || -> anyhow::Result<ForwardResult> {
                let info = gguf_core::tokenizer::extract(&g)
                    .ok_or_else(|| anyhow::anyhow!("Kein Tokenizer in dieser Datei gefunden"))?;
                let pieces: Vec<_> = inference::tokenize::greedy_tokenize(&info, &prompt)
                    .into_iter()
                    .filter(|p| p.id != u32::MAX)
                    .collect();
                if pieces.is_empty() {
                    anyhow::bail!("Prompt konnte nicht tokenisiert werden");
                }
                inference::run_forward(&g, &pieces, n_layers, move |ev| {
                    let _ = tx_events.send(AsyncMsg::ForwardReadEvent(ev));
                })
            })
            .await
            .unwrap_or_else(|e| Err(e.into()));
            let _ = tx.send(AsyncMsg::ForwardDone(res));
        });
    }

    /// Schreibt das aktuelle Live-Ausführungslog als lesbare Textdatei.
    pub fn export_forward_log(&mut self) {
        if self.forward_log.is_empty() {
            return;
        }
        let Some(path) = rfd::FileDialog::new()
            .set_file_name("forward-pass-log.txt")
            .add_filter("Text", &["txt"])
            .save_file()
        else {
            return;
        };
        let mut out = String::new();
        out.push_str("GGUF Studio — Forward-Pass Live-Ausführungslog\n");
        out.push_str(&format!("Prompt: {}\n", self.forward_prompt));
        out.push_str(&format!("Layer: {}\n\n", self.forward_n_layers));
        for (i, ev) in self.forward_log.iter().enumerate() {
            out.push_str(&format!(
                "{:>4}. [{}] {} — Offset {:#x}, {} Bytes — {}\n",
                i + 1, ev.step_label, ev.tensor_name, ev.file_offset, ev.byte_len, ev.reason
            ));
        }
        out.push_str(&format!(
            "\nGesamt: {} Lesezugriffe, {} Bytes",
            self.forward_log.len(), self.forward_log_bytes
        ));
        if let Some(d) = self.forward_duration {
            out.push_str(&format!(", {:.1} ms", d.as_secs_f64() * 1000.0));
        }
        match std::fs::write(&path, out) {
            Ok(()) => self.status = format!("Log exportiert: {}", path.display()),
            Err(e) => self.status = format!("Export fehlgeschlagen: {e}"),
        }
    }

    pub fn send_chat(&mut self) {
        if self.chat_running || self.chat_model.is_empty() || self.chat_prompt.is_empty() {
            return;
        }
        self.chat_running = true;
        self.chat_output.clear();
        self.chat_last_prompt = self.chat_prompt.clone();
        let (tx, client) = (self.tx.clone(), self.ollama.clone());
        let (model, prompt) = (self.chat_model.clone(), self.chat_prompt.clone());
        self.rt.spawn(async move {
            let tx2 = tx.clone();
            let res = client
                .generate_stream(&model, &prompt, None, move |t| {
                    let _ = tx2.send(AsyncMsg::ChatToken(t.to_string()));
                })
                .await;
            let _ = tx.send(AsyncMsg::ChatDone(res));
        });
    }

    pub fn poll_async(&mut self, ctx: &eframe::egui::Context) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                AsyncMsg::Models(Ok(m)) => {
                    self.status = format!("{} Ollama-Modelle gefunden", m.len());
                    if self.chat_model.is_empty() {
                        if let Some(first) = m.first() { self.chat_model = first.name.clone(); }
                    }
                    self.models = m;
                }
                AsyncMsg::Models(Err(e)) => self.status = format!("Ollama: {e}"),
                AsyncMsg::FileOpened(Ok((g, h))) => {
                    self.status = "Datei geladen".into();
                    self.gguf = Some(g);
                    self.hex = Some(h);
                    self.selected_tensor = None;
                    self.recompute_diff();
                }
                AsyncMsg::FileOpened(Err(e)) => self.status = format!("Fehler: {e}"),
                AsyncMsg::CompareFileOpened(Ok(g)) => {
                    self.status = "Vergleichsdatei geladen".into();
                    self.compare_gguf = Some(g);
                    self.recompute_diff();
                }
                AsyncMsg::CompareFileOpened(Err(e)) => self.status = format!("Fehler: {e}"),
                AsyncMsg::ChatToken(t) => self.chat_output.push_str(&t),
                AsyncMsg::ChatDone(res) => {
                    self.chat_running = false;
                    if let Ok(b) = res {
                        self.status = format!("{:.1} tok/s", b.tokens_per_sec);
                    }
                }
                AsyncMsg::Status(s) => self.status = s,
                AsyncMsg::ForwardDone(res) => {
                    self.forward_running = false;
                    self.forward_duration = self.forward_started_at.map(|t| t.elapsed());
                    match res {
                        Ok(r) => self.forward_result = Some(r),
                        Err(e) => self.forward_error = Some(e.to_string()),
                    }
                }
                AsyncMsg::ForwardReadEvent(ev) => {
                    self.forward_log_bytes += ev.byte_len;
                    self.forward_log.push(ev);
                }
            }
            ctx.request_repaint();
        }
        if self.chat_running || self.forward_running {
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        }
    }
}
