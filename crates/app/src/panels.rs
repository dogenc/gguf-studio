//! Alle Dock-Panels. Sämtliche großen Listen nutzen `show_rows`
//! (virtuelles Scrolling) — es werden nur sichtbare Zeilen gerendert.

use crate::backup;
use crate::plugin::PluginRegistry;
use crate::state::AppState;
use eframe::egui::{self, RichText};
use gguf_core::diff::DiffStatus;
use gguf_core::{dequant, quant, tokenizer};
use hexview::{format_row, BYTES_PER_ROW};

/// Panel-Kennung im Dock — der stabile Plugin-`id()`-String.
pub type PanelId = &'static str;

pub struct PanelViewer<'a> {
    pub state: &'a mut AppState,
    pub registry: &'a PluginRegistry,
}

impl egui_dock::TabViewer for PanelViewer<'_> {
    type Tab = PanelId;

    fn title(&mut self, tab: &mut PanelId) -> egui::WidgetText {
        self.registry.by_id(tab).map(|p| p.title()).unwrap_or(*tab).into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut PanelId) {
        if let Some(plugin) = self.registry.by_id(tab) {
            plugin.ui(ui, self.state);
        } else {
            ui.label(format!("Unbekanntes Panel-Plugin: {tab}"));
        }
    }
}

pub(crate) fn human_bytes(b: u64) -> String {
    const U: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 { v /= 1024.0; i += 1; }
    format!("{v:.2} {}", U[i])
}

pub(crate) fn library(ui: &mut egui::Ui, s: &mut AppState) {
    ui.horizontal(|ui| {
        ui.heading("Ollama-Modelle");
        if ui.button("⟳").clicked() { s.refresh_models(); }
    });
    ui.separator();
    egui::ScrollArea::vertical().show(ui, |ui| {
        for m in s.models.clone() {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&m.name).strong());
                    ui.label(human_bytes(m.size));
                });
                ui.horizontal(|ui| {
                    if !m.details.family.is_empty() { ui.label(format!("Familie: {}", m.details.family)); }
                    if !m.details.parameter_size.is_empty() { ui.label(format!("Params: {}", m.details.parameter_size)); }
                    if !m.details.quantization_level.is_empty() { ui.label(format!("Quant: {}", m.details.quantization_level)); }
                });
                ui.horizontal(|ui| {
                    if ui.small_button("Im Chat testen").clicked() { s.chat_model = m.name.clone(); }
                    if ui.small_button("Im GGUF Explorer öffnen").clicked() {
                        s.open_ollama_model(&m.name);
                    }
                });
            });
        }
        if s.models.is_empty() {
            ui.label("Keine Modelle — läuft Ollama? (http://127.0.0.1:11434)");
        }
    });
}

pub(crate) fn explorer(ui: &mut egui::Ui, s: &mut AppState) {
    let Some(g) = s.gguf.clone() else {
        ui.label("Keine GGUF-Datei geöffnet (Strg+O).");
        return;
    };
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("GGUF v{}", g.version)).strong());
        ui.label(format!("Architektur: {}", g.architecture().unwrap_or("?")));
        ui.label(format!("Parameter: {:.2} B", g.param_count() as f64 / 1e9));
        ui.label(format!("Alignment: {}", g.alignment));
        ui.label(format!("Datenoffset: {:#x}", g.data_offset));
    });
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("Filter:");
        ui.text_edit_singleline(&mut s.metadata_filter);
    });
    let f = s.metadata_filter.to_lowercase();
    let entries: Vec<(&String, &gguf_core::value::GgufValue)> = g
        .metadata
        .iter()
        .filter(|(k, _)| f.is_empty() || k.to_lowercase().contains(&f))
        .collect();
    let row_h = ui.text_style_height(&egui::TextStyle::Body);
    egui::ScrollArea::vertical().show_rows(ui, row_h, entries.len(), |ui, range| {
        egui::Grid::new("kv").striped(true).min_col_width(180.0).show(ui, |ui| {
            for (k, v) in &entries[range] {
                ui.label(RichText::new(*k).monospace());
                ui.label(v.display_short(120));
                ui.end_row();
            }
        });
    });
}

pub(crate) fn tensors(ui: &mut egui::Ui, s: &mut AppState) {
    let Some(g) = s.gguf.clone() else {
        ui.label("Keine GGUF-Datei geöffnet.");
        return;
    };
    ui.horizontal(|ui| {
        ui.label(format!("{} Tensoren", g.tensors.len()));
        ui.label("Filter:");
        ui.text_edit_singleline(&mut s.tensor_filter);
    });
    let f = s.tensor_filter.to_lowercase();
    let idx: Vec<usize> = g.tensors.iter().enumerate()
        .filter(|(_, t)| f.is_empty() || t.name.to_lowercase().contains(&f))
        .map(|(i, _)| i)
        .collect();
    let row_h = ui.text_style_height(&egui::TextStyle::Monospace);
    egui::ScrollArea::both().max_height(ui.available_height() * 0.55).show_rows(ui, row_h, idx.len(), |ui, range| {
        egui::Grid::new("tensors").striped(true).show(ui, |ui| {
            for &i in &idx[range] {
                let t = &g.tensors[i];
                let selected = s.selected_tensor == Some(i);
                if ui.selectable_label(selected, &t.name).clicked() {
                    s.selected_tensor = Some(i);
                }
                ui.monospace(format!("{:?}", t.dtype));
                ui.monospace(format!("{:?}", t.shape));
                ui.monospace(human_bytes(t.size_bytes));
                ui.monospace(format!("{:#x}", t.offset));
                if ui.small_button("Hex").clicked() {
                    s.hex_goto = format!("{:x}", g.data_offset + t.offset);
                }
                ui.end_row();
            }
        });
    });

    ui.separator();
    ui.heading("Wertevorschau & Statistik");
    let Some(sel) = s.selected_tensor.filter(|&i| i < g.tensors.len()) else {
        ui.label("Tensor in der Tabelle auswählen, um Werte und Statistik zu sehen.");
        return;
    };
    let t = &g.tensors[sel];
    const PREVIEW_N: usize = 4096;
    let n_preview = PREVIEW_N.min(t.size_bytes as usize).max(1);
    match g.tensor_bytes(t, 0, n_preview) {
        Ok(raw) => {
            let values = dequant::dequantize_preview(t.dtype, raw, 2048);
            if values.is_empty() {
                ui.label("Keine Vorschau für diesen Datentyp verfügbar.");
                return;
            }
            let stats = dequant::compute_stats(&values);
            ui.horizontal(|ui| {
                ui.label(format!("n={}", stats.count));
                ui.label(format!("min={:.4}", stats.min));
                ui.label(format!("max={:.4}", stats.max));
                ui.label(format!("mean={:.4}", stats.mean));
                ui.label(format!("std={:.4}", stats.std_dev));
            });
            let max_bin = *stats.histogram.iter().max().unwrap_or(&1).max(&1);
            ui.horizontal(|ui| {
                for &b in &stats.histogram {
                    let h = 60.0 * (b as f32 / max_bin as f32).max(0.02);
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(6.0, 60.0), egui::Sense::hover());
                    let bar = egui::Rect::from_min_size(
                        egui::pos2(rect.min.x, rect.max.y - h),
                        egui::vec2(6.0, h),
                    );
                    ui.painter().rect_filled(bar, 0.0, ui.visuals().selection.bg_fill);
                }
            });
            ui.label(RichText::new(format!(
                "Erste {} Werte (von {} Elementen gesamt):",
                values.len().min(64),
                t.n_elements()
            )).weak());
            egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                let sample: String = values.iter().take(64)
                    .map(|v| format!("{v:.4}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                ui.monospace(sample);
            });
        }
        Err(e) => {
            ui.colored_label(egui::Color32::LIGHT_RED, format!("Lesefehler: {e}"));
        }
    }
}

/// Feste Farbpalette für Quant-Typen — konsistent zwischen Kreisdiagramm,
/// Legende und Bit-Layout-Balken.
const QUANT_PALETTE: &[egui::Color32] = &[
    egui::Color32::from_rgb(97, 175, 239),
    egui::Color32::from_rgb(198, 120, 221),
    egui::Color32::from_rgb(224, 108, 117),
    egui::Color32::from_rgb(152, 195, 121),
    egui::Color32::from_rgb(229, 192, 123),
    egui::Color32::from_rgb(86, 182, 194),
    egui::Color32::from_rgb(209, 154, 102),
    egui::Color32::from_rgb(171, 178, 191),
];

fn draw_pie_chart(ui: &mut egui::Ui, slices: &[(f64, egui::Color32)], diameter: f32) {
    let (rect, _resp) = ui.allocate_exact_size(egui::vec2(diameter, diameter), egui::Sense::hover());
    let painter = ui.painter();
    let center = rect.center();
    let radius = diameter * 0.46;
    let total: f64 = slices.iter().map(|s| s.0).sum::<f64>().max(1e-9);
    let mut start = -std::f64::consts::FRAC_PI_2;
    for &(value, color) in slices {
        let sweep = (value / total) * std::f64::consts::TAU;
        let steps = ((sweep / 0.04).ceil() as usize).max(1);
        let mut points = vec![center];
        for i in 0..=steps {
            let a = start + sweep * (i as f64 / steps as f64);
            points.push(center + egui::vec2((a.cos() as f32) * radius, (a.sin() as f32) * radius));
        }
        painter.add(egui::Shape::convex_polygon(points, color, egui::Stroke::NONE));
        start += sweep;
    }
    painter.circle_stroke(center, radius, egui::Stroke::new(1.5_f32, ui.visuals().window_fill()));
}

fn draw_bit_layout(ui: &mut egui::Ui, fields: &[gguf_core::quant::BitField]) {
    let total: f32 = fields.iter().map(|f| f.bytes).sum::<f32>().max(0.001);
    let width = ui.available_width().min(600.0);
    let height = 40.0;
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter();
    let mut x = rect.min.x;
    for (i, f) in fields.iter().enumerate() {
        let w = (f.bytes / total) * width;
        let seg = egui::Rect::from_min_size(egui::pos2(x, rect.min.y), egui::vec2(w.max(1.0), height));
        let color = QUANT_PALETTE[i % QUANT_PALETTE.len()];
        painter.rect_filled(seg, 3.0, color);
        painter.rect_stroke(seg, 3.0, egui::Stroke::new(1.0_f32, ui.visuals().window_fill()));
        if w > 34.0 {
            painter.text(seg.center(), egui::Align2::CENTER_CENTER, format!("{:.0}B", f.bytes), egui::FontId::monospace(10.5), egui::Color32::BLACK);
        }
        x += w;
    }
    ui.add_space(6.0);
    for (i, f) in fields.iter().enumerate() {
        let color = QUANT_PALETTE[i % QUANT_PALETTE.len()];
        ui.horizontal(|ui| {
            let (r, _) = ui.allocate_exact_size(egui::vec2(11.0, 11.0), egui::Sense::hover());
            ui.painter().rect_filled(r, 2.0, color);
            ui.label(RichText::new(format!("{} — {:.0} Byte", f.name, f.bytes)).strong().small());
        });
        ui.label(RichText::new(&f.description).weak().small());
        ui.add_space(3.0);
    }
}

pub(crate) fn quantization(ui: &mut egui::Ui, s: &mut AppState) {
    let Some(g) = s.gguf.clone() else {
        ui.label("Keine GGUF-Datei geöffnet.");
        return;
    };
    let stats = quant::analyze(&g);
    let total: u64 = stats.iter().map(|q| q.total_bytes).sum::<u64>().max(1);

    ui.heading("Quantisierungs-Analyzer");
    ui.label(RichText::new(format!(
        "Kompression vs. F32: {:.2}x · Gesamtgröße: {}",
        quant::compression_vs_f32(&stats), human_bytes(total)
    )).strong());
    ui.separator();

    ui.horizontal(|ui| {
        draw_pie_chart(ui, &stats.iter().enumerate().map(|(i, q)| (q.total_bytes as f64, QUANT_PALETTE[i % QUANT_PALETTE.len()])).collect::<Vec<_>>(), 170.0);
        ui.add_space(12.0);
        egui::Grid::new("quant_legend").striped(true).show(ui, |ui| {
            ui.label(RichText::new("Typ").strong());
            ui.label(RichText::new("Anteil").strong());
            ui.label(RichText::new("Größe").strong());
            ui.label(RichText::new("bpw").strong());
            ui.end_row();
            for (i, q) in stats.iter().enumerate() {
                let color = QUANT_PALETTE[i % QUANT_PALETTE.len()];
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), egui::Sense::hover());
                    ui.painter().rect_filled(r, 2.0, color);
                    ui.label(RichText::new(format!("{:?}", q.dtype)).strong());
                });
                let frac = q.total_bytes as f64 / total as f64;
                ui.label(format!("{:.1} %", frac * 100.0));
                ui.label(human_bytes(q.total_bytes));
                ui.label(format!("{:.2}", q.bits_per_weight));
                ui.end_row();
            }
        });
    });

    ui.separator();
    ui.heading("Genauigkeit & Geschwindigkeit je Typ");
    egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
        for q in &stats {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{:?}", q.dtype)).strong().monospace());
                ui.label("Genauigkeit (rel.):");
                ui.add(egui::ProgressBar::new(quant::relative_quality(q.dtype)).desired_width(140.0));
                ui.label("Geschwindigkeit (rel.):");
                ui.add(egui::ProgressBar::new(quant::relative_speed(q.dtype)).desired_width(140.0));
            });
            ui.label(RichText::new(quant::describe(q.dtype)).weak().small());
            ui.add_space(4.0);
        }
    });

    ui.separator();
    ui.heading("Bit-Layout eines einzelnen Quantisierungs-Blocks");
    ui.label(RichText::new(
        "Zeigt, wie die Bytes EINES Blocks (z. B. 32 oder 256 Gewichte) aufgeteilt sind: \
         wie viele Bytes für Skalierungsfaktoren draufgehen und wie viele für die eigentlichen \
         gepackten Gewichte."
    ).weak().small());
    ui.horizontal(|ui| {
        ui.label("Typ:");
        egui::ComboBox::from_id_salt("quant_bitlayout_select")
            .selected_text(s.quant_selected.map(|d| format!("{d:?}")).unwrap_or_else(|| "Wählen…".into()))
            .show_ui(ui, |ui| {
                for q in &stats {
                    ui.selectable_value(&mut s.quant_selected, Some(q.dtype), format!("{:?}", q.dtype));
                }
            });
    });
    if s.quant_selected.is_none() {
        if let Some(first) = stats.first() {
            s.quant_selected = Some(first.dtype);
        }
    }
    if let Some(dtype) = s.quant_selected {
        let fields = quant::bit_layout(dtype);
        draw_bit_layout(ui, &fields);
    }
}

pub(crate) fn hex(ui: &mut egui::Ui, s: &mut AppState) {
    if s.hex.is_none() {
        ui.label("Keine Datei geöffnet.");
        return;
    }
    let mut new_status: Option<String> = None;
    let mut edit_mode = s.hex_edit_mode;

    ui.horizontal(|ui| {
        ui.label("Offset (hex):");
        let resp = ui.text_edit_singleline(&mut s.hex_goto);
        let go = ui.button("Springen").clicked()
            || (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
            || s.hex_pending_jump;
        s.hex_pending_jump = false;
        if go {
            if let Ok(off) = u64::from_str_radix(s.hex_goto.trim_start_matches("0x"), 16) {
                let row = off / BYTES_PER_ROW as u64;
                ui.ctx().memory_mut(|m| m.data.insert_temp(egui::Id::new("hex_scroll_to"), row));
            }
        }
        let h = s.hex.as_mut().unwrap();
        ui.label(format!("Größe: {}", human_bytes(h.len)));
        ui.separator();
        ui.checkbox(&mut edit_mode, "Bearbeiten");
        if h.has_pending_changes() {
            ui.colored_label(
                egui::Color32::YELLOW,
                format!("{} ungespeicherte Änderung(en)", h.pending_change_count()),
            );
            if ui.button("Speichern (mit Backup)").clicked() {
                new_status = Some(match h.commit(&backup::binary_backup_dir()) {
                    Ok(Some(bak)) => format!("Gespeichert. Backup: {}", bak.display()),
                    Ok(None) => return,
                    Err(e) => format!("Speichern fehlgeschlagen: {e}"),
                });
            }
            if ui.button("Verwerfen").clicked() {
                h.discard_changes();
            }
        }
    });
    s.hex_edit_mode = edit_mode;
    if let Some(msg) = new_status {
        s.status = msg;
    }
    ui.separator();

    let edit_mode = s.hex_edit_mode;
    let h = s.hex.as_mut().unwrap();
    let total = h.total_rows() as usize;
    let row_h = ui.text_style_height(&egui::TextStyle::Monospace);
    let mut area = egui::ScrollArea::vertical();
    if let Some(row) = ui.ctx().memory_mut(|m| m.data.remove_temp::<u64>(egui::Id::new("hex_scroll_to"))) {
        area = area.vertical_scroll_offset(row as f32 * row_h);
    }
    area.show_rows(ui, row_h, total, |ui, range| {
        let rows = h.rows(range.start as u64, range.len());
        for r in &rows {
            let (off, _, ascii) = format_row(r);
            ui.horizontal(|ui| {
                ui.monospace(RichText::new(off).weak());
                if edit_mode {
                    for (i, b) in r.bytes.iter().enumerate() {
                        let mut text = format!("{b:02X}");
                        let id = egui::Id::new(("hex_byte", r.offset + i as u64));
                        let resp = ui.add(
                            egui::TextEdit::singleline(&mut text)
                                .id(id)
                                .desired_width(18.0)
                                .font(egui::TextStyle::Monospace),
                        );
                        if resp.changed() {
                            if let Ok(v) = u8::from_str_radix(text.trim(), 16) {
                                h.set_byte(r.offset + i as u64, v);
                            }
                        }
                    }
                } else {
                    let hexs: String = r.bytes.iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" ");
                    ui.monospace(format!("{hexs:<47}"));
                }
                ui.monospace(ascii);
            });
        }
    });
}

const MODELFILE_DIRECTIVES: &[&str] = &["FROM", "PARAMETER", "TEMPLATE", "SYSTEM", "ADAPTER", "LICENSE", "MESSAGE"];
const MODELFILE_PARAMETERS: &[&str] = &[
    "temperature", "top_p", "top_k", "num_ctx", "num_predict", "repeat_penalty",
    "repeat_last_n", "seed", "stop", "mirostat", "mirostat_eta", "mirostat_tau",
];

fn directive_help(directive: &str) -> &'static str {
    match directive.to_ascii_uppercase().as_str() {
        "FROM" => "Legt fest, welches Basis-GGUF-Modell verwendet wird (Pflichtfeld, muss die erste \
            Zeile sein). Beispiel: 'FROM llama3' nutzt ein bereits über Ollama gezogenes Modell, \
            'FROM ./pfad/zu/datei.gguf' nutzt eine lokale Datei.",
        "PARAMETER" => "Setzt einen Inferenz-Parameter, der steuert, WIE das Modell antwortet \
            (nicht was es weiß). Schau dir die Erklärung des jeweiligen Parameters unten an.",
        "TEMPLATE" => "Das Prompt-Template — bestimmt, in welchem exakten Textformat System-Prompt, \
            Nutzer-Nachricht und Modell-Antwort zusammengebaut werden, bevor sie an das Modell gehen \
            (z. B. mit speziellen Markern wie <|user|>). Falsches Template = das Modell 'versteht' \
            die Struktur der Konversation nicht richtig.",
        "SYSTEM" => "Der System-Prompt — eine feste Anweisung, die vor jeder Konversation unsichtbar \
            mitgeschickt wird, z. B. 'Du bist ein hilfreicher Assistent, der auf Deutsch antwortet.' \
            Beeinflusst Tonfall und Verhalten in JEDEM Gespräch mit diesem Modell.",
        "ADAPTER" => "Bindet einen LoRA-Adapter ein — eine kleine, nachtrainierte Zusatzdatei, die das \
            Basismodell für einen speziellen Zweck anpasst, ohne das ganze Modell neu zu trainieren.",
        "LICENSE" => "Reine Metadaten-Angabe zur Lizenz des Modells, hat keinen Einfluss auf das \
            Verhalten.",
        "MESSAGE" => "Fügt Beispiel-Nachrichten in die 'Erinnerung' des Modells ein (Few-Shot-Beispiele), \
            um das gewünschte Antwortverhalten vorzuzeigen, bevor der Nutzer überhaupt etwas schreibt.",
        _ => "Unbekannte Direktive.",
    }
}

fn parameter_help(param: &str) -> &'static str {
    match param {
        "temperature" => "Steuert die Zufälligkeit der Wortwahl. Niedrig (z. B. 0.1) = das Modell \
            wählt fast immer das wahrscheinlichste nächste Wort → sehr vorhersehbare, 'trockene' \
            Antworten. Hoch (z. B. 1.2) = auch unwahrschein­lichere Wörter kommen öfter zum Zug → \
            kreativere, aber auch fehleranfälligere Antworten. Standard meist 0.7-0.8.",
        "top_p" => "'Nucleus Sampling': das Modell wählt das nächste Wort nur aus der kleinsten \
            Gruppe von Kandidaten, deren Wahrscheinlichkeiten zusammen mindestens top_p ergeben \
            (z. B. 0.9 = die wahrscheinlichsten Wörter, bis 90% der Gesamtwahrscheinlichkeit erreicht \
            sind). Schneidet sehr unwahrscheinliche Ausreißer ab.",
        "top_k" => "Begrenzt die Wortwahl auf die k wahrscheinlichsten Kandidaten (z. B. top_k 40 = \
            nur aus den 40 wahrscheinlichsten nächsten Wörtern wird gezogen). Ein einfacherer, \
            harter Filter im Vergleich zu top_p.",
        "num_ctx" => "Die Kontextfenster-Größe in Tokens — wie viel Text (Konversation + Prompt) das \
            Modell gleichzeitig 'im Blick' hat. Größer = mehr Gedächtnis für lange Gespräche, aber \
            mehr RAM/VRAM-Bedarf und langsamer.",
        "num_predict" => "Maximale Anzahl an Tokens, die das Modell in einer Antwort erzeugen darf, \
            bevor es zwangsweise abbricht (-1 = unbegrenzt).",
        "repeat_penalty" => "Bestraft Wörter, die das Modell kürzlich schon verwendet hat, damit es \
            sich nicht in Wiederholungsschleifen verfängt (Werte > 1.0 verringern Wiederholungen).",
        "repeat_last_n" => "Wie viele der letzten Tokens bei der Wiederholungs-Bestrafung \
            berücksichtigt werden.",
        "seed" => "Startwert für den Zufallsgenerator bei der Wortwahl. Gleicher Seed + gleicher \
            Prompt + gleiche Parameter = reproduzierbar dieselbe Antwort (nützlich zum Testen).",
        "stop" => "Ein Text, bei dessen Auftreten das Modell die Antwort sofort abbricht (z. B. um \
            zu verhindern, dass es eine fiktive nächste Nutzer-Nachricht mit-generiert).",
        "mirostat" => "Alternative Sampling-Strategie, die die 'Überraschung' der Ausgabe aktiv auf \
            einem Zielwert hält, statt fest top_p/top_k zu nutzen (0 = aus, 1/2 = Varianten aktiv).",
        "mirostat_eta" => "Lernrate des Mirostat-Algorithmus — wie schnell er nachregelt.",
        "mirostat_tau" => "Zielwert für die 'Überraschung' (Perplexität) bei Mirostat — niedriger \
            Wert = fokussiertere, höherer Wert = vielfältigere Ausgaben.",
        _ => "Unbekannter Parameter.",
    }
}

const MODELFILE_TEMPLATE_MINIMAL: &str = "FROM llama3\n\
# System-Prompt: feste Anweisung, die vor jedem Gespräch mitgeschickt wird\n\
SYSTEM \"Du bist ein hilfreicher Assistent, der kurz und präzise auf Deutsch antwortet.\"\n\n\
# Inferenz-Parameter: steuern WIE geantwortet wird\n\
PARAMETER temperature 0.7\n\
PARAMETER top_p 0.9\n\
PARAMETER num_ctx 4096\n";

fn highlight_modelfile(ui: &egui::Ui, text: &str, wrap_width: f32) -> std::sync::Arc<egui::Galley> {
    use egui::text::{LayoutJob, TextFormat};
    let mut job = LayoutJob::default();
    let directive_color = egui::Color32::from_rgb(198, 120, 221);
    let param_color = egui::Color32::from_rgb(97, 175, 239);
    let comment_color = egui::Color32::from_rgb(106, 153, 85);
    let string_color = egui::Color32::from_rgb(152, 195, 121);
    let default_color = ui.visuals().text_color();

    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            job.append("\n", 0.0, TextFormat::default());
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            job.append(line, 0.0, TextFormat { color: comment_color, ..Default::default() });
            continue;
        }
        let mut rest = line;
        let leading_ws = line.len() - trimmed.len();
        if leading_ws > 0 {
            job.append(&line[..leading_ws], 0.0, TextFormat::default());
            rest = &line[leading_ws..];
        }
        let word_end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        let word = &rest[..word_end];
        let upper = word.to_ascii_uppercase();
        if MODELFILE_DIRECTIVES.contains(&upper.as_str()) {
            job.append(word, 0.0, TextFormat { color: directive_color, ..Default::default() });
            let remainder = &rest[word_end..];
            if upper == "PARAMETER" {
                let p_end = remainder.trim_start().find(char::is_whitespace).map(|e| e + (remainder.len() - remainder.trim_start().len())).unwrap_or(remainder.len());
                let ws_len = remainder.len() - remainder.trim_start().len();
                job.append(&remainder[..ws_len], 0.0, TextFormat::default());
                let pname = remainder[ws_len..p_end].trim();
                let color = if MODELFILE_PARAMETERS.contains(&pname) { param_color } else { default_color };
                job.append(pname, 0.0, TextFormat { color, ..Default::default() });
                job.append(&remainder[p_end..], 0.0, TextFormat::default());
            } else if remainder.trim_start().starts_with('"') {
                job.append(remainder, 0.0, TextFormat { color: string_color, ..Default::default() });
            } else {
                job.append(remainder, 0.0, TextFormat::default());
            }
        } else {
            job.append(rest, 0.0, TextFormat::default());
        }
    }
    job.wrap.max_width = wrap_width;
    ui.fonts(|f| f.layout_job(job))
}

pub(crate) fn modelfile(ui: &mut egui::Ui, s: &mut AppState) {
    ui.horizontal(|ui| {
        ui.label("Zielname:");
        if ui.text_edit_singleline(&mut s.modelfile_target).changed() {
            s.refresh_modelfile_versions();
        }
        if ui.button("Validieren").clicked() {
            s.modelfile_errors = ollama_client::validate_modelfile(&s.modelfile_src);
        }
        if ui.button("Rebuild via Ollama").clicked() {
            s.modelfile_errors = ollama_client::validate_modelfile(&s.modelfile_src);
            if s.modelfile_errors.is_empty() {
                let _ = backup::backup_text(&s.modelfile_target, &s.modelfile_src);
                s.refresh_modelfile_versions();
                let (tx, c) = (s.tx.clone(), s.ollama.clone());
                let (name, src) = (s.modelfile_target.clone(), s.modelfile_src.clone());
                s.rt.spawn(async move {
                    let msg = match c.create(&name, &src).await {
                        Ok(()) => format!("Modell '{name}' erstellt"),
                        Err(e) => format!("Create fehlgeschlagen: {e}"),
                    };
                    let _ = tx.send(crate::state::AsyncMsg::Status(msg));
                });
            }
        }
        if ui.button("Einsteiger-Vorlage einfügen").clicked() {
            s.modelfile_src = MODELFILE_TEMPLATE_MINIMAL.to_string();
        }
    });
    for e in &s.modelfile_errors {
        ui.colored_label(egui::Color32::LIGHT_RED, e);
    }
    ui.separator();

    egui::SidePanel::right("modelfile_versions")
        .default_width(220.0)
        .show_inside(ui, |ui| {
            ui.heading("Versionen");
            if ui.button("Aktualisieren").clicked() {
                s.refresh_modelfile_versions();
            }
            egui::ScrollArea::vertical().show(ui, |ui| {
                for v in s.modelfile_versions.clone() {
                    let label = v.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(label).small());
                        if ui.small_button("Wiederherstellen").clicked() {
                            s.restore_modelfile_version(&v);
                        }
                    });
                }
                if s.modelfile_versions.is_empty() {
                    ui.label(RichText::new("Noch keine Backups.").weak());
                }
            });
        });

    egui::SidePanel::left("modelfile_help")
        .default_width(280.0)
        .show_inside(ui, |ui| {
            ui.heading("Was bedeutet das?");
            ui.label(RichText::new(
                "Ein Modelfile beschreibt, WIE Ollama ein Modell für dich vorbereitet — nicht was \
                 das Modell 'weiß' (das steckt in den GGUF-Gewichten), sondern wie es sich verhält: \
                 Tonfall, Kreativität, Gedächtnisgröße."
            ).weak().small());
            ui.add_space(6.0);

            ui.collapsing(RichText::new("Direktiven").strong(), |ui| {
                for &d in MODELFILE_DIRECTIVES {
                    ui.label(RichText::new(d).strong().monospace());
                    ui.label(RichText::new(directive_help(d)).small());
                    ui.add_space(4.0);
                }
            });
            ui.collapsing(RichText::new("PARAMETER-Werte").strong(), |ui| {
                for &p in MODELFILE_PARAMETERS {
                    ui.label(RichText::new(p).strong().monospace());
                    ui.label(RichText::new(parameter_help(p)).small());
                    ui.add_space(4.0);
                }
            });
        });

    egui::ScrollArea::vertical().show(ui, |ui| {
        let mut layouter = |ui: &egui::Ui, text: &str, wrap_width: f32| {
            highlight_modelfile(ui, text, wrap_width)
        };
        ui.add(
            egui::TextEdit::multiline(&mut s.modelfile_src)
                .desired_width(f32::INFINITY)
                .desired_rows(24)
                .layouter(&mut layouter),
        );
    });
}

pub(crate) fn chat(ui: &mut egui::Ui, s: &mut AppState) {
    ui.horizontal(|ui| {
        ui.label("Modell:");
        egui::ComboBox::from_id_salt("chat_model")
            .selected_text(&s.chat_model)
            .show_ui(ui, |ui| {
                for m in &s.models.clone() {
                    ui.selectable_value(&mut s.chat_model, m.name.clone(), &m.name);
                }
            });
    });
    ui.horizontal(|ui| {
        let resp = ui.add(
            egui::TextEdit::singleline(&mut s.chat_prompt)
                .hint_text("Prompt…")
                .desired_width(ui.available_width() - 80.0),
        );
        let send = ui.add_enabled(!s.chat_running, egui::Button::new("Senden")).clicked()
            || (resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)));
        if send { s.send_chat(); }
    });
    ui.separator();
    egui::ScrollArea::vertical().stick_to_bottom(true).max_height(ui.available_height() - 60.0).show(ui, |ui| {
        ui.label(&s.chat_output);
        if s.chat_running { ui.spinner(); }
    });

    if !s.chat_last_prompt.is_empty() && !s.chat_output.is_empty() && !s.chat_running {
        ui.separator();
        let has_gguf = s.gguf.is_some();
        ui.horizontal(|ui| {
            if ui.add_enabled(has_gguf, egui::Button::new("🔍 Erklären: woher kam diese Antwort?")).clicked() {
                s.explain_last_chat();
            }
            if !has_gguf {
                ui.label(RichText::new(
                    "Öffne zuerst die GGUF-Datei dieses Modells (Bibliothek → „Im GGUF Explorer öffnen“)."
                ).weak().small());
            }
        });
        ui.label(RichText::new(
            "Öffnet den Forward-Pass-Explorer mit demselben Prompt und rechnet die ersten Layer \
             mit den echten Gewichten der geöffneten Datei durch — inkl. Live-Log, welche Tensor- \
             Zeilen dabei gelesen wurden. Achtung: Ollama selbst läuft als eigener Prozess und gibt \
             über die API keine internen Rechendetails zurück — das ist daher eine Annäherung mit \
             denselben Gewichten (nur die ersten paar Layer), nicht die 1:1-Rekonstruktion der \
             tatsächlichen Ollama-Antwort."
        ).weak().small());
    }
}

pub(crate) fn tokenizer_panel(ui: &mut egui::Ui, s: &mut AppState) {
    let Some(g) = s.gguf.clone() else {
        ui.label("Keine GGUF-Datei geöffnet.");
        return;
    };
    let Some(info) = tokenizer::extract(&g) else {
        ui.label("Keine Tokenizer-Metadaten in dieser Datei gefunden.");
        return;
    };
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!(
            "Modell: {} · Vokabulargröße: {}",
            info.model.as_deref().unwrap_or("?"),
            info.vocab_size()
        )).strong());
    });
    if !info.special.is_empty() {
        ui.horizontal(|ui| {
            for sp in &info.special {
                ui.label(format!("{}={}", sp.name, sp.id));
            }
        });
    }
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("Suche (Text oder ID):");
        ui.text_edit_singleline(&mut s.tokenizer_filter);
    });
    let f = s.tokenizer_filter.trim();
    let f_lower = f.to_lowercase();
    let as_id: Option<u32> = f.parse().ok();
    let idx: Vec<usize> = info.tokens.iter().enumerate()
        .filter(|(i, t)| {
            f.is_empty()
                || t.text.to_lowercase().contains(&f_lower)
                || as_id.map(|id| id as usize == *i).unwrap_or(false)
        })
        .map(|(i, _)| i)
        .collect();

    let row_h = ui.text_style_height(&egui::TextStyle::Monospace);
    egui::ScrollArea::both().show_rows(ui, row_h, idx.len(), |ui, range| {
        egui::Grid::new("tokens").striped(true).show(ui, |ui| {
            ui.label(RichText::new("ID").strong());
            ui.label(RichText::new("Text").strong());
            ui.label(RichText::new("Bytes (Hex)").strong());
            ui.label(RichText::new("Typ").strong());
            ui.label(RichText::new("Score").strong());
            ui.end_row();
            for &i in &idx[range] {
                let t = &info.tokens[i];
                ui.monospace(t.id.to_string());
                ui.monospace(format!("{:?}", t.text));
                ui.monospace(tokenizer::bytes_hex(t));
                ui.label(t.kind.label());
                ui.label(t.score.map(|v| format!("{v:.3}")).unwrap_or_default());
                ui.end_row();
            }
        });
    });
}

fn diff_color(ui: &egui::Ui, status: &DiffStatus) -> egui::Color32 {
    match status {
        DiffStatus::Same => ui.visuals().text_color(),
        DiffStatus::Changed => egui::Color32::from_rgb(230, 180, 60),
        DiffStatus::OnlyInA => egui::Color32::from_rgb(220, 100, 100),
        DiffStatus::OnlyInB => egui::Color32::from_rgb(100, 180, 220),
    }
}

pub(crate) fn compare(ui: &mut egui::Ui, s: &mut AppState) {
    ui.horizontal(|ui| {
        ui.heading("Modellvergleich");
        if ui.button("Vergleichsdatei öffnen…").clicked() {
            s.open_compare_dialog();
        }
    });
    let (Some(_a), Some(b)) = (&s.gguf, &s.compare_gguf) else {
        ui.label("A: aktuell geöffnete GGUF-Datei · B: über 'Vergleichsdatei öffnen…' laden.");
        return;
    };
    ui.label(format!(
        "A: {}   ↔   B: {}",
        s.gguf.as_ref().unwrap().path.display(),
        b.path.display()
    ));
    let Some(diff) = &s.compare_diff else {
        ui.label("Berechne Diff…");
        return;
    };

    ui.collapsing("Header", |ui| {
        egui::Grid::new("diff_header").striped(true).show(ui, |ui| {
            for h in &diff.header {
                ui.label(h.field);
                ui.colored_label(diff_color(ui, &h.status), &h.a);
                ui.colored_label(diff_color(ui, &h.status), &h.b);
                ui.end_row();
            }
        });
    });

    ui.collapsing(format!("Metadaten ({} Einträge)", diff.metadata.len()), |ui| {
        let changed_only = diff.metadata.iter().filter(|m| m.status != DiffStatus::Same).count();
        ui.label(format!("{changed_only} unterschiedlich"));
        egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
            egui::Grid::new("diff_meta").striped(true).show(ui, |ui| {
                for m in &diff.metadata {
                    if m.status == DiffStatus::Same { continue; }
                    ui.label(RichText::new(&m.key).monospace());
                    ui.colored_label(diff_color(ui, &m.status), m.a.clone().unwrap_or_else(|| "—".into()));
                    ui.colored_label(diff_color(ui, &m.status), m.b.clone().unwrap_or_else(|| "—".into()));
                    ui.end_row();
                }
            });
        });
    });

    ui.collapsing(format!("Tensoren ({} Einträge)", diff.tensors.len()), |ui| {
        let changed_only = diff.tensors.iter().filter(|t| t.status != DiffStatus::Same).count();
        ui.label(format!("{changed_only} unterschiedlich"));
        egui::ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
            egui::Grid::new("diff_tensors").striped(true).show(ui, |ui| {
                for t in &diff.tensors {
                    if t.status == DiffStatus::Same { continue; }
                    ui.label(RichText::new(&t.name).monospace());
                    let fa = t.a.as_ref().map(|(dt, sh, sz)| format!("{dt} {sh:?} {}", human_bytes(*sz))).unwrap_or_else(|| "—".into());
                    let fb = t.b.as_ref().map(|(dt, sh, sz)| format!("{dt} {sh:?} {}", human_bytes(*sz))).unwrap_or_else(|| "—".into());
                    ui.colored_label(diff_color(ui, &t.status), fa);
                    ui.colored_label(diff_color(ui, &t.status), fb);
                    ui.end_row();
                }
            });
        });
    });

    ui.collapsing("Tokenizer", |ui| {
        ui.label(format!(
            "Vokabulargröße A: {} · B: {}",
            diff.tokenizer_vocab_a.map(|v| v.to_string()).unwrap_or_else(|| "—".into()),
            diff.tokenizer_vocab_b.map(|v| v.to_string()).unwrap_or_else(|| "—".into()),
        ));
        if diff.tokenizer_diff_sample.is_empty() {
            ui.label("Keine unterschiedlichen Tokens (oder kein Tokenizer in einer der Dateien).");
        } else {
            ui.label(format!("{} unterschiedliche Tokens (Ausschnitt):", diff.tokenizer_diff_sample.len()));
            egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                egui::Grid::new("diff_tok").striped(true).show(ui, |ui| {
                    for (id, a, b) in &diff.tokenizer_diff_sample {
                        ui.monospace(id.to_string());
                        ui.label(a.clone().unwrap_or_else(|| "—".into()));
                        ui.label(b.clone().unwrap_or_else(|| "—".into()));
                        ui.end_row();
                    }
                });
            });
        }
    });
}

pub(crate) fn forward_panel(ui: &mut egui::Ui, s: &mut AppState) {
    if s.gguf.is_none() {
        ui.label("Keine GGUF-Datei geöffnet.");
        return;
    }
    ui.heading("Forward-Pass-Explorer");
    ui.label(RichText::new(
        "Rechnet Embedding + die ersten Layer echt mit den Gewichten dieser Datei durch \
         (keine Simulation) — für kurze Prompts und wenige Layer, zum Nachvollziehen der \
         Architektur. Ergebnisse können vom vollständigen Modell abweichen, da spätere \
         Layer nicht mitgerechnet werden."
    ).weak().small());
    ui.separator();

    ui.horizontal(|ui| {
        ui.label("Prompt:");
        ui.text_edit_singleline(&mut s.forward_prompt);
    });
    ui.horizontal(|ui| {
        ui.label("Anzahl Layer (1-4 empfohlen):");
        ui.add(egui::Slider::new(&mut s.forward_n_layers, 1..=8));
        if ui.add_enabled(!s.forward_running, egui::Button::new("Durchrechnen")).clicked() {
            s.run_forward_pass();
        }
        if s.forward_running {
            ui.spinner();
            ui.label("Rechnet…");
        }
    });

    if let Some(err) = &s.forward_error {
        ui.colored_label(egui::Color32::LIGHT_RED, err);
    }

    let Some(result) = s.forward_result.clone() else {
        ui.label("Noch keine Berechnung durchgeführt.");
        return;
    };

    let mut jump_target: Option<u64> = None;
    egui::ScrollArea::vertical().show(ui, |ui| {
        for (i, step) in result.steps.iter().enumerate() {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.label(RichText::new(format!("{}. {}", i + 1, step.label)).strong());
                ui.label(&step.explanation);
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("Shape: {}", step.shape)).monospace().weak());
                });
                let sample: String = step.sample.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(", ");
                ui.label(RichText::new(format!("Erste Werte: [{sample}, …]")).monospace().small());
                if !step.hex_refs.is_empty() {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Verwendete Datei-Ausschnitte:").weak().small());
                        for r in &step.hex_refs {
                            let label = format!("{} (Offset {:#x}, {} Bytes)", r.tensor_name, r.file_offset, r.byte_len);
                            if ui.small_button(format!("🔎 {label}")).clicked() {
                                jump_target = Some(r.file_offset);
                            }
                        }
                    });
                }
            });
            ui.add_space(6.0);
        }

        if !result.attention.is_empty() {
            ui.collapsing("Attention-Gewichte (Layer 0, letzte Position)", |ui| {
                ui.label(RichText::new(
                    "Zeigt, wie stark jeder Attention-Head der letzten Token-Position auf \
                     jede vorherige Position 'achtet' (Softmax-Gewichte, Summe = 1)."
                ).weak().small());
                for at in &result.attention {
                    ui.horizontal(|ui| {
                        ui.label(format!("Head {}: ", at.head));
                        for (pos, &wgt) in at.weights.iter().enumerate() {
                            ui.add(egui::ProgressBar::new(wgt).desired_width(30.0).text(format!("{pos}")));
                        }
                    });
                }
            });
        }

        if !result.top_next_tokens.is_empty() {
            ui.collapsing("Vorhersage für das nächste Token (volles Modell nötig für Genauigkeit!)", |ui| {
                ui.label(RichText::new(
                    "Achtung: da nur die ersten paar Layer gerechnet wurden, ist diese Vorhersage \
                     NICHT die tatsächliche Modellausgabe — sie dient nur der Veranschaulichung, \
                     wie aus dem letzten Hidden-State per Output-Projektion + Softmax eine \
                     Wahrscheinlichkeitsverteilung über das gesamte Vokabular entsteht."
                ).weak().small());
                egui::Grid::new("top_tokens").striped(true).show(ui, |ui| {
                    ui.label(RichText::new("Token-ID").strong());
                    ui.label(RichText::new("Logit").strong());
                    ui.label(RichText::new("Wahrscheinlichkeit").strong());
                    ui.end_row();
                    for (id, logit, prob) in &result.top_next_tokens {
                        ui.monospace(id.to_string());
                        ui.monospace(format!("{logit:.4}"));
                        ui.monospace(format!("{:.2}%", prob * 100.0));
                        ui.end_row();
                    }
                });
            });
        }
    });

    if let Some(offset) = jump_target {
        s.jump_to_hex(offset);
    }
}

pub(crate) fn forward_log_panel(ui: &mut egui::Ui, s: &mut AppState) {
    ui.heading("Live-Ausführungslog");
    ui.label(RichText::new(
        "Jeder tatsächliche Datei-Lesezugriff während des letzten Forward-Pass-Laufs, in exakt \
         der Reihenfolge, in der er stattgefunden hat — unabhängig davon, was du eingegeben hast. \
         Das ist keine Simulation: jede Zeile entspricht einem echten mmap-Zugriff auf die Datei."
    ).weak().small());
    ui.separator();

    if s.forward_running {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Läuft — Log füllt sich live…");
        });
    }

    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{} Lesezugriffe", s.forward_log.len())).strong());
        ui.label(format!("· {} gelesen", human_bytes(s.forward_log_bytes)));
        if let Some(d) = s.forward_duration {
            ui.label(format!("· {:.1} ms", d.as_secs_f64() * 1000.0));
        }
        if !s.forward_log.is_empty() && ui.button("Log exportieren…").clicked() {
            s.export_forward_log();
        }
    });
    ui.separator();

    if s.forward_log.is_empty() {
        ui.label("Noch kein Lauf. Starte eine Berechnung im Forward-Pass-Explorer.");
        return;
    }

    let row_h = ui.text_style_height(&egui::TextStyle::Monospace);
    egui::ScrollArea::vertical()
        .stick_to_bottom(true)
        .max_height(ui.available_height() * 0.6)
        .show_rows(ui, row_h, s.forward_log.len(), |ui, range| {
            egui::Grid::new("forward_log_grid").striped(true).min_col_width(60.0).show(ui, |ui| {
                ui.label(RichText::new("#").strong());
                ui.label(RichText::new("Schritt").strong());
                ui.label(RichText::new("Tensor").strong());
                ui.label(RichText::new("Offset").strong());
                ui.label(RichText::new("Bytes").strong());
                ui.label(RichText::new("Grund").strong());
                ui.end_row();
                for i in range {
                    let ev = &s.forward_log[i];
                    ui.monospace(format!("{}", i + 1));
                    ui.monospace(&ev.step_label);
                    ui.monospace(&ev.tensor_name);
                    ui.monospace(format!("{:#x}", ev.file_offset));
                    ui.monospace(human_bytes(ev.byte_len));
                    ui.label(RichText::new(&ev.reason).weak().small());
                    ui.end_row();
                }
            });
        });

    if !s.forward_running {
        ui.separator();
        ui.collapsing(RichText::new("Zusammenfassung nach Tensor").strong(), |ui| {
            let mut by_tensor: std::collections::BTreeMap<String, (u32, u64)> = Default::default();
            for ev in &s.forward_log {
                let e = by_tensor.entry(ev.tensor_name.clone()).or_insert((0, 0));
                e.0 += 1;
                e.1 += ev.byte_len;
            }
            egui::Grid::new("forward_log_summary").striped(true).show(ui, |ui| {
                ui.label(RichText::new("Tensor").strong());
                ui.label(RichText::new("Zugriffe").strong());
                ui.label(RichText::new("Bytes gesamt").strong());
                ui.end_row();
                for (name, (count, bytes)) in &by_tensor {
                    ui.monospace(name);
                    ui.monospace(count.to_string());
                    ui.monospace(human_bytes(*bytes));
                    ui.end_row();
                }
            });
        });
    }
}
