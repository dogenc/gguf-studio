//! DGKN@Labs – GGUF Studio
//! Docking-UI (egui_dock) mit modularen Panels: Bibliothek, GGUF Explorer,
//! Tensor-Browser, Quant-Analyzer, Hex-Viewer, Modelfile-Editor, Chat/Test.

mod backup;
mod panels;
mod plugin;
mod splash;
mod state;

use eframe::egui::{self, Vec2};
use egui_dock::{DockArea, DockState, NodeIndex};
use panels::PanelId;
use plugin::PluginRegistry;
use splash::{Splash, SPLASH_SIZE};
use state::AppState;

const APP_SIZE: (f32, f32) = (1440.0, 900.0);

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(SPLASH_SIZE)
            .with_resizable(false)
            .with_decorations(false)
            .with_title("DGKN@Labs – GGUF Studio"),
        ..Default::default()
    };
    eframe::run_native(
        "GGUF Studio",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(StudioApp::new()))
        }),
    )
}

struct StudioApp {
    dock: DockState<PanelId>,
    state: AppState,
    registry: PluginRegistry,
    splash: Splash,
    transitioned: bool,
}

impl StudioApp {
    fn new() -> Self {
        let mut dock = DockState::new(vec!["library"]);
        let surface = dock.main_surface_mut();
        let [left, center] = surface.split_right(NodeIndex::root(), 0.22, vec!["explorer", "tensors"]);
        let [center, right] = surface.split_right(center, 0.55, vec!["quant", "tokenizer", "compare", "forward"]);
        surface.split_below(left, 0.6, vec!["hex"]);
        surface.split_below(center, 0.6, vec!["modelfile"]);
        surface.split_below(right, 0.6, vec!["chat", "forward_log"]);
        Self {
            dock,
            state: AppState::new(),
            registry: plugin::builtin_registry(),
            splash: Splash::new(),
            transitioned: false,
        }
    }

    /// Öffnet ein Panel: falls es bereits offen ist, wird sein Tab
    /// fokussiert; sonst wird es als neuer Tab im aktiven Bereich eingefügt.
    fn open_panel(&mut self, id: PanelId) {
        if let Some((surface, node, tab)) = self.dock.find_tab(&id) {
            self.dock.set_active_tab((surface, node, tab));
        } else {
            self.dock.push_to_focused_leaf(id);
        }
    }

    fn is_panel_open(&self, id: PanelId) -> bool {
        self.dock.find_tab(&id).is_some()
    }
}

impl eframe::App for StudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.splash.active() {
            self.splash.show(ctx);
            return;
        }
        if !self.transitioned {
            self.transitioned = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(true));
            ctx.send_viewport_cmd(egui::ViewportCommand::Resizable(true));
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(Vec2::from(APP_SIZE)));
            if let Some(cmd) = egui::ViewportCommand::center_on_screen(ctx) {
                ctx.send_viewport_cmd(cmd);
            }
        }

        self.state.poll_async(ctx);

        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("Datei", |ui| {
                    if ui.button("GGUF öffnen…  (Strg+O)").clicked() {
                        self.state.open_file_dialog();
                        ui.close_menu();
                    }
                    if ui.button("Vergleichsdatei öffnen…").clicked() {
                        self.state.open_compare_dialog();
                        ui.close_menu();
                    }
                });
                ui.menu_button("Ollama", |ui| {
                    if ui.button("Modelle neu laden").clicked() {
                        self.state.refresh_models();
                        ui.close_menu();
                    }
                });
                ui.menu_button("Ansicht", |ui| {
                    if ui.button("Dark Mode").clicked() { ctx.set_visuals(egui::Visuals::dark()); }
                    if ui.button("Light Mode").clicked() { ctx.set_visuals(egui::Visuals::light()); }
                });
                ui.menu_button("Fenster", |ui| {
                    for id in self.registry.ids().collect::<Vec<_>>() {
                        let title = self.registry.by_id(id).map(|p| p.title()).unwrap_or(id);
                        let mut open = self.is_panel_open(id);
                        if ui.checkbox(&mut open, title).clicked() {
                            if open {
                                self.open_panel(id);
                            } else if let Some(tab) = self.dock.find_tab(&id) {
                                self.dock.remove_tab(tab);
                            }
                            ui.close_menu();
                        }
                    }
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(self.state.status_line());
                });
            });
        });

        if ctx.input(|i| i.key_pressed(egui::Key::O) && i.modifiers.ctrl) {
            self.state.open_file_dialog();
        }

        DockArea::new(&mut self.dock)
            .show(ctx, &mut panels::PanelViewer { state: &mut self.state, registry: &self.registry });

        if let Some(id) = self.state.jump_to_panel.take() {
            self.open_panel(id);
        }
    }
}
