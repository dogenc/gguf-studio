//! Plugin-Registry: jedes Panel wird über ein `PanelPlugin`-Trait-Objekt
//! bereitgestellt statt hart über ein Enum verdrahtet zu sein. Neue Panels
//! (auch aus zukünftigen dynamisch geladenen Erweiterungen) registrieren
//! sich einfach über `PluginRegistry::register`.

use crate::state::AppState;
use eframe::egui;

pub trait PanelPlugin: Send + Sync {
    /// Eindeutiger, stabiler Bezeichner (z. B. für Layout-Persistenz).
    fn id(&self) -> &'static str;
    /// Anzeigename im Tab.
    fn title(&self) -> &'static str;
    /// Rendert den Panel-Inhalt.
    fn ui(&self, ui: &mut egui::Ui, state: &mut AppState);
}

pub struct PluginRegistry {
    plugins: Vec<Box<dyn PanelPlugin>>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self { plugins: Vec::new() }
    }

    pub fn register(&mut self, plugin: Box<dyn PanelPlugin>) -> &mut Self {
        self.plugins.push(plugin);
        self
    }

    pub fn by_id(&self, id: &str) -> Option<&dyn PanelPlugin> {
        self.plugins.iter().find(|p| p.id() == id).map(|b| b.as_ref())
    }

    pub fn ids(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.plugins.iter().map(|p| p.id())
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

macro_rules! panel_plugin {
    ($struct_name:ident, $id:literal, $title:literal, $func:path) => {
        pub struct $struct_name;
        impl PanelPlugin for $struct_name {
            fn id(&self) -> &'static str { $id }
            fn title(&self) -> &'static str { $title }
            fn ui(&self, ui: &mut egui::Ui, state: &mut AppState) { $func(ui, state) }
        }
    };
}

panel_plugin!(LibraryPlugin, "library", "Modell-Bibliothek", crate::panels::library);
panel_plugin!(ExplorerPlugin, "explorer", "GGUF Explorer", crate::panels::explorer);
panel_plugin!(TensorsPlugin, "tensors", "Tensor-Browser", crate::panels::tensors);
panel_plugin!(QuantPlugin, "quant", "Quantisierung", crate::panels::quantization);
panel_plugin!(HexPlugin, "hex", "Hex-Viewer", crate::panels::hex);
panel_plugin!(ModelfilePlugin, "modelfile", "Modelfile-Editor", crate::panels::modelfile);
panel_plugin!(ChatPlugin, "chat", "Test / Chat", crate::panels::chat);
panel_plugin!(TokenizerPlugin, "tokenizer", "Tokenizer", crate::panels::tokenizer_panel);
panel_plugin!(ComparePlugin, "compare", "Modellvergleich", crate::panels::compare);
panel_plugin!(ForwardPlugin, "forward", "Forward-Pass-Explorer", crate::panels::forward_panel);
panel_plugin!(ForwardLogPlugin, "forward_log", "Live-Ausführungslog", crate::panels::forward_log_panel);

/// Registriert alle eingebauten Panels. Zukünftige Plugins (z. B. dynamisch
/// aus einer `plugins/`-Verzeichnisstruktur geladen) hängen sich hier über
/// `register` zusätzlich ein, ohne dass Kernmodule angefasst werden müssen.
pub fn builtin_registry() -> PluginRegistry {
    let mut r = PluginRegistry::new();
    r.register(Box::new(LibraryPlugin))
        .register(Box::new(ExplorerPlugin))
        .register(Box::new(TensorsPlugin))
        .register(Box::new(QuantPlugin))
        .register(Box::new(HexPlugin))
        .register(Box::new(ModelfilePlugin))
        .register(Box::new(ChatPlugin))
        .register(Box::new(TokenizerPlugin))
        .register(Box::new(ComparePlugin))
        .register(Box::new(ForwardPlugin))
        .register(Box::new(ForwardLogPlugin));
    r
}
