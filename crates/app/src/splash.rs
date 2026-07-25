//! Animierter Splashscreen: professioneller Daten-Globus mit echtem
//! Breiten-/Längengrad-Gitter (Latitude-Ringe + Longitude-Meridiane) über
//! schattierten Kontinenten, auf reinem Schwarz. Dazu Logo-Einblendung und
//! Ladebalken mit großzügigem Abstand zur Kugel. Rein dekorativ — läuft für
//! eine feste, kurze Dauer in einem kompakten Fenster, dann übernimmt die
//! Haupt-UI (die auf die reguläre Fenstergröße wechselt).

use eframe::egui::{self, Color32, FontId, Pos2, Rect, Stroke, Vec2};

pub const SPLASH_SIZE: (f32, f32) = (460.0, 460.0);
const DURATION_SECS: f32 = 5.0;

const LAT_RINGS: usize = 17;
const LON_MERIDIANS: usize = 17;
const GRID_STEPS: usize = 90;
const CONTINENT_POINTS: usize = 1300;

/// Winziger deterministischer PRNG (xorshift64) — kein externer Zufalls-Crate nötig.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 1_000_000) as f32 / 1_000_000.0
    }
}

/// Grobes Wertrauschen über (lat, lon) für zusammenhängende "Kontinent"-Flecken.
fn land_noise(lat: f32, lon: f32) -> f32 {
    let a = (lat * 2.3 + lon * 1.7).sin();
    let b = (lat * 4.1 - lon * 2.9 + 1.3).sin();
    let c = (lat * 1.1 + lon * 5.3 + 2.7).cos();
    (a * 0.5 + b * 0.3 + c * 0.2 + 1.0) * 0.5
}

/// Ein Gitter-Kreis (entweder ein Breitengrad-Ring oder ein Längengrad-
/// Meridian) als Folge fester (lat, lon)-Paare, die zusammen rotieren.
struct GridLine {
    points: Vec<(f32, f32)>, // (lat, lon)
}

struct ContinentPoint {
    lat: f32,
    lon: f32,
}

pub struct Splash {
    start: Option<f32>,
    elapsed: f32,
    grid: Vec<GridLine>,
    continent: Vec<ContinentPoint>,
}

impl Splash {
    pub fn new() -> Self {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for _ in 0..3 { rng.next(); }

        let mut grid = Vec::with_capacity(LAT_RINGS + LON_MERIDIANS);

        // Breitengrad-Ringe (horizontale Kreise), Pole ausgespart.
        for r in 1..LAT_RINGS {
            let lat = -std::f32::consts::FRAC_PI_2
                + r as f32 / LAT_RINGS as f32 * std::f32::consts::PI;
            let points = (0..=GRID_STEPS)
                .map(|s| (lat, s as f32 / GRID_STEPS as f32 * std::f32::consts::TAU))
                .collect();
            grid.push(GridLine { points });
        }
        // Längengrad-Meridiane (vertikale Halbkreise von Pol zu Pol).
        for m in 0..LON_MERIDIANS {
            let lon0 = m as f32 / LON_MERIDIANS as f32 * std::f32::consts::TAU;
            let points = (0..=GRID_STEPS)
                .map(|s| {
                    let lat = -std::f32::consts::FRAC_PI_2 + s as f32 / GRID_STEPS as f32 * std::f32::consts::PI;
                    (lat, lon0)
                })
                .collect();
            grid.push(GridLine { points });
        }

        // Fibonacci-Sphäre für die Kontinent-Schattierung (nahezu
        // gleichverteilte Punkte), nur "Land"-Punkte werden später gefüllt.
        let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        let continent = (0..CONTINENT_POINTS)
            .filter_map(|i| {
                let y = 1.0 - (i as f32 / (CONTINENT_POINTS - 1) as f32) * 2.0;
                let lat = y.asin();
                let lon = (golden * i as f32).rem_euclid(std::f32::consts::TAU);
                if land_noise(lat, lon) > 0.55 { Some(ContinentPoint { lat, lon }) } else { None }
            })
            .collect();

        let _ = rng.unit();
        Self { start: None, elapsed: 0.0, grid, continent }
    }

    pub fn active(&self) -> bool {
        self.elapsed < DURATION_SECS
    }

    pub fn show(&mut self, ctx: &egui::Context) {
        let now = ctx.input(|i| i.time) as f32;
        let start = *self.start.get_or_insert(now);
        self.elapsed = now - start;
        ctx.request_repaint();

        let t = (self.elapsed / DURATION_SECS).clamp(0.0, 1.0);
        let fade_in = (self.elapsed / 0.45).clamp(0.0, 1.0);
        let fade_out = ((DURATION_SECS - self.elapsed) / 0.4).clamp(0.0, 1.0);
        let alpha = fade_in.min(fade_out);

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(Color32::BLACK))
            .show(ctx, |ui| {
                let rect = ui.max_rect();
                let painter = ui.painter();

                let radius = 88.0;
                // Großzügiger, fester Abstand zwischen Kugel und Text, damit
                // sich nichts überlappt: Titel/Untertitel klar oberhalb,
                // Status/Ladebalken klar unterhalb der Kugel-Silhouette
                // (inkl. Halo-Rand) platziert.
                let globe_center = rect.center() - Vec2::new(0.0, 18.0);
                let halo_edge = radius * 1.22;
                let title_y = globe_center.y - halo_edge - 46.0;
                let subtitle_y = title_y + 32.0;
                let status_y = globe_center.y + halo_edge + 34.0;
                let bar_y = status_y + 22.0;

                self.draw_globe(painter, globe_center, radius, alpha);

                // Logo-Text, buchstabenweise eingeblendet.
                let title = "GGUF STUDIO";
                let reveal = ((self.elapsed - 0.15) / 0.8).clamp(0.0, 1.0);
                let n_visible = (title.chars().count() as f32 * reveal).ceil() as usize;
                let visible: String = title.chars().take(n_visible).collect();
                painter.text(
                    Pos2::new(globe_center.x, title_y),
                    egui::Align2::CENTER_CENTER,
                    &visible,
                    FontId::monospace(26.0),
                    Color32::from_rgba_unmultiplied(215, 235, 255, (255.0 * alpha) as u8),
                );

                let sub_alpha = ((self.elapsed - 1.0) / 0.5).clamp(0.0, 1.0) * alpha;
                painter.text(
                    Pos2::new(globe_center.x, subtitle_y),
                    egui::Align2::CENTER_CENTER,
                    "DGKN@Labs — GGUF & Ollama Workbench",
                    FontId::monospace(11.0),
                    Color32::from_rgba_unmultiplied(110, 150, 180, (255.0 * sub_alpha) as u8),
                );

                // Rotierender Status-Text.
                const STATUSES: &[&str] = &[
                    "initialisiere mmap-engine…",
                    "suche ollama-modelle…",
                    "lade plugin-registry…",
                    "bereite docking-ui vor…",
                ];
                let status_idx = ((self.elapsed / (DURATION_SECS / STATUSES.len() as f32)) as usize).min(STATUSES.len() - 1);
                let status_alpha = ((self.elapsed - 0.3) / 0.4).clamp(0.0, 1.0) * alpha;
                painter.text(
                    Pos2::new(globe_center.x, status_y),
                    egui::Align2::CENTER_CENTER,
                    STATUSES[status_idx],
                    FontId::monospace(11.0),
                    Color32::from_rgba_unmultiplied(90, 130, 160, (255.0 * status_alpha) as u8),
                );

                // Kompakter, glühender Ladebalken.
                let bar_w = 220.0;
                let bar_h = 2.0;
                let bar_rect = Rect::from_center_size(Pos2::new(globe_center.x, bar_y), Vec2::new(bar_w, bar_h));
                painter.rect_filled(bar_rect, 1.0, Color32::from_rgba_unmultiplied(120, 160, 190, (24.0 * alpha) as u8));
                let fill_rect = Rect::from_min_size(bar_rect.min, Vec2::new(bar_w * t, bar_h));
                painter.rect_filled(fill_rect, 1.0, Color32::from_rgba_unmultiplied(140, 210, 255, (230.0 * alpha) as u8));
                let glow_rect = fill_rect.expand2(Vec2::new(0.0, 1.5));
                painter.rect_filled(glow_rect, 1.5, Color32::from_rgba_unmultiplied(140, 210, 255, (40.0 * alpha) as u8));
            });
    }

    /// Zeichnet den Globus: schattierte Kontinent-Punktwolke unter einem
    /// echten Breiten-/Längengrad-Gitter, auf reinem Schwarz — angelehnt an
    /// professionelle Daten-Globus-Visualisierungen.
    fn draw_globe(&self, painter: &egui::Painter, center: Pos2, radius: f32, alpha: f32) {
        let rot = self.elapsed * 0.4;
        let tilt = 0.32_f32;
        let (tilt_sin, tilt_cos) = tilt.sin_cos();

        let project = |lat: f32, lon: f32| -> (Pos2, f32) {
            let lon = lon + rot;
            let (lat_sin, lat_cos) = lat.sin_cos();
            let (s_lon, c_lon) = lon.sin_cos();
            let x0 = lat_cos * s_lon;
            let y0 = lat_sin;
            let z0 = lat_cos * c_lon;
            let y = y0 * tilt_cos - z0 * tilt_sin;
            let z = y0 * tilt_sin + z0 * tilt_cos;
            (Pos2::new(center.x + x0 * radius, center.y + y * radius), z)
        };

        // Weicher Halo strikt außerhalb des Radius.
        for i in (0..6).rev() {
            let f = i as f32 / 6.0;
            let a = ((1.0 - f) * 16.0 * alpha) as u8;
            painter.circle_filled(center, radius * (1.04 + 0.2 * f), Color32::from_rgba_unmultiplied(60, 130, 190, a));
        }

        // Dunkle Kugel-Grundfläche, damit die Rückseite des Gitters nicht
        // durchscheint und Kontinente sich klar vom "Ozean" abheben.
        painter.circle_filled(center, radius, Color32::from_rgb(6, 10, 16));

        // Kontinent-Schattierung (unter dem Gitter).
        for p in &self.continent {
            let (screen, z) = project(p.lat, p.lon);
            if z < -0.03 { continue; }
            let shade = (0.35 + 0.65 * z.clamp(0.0, 1.0)).min(1.0);
            let r = (74.0 * shade) as u8;
            let g = (86.0 * shade) as u8;
            let b = (98.0 * shade) as u8;
            let a = (shade * 235.0 * alpha) as u8;
            painter.circle_filled(screen, 1.7, Color32::from_rgba_unmultiplied(r, g, b, a));
        }

        // Breiten-/Längengrad-Gitter, gleichmäßig hell (leichtes Rim-Glow).
        for line in &self.grid {
            let mut prev: Option<Pos2> = None;
            for &(lat, lon) in &line.points {
                let (screen, z) = project(lat, lon);
                if z < -0.02 {
                    prev = None;
                    continue;
                }
                let rim = (1.0 - z.clamp(0.0, 1.0)).powf(1.4);
                let intensity = (0.62 + 0.38 * rim).min(1.0);
                let a = (intensity * 165.0 * alpha) as u8;
                let color = Color32::from_rgba_unmultiplied(150, 205, 245, a);
                if let Some(prev_pos) = prev {
                    painter.line_segment([prev_pos, screen], Stroke::new(1.0_f32, color));
                }
                prev = Some(screen);
            }
        }

        // Kristalline Rand-Silhouette.
        painter.circle_stroke(center, radius, Stroke::new(1.3_f32, Color32::from_rgba_unmultiplied(175, 220, 255, (90.0 * alpha) as u8)));
    }
}
