//! Löst Ollama-Modellnamen zu ihrem GGUF-Blob-Pfad auf dem lokalen
//! Dateisystem auf. Ollama speichert Modelldateien ohne `.gguf`-Endung unter
//! `<models>/blobs/sha256-<digest>` und referenziert sie über JSON-Manifeste
//! unter `<models>/manifests/<registry>/<namespace>/<name>/<tag>`.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
struct Manifest {
    layers: Vec<Layer>,
}

#[derive(Debug, Deserialize)]
struct Layer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
}

/// Basisverzeichnis der Ollama-Modelldaten (`OLLAMA_MODELS` env var oder
/// Standardpfad `~/.ollama/models`).
pub fn models_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("OLLAMA_MODELS") {
        return PathBuf::from(dir);
    }
    dirs::home_dir().unwrap_or_default().join(".ollama").join("models")
}

/// Zerlegt einen Ollama-Modellnamen (`name:tag`, ggf. mit Registry/Namespace,
/// z. B. `registry.ollama.ai/library/llama3:8b`) in den Manifest-Pfad.
fn manifest_path(models_dir: &Path, name: &str) -> PathBuf {
    let (repo, tag) = name.split_once(':').unwrap_or((name, "latest"));
    let parts: Vec<&str> = repo.split('/').collect();
    let (registry, namespace, model) = match parts.as_slice() {
        [m] => ("registry.ollama.ai".to_string(), "library".to_string(), m.to_string()),
        [ns, m] => ("registry.ollama.ai".to_string(), ns.to_string(), m.to_string()),
        [reg, ns, m] => (reg.to_string(), ns.to_string(), m.to_string()),
        _ => ("registry.ollama.ai".to_string(), "library".to_string(), repo.to_string()),
    };
    models_dir.join("manifests").join(registry).join(namespace).join(model).join(tag)
}

fn digest_to_blob_path(models_dir: &Path, digest: &str) -> PathBuf {
    // Manifest-Digests sind "sha256:abc…", Blob-Dateinamen "sha256-abc…".
    let file_name = digest.replace(':', "-");
    models_dir.join("blobs").join(file_name)
}

/// Findet den Dateipfad des GGUF-Modell-Blobs für einen Ollama-Modellnamen.
pub fn resolve_model_path(name: &str) -> Result<PathBuf> {
    let dir = models_dir();
    let manifest_file = manifest_path(&dir, name);
    let content = std::fs::read_to_string(&manifest_file)
        .with_context(|| format!("Manifest nicht gefunden: {}", manifest_file.display()))?;
    let manifest: Manifest = serde_json::from_str(&content)
        .with_context(|| format!("Manifest konnte nicht gelesen werden: {}", manifest_file.display()))?;
    let model_layer = manifest.layers.iter()
        .find(|l| l.media_type == "application/vnd.ollama.image.model")
        .ok_or_else(|| anyhow!("Kein Modell-Layer im Manifest für '{name}' gefunden"))?;
    let blob = digest_to_blob_path(&dir, &model_layer.digest);
    if !blob.exists() {
        return Err(anyhow!("Blob-Datei fehlt: {}", blob.display()));
    }
    Ok(blob)
}
