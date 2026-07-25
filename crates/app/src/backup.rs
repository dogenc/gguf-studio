//! Automatische Backups vor jeder Änderung. Textinhalte (z. B. Modelfiles)
//! werden zeitgestempelt und mit SHA-256-Hash im Nutzerdatenverzeichnis
//! abgelegt — einfache, nachvollziehbare Versionshistorie.

use anyhow::Result;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

pub fn backup_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("gguf-studio")
        .join("backups")
}

pub fn backup_text(name: &str, content: &str) -> Result<PathBuf> {
    let dir = backup_dir().join(sanitize(name));
    fs::create_dir_all(&dir)?;
    let hash = hex(&Sha256::digest(content.as_bytes())[..6]);
    let ts = chrono::Local::now().format("%Y%m%d-%H%M%S");
    let path = dir.join(format!("{ts}-{hash}.txt"));
    fs::write(&path, content)?;
    Ok(path)
}

pub fn list_versions(name: &str) -> Vec<PathBuf> {
    let dir = backup_dir().join(sanitize(name));
    let mut v: Vec<_> = fs::read_dir(&dir)
        .map(|it| it.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    v.sort();
    v.reverse();
    v
}

/// Verzeichnis für binäre Datei-Backups (z. B. Hex-Editor-Änderungen).
pub fn binary_backup_dir() -> PathBuf {
    backup_dir().join("files")
}

/// Liest eine Backup-Version zurück ein.
pub fn read_version(path: &PathBuf) -> Result<String> {
    Ok(fs::read_to_string(path)?)
}

fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
