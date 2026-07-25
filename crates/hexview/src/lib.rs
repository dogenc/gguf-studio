//! Block-basierter Hex-Reader für beliebig große Dateien.
//! Es wird ausschließlich das sichtbare Fenster gelesen (mmap-Slice),
//! niemals die gesamte Datei. Geeignet für virtuelles Scrolling.
//!
//! Editieren erfolgt Copy-on-Write: einzelne Byte-Änderungen werden in einer
//! sparsen Patch-Map gehalten, bis `commit` sie — nach automatischem Backup
//! der Originaldatei — auf die Platte schreibt.

use anyhow::Result;
use memmap2::Mmap;
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

pub const BYTES_PER_ROW: usize = 16;

pub struct HexSource {
    path: PathBuf,
    mmap: Mmap,
    pub len: u64,
    /// Ungespeicherte Byte-Änderungen (Copy-on-Write, Offset -> neuer Wert).
    patches: BTreeMap<u64, u8>,
}

#[derive(Debug, Clone)]
pub struct HexRow {
    pub offset: u64,
    pub bytes: Vec<u8>,
}

impl HexSource {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let f = File::open(&path)?;
        let len = f.metadata()?.len();
        // SAFETY: read-only.
        let mmap = unsafe { Mmap::map(&f)? };
        Ok(Self { path, mmap, len, patches: BTreeMap::new() })
    }

    pub fn total_rows(&self) -> u64 {
        self.len.div_ceil(BYTES_PER_ROW as u64)
    }

    /// Liefert die Zeilen [first_row, first_row + count) — nur dieser Block
    /// wird angefasst; das OS lädt die Pages lazy. Ungespeicherte Patches
    /// werden hier bereits eingeblendet.
    pub fn rows(&self, first_row: u64, count: usize) -> Vec<HexRow> {
        let mut out = Vec::with_capacity(count);
        for r in first_row..first_row + count as u64 {
            let off = r * BYTES_PER_ROW as u64;
            if off >= self.len { break; }
            let end = ((off as usize) + BYTES_PER_ROW).min(self.len as usize);
            let mut bytes = self.mmap[off as usize..end].to_vec();
            for (i, b) in bytes.iter_mut().enumerate() {
                if let Some(&patched) = self.patches.get(&(off + i as u64)) {
                    *b = patched;
                }
            }
            out.push(HexRow { offset: off, bytes });
        }
        out
    }

    /// Merkt eine Byte-Änderung vor (noch nicht auf Platte geschrieben).
    pub fn set_byte(&mut self, offset: u64, value: u8) {
        if offset < self.len {
            self.patches.insert(offset, value);
        }
    }

    pub fn has_pending_changes(&self) -> bool {
        !self.patches.is_empty()
    }

    pub fn pending_change_count(&self) -> usize {
        self.patches.len()
    }

    pub fn discard_changes(&mut self) {
        self.patches.clear();
    }

    /// Schreibt ein Backup der Originaldatei und wendet dann alle
    /// vorgemerkten Byte-Änderungen direkt auf der Platte an (kein
    /// Neu-Mapping der ganzen Datei nötig — nur die geänderten Offsets
    /// werden angefasst). Danach wird die Datei neu gemappt.
    pub fn commit(&mut self, backup_dir: &Path) -> Result<Option<PathBuf>> {
        if self.patches.is_empty() {
            return Ok(None);
        }
        fs::create_dir_all(backup_dir)?;
        let stamp = crate::timestamp();
        let file_name = self
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "datei".into());
        let backup_path = backup_dir.join(format!("{file_name}.{stamp}.bak"));
        fs::copy(&self.path, &backup_path)?;

        let mut f = OpenOptions::new().write(true).open(&self.path)?;
        for (&offset, &value) in &self.patches {
            f.seek(SeekFrom::Start(offset))?;
            f.write_all(&[value])?;
        }
        f.flush()?;
        drop(f);

        let reopened = File::open(&self.path)?;
        // SAFETY: read-only Neu-Mapping nach abgeschlossenem Schreibvorgang.
        self.mmap = unsafe { Mmap::map(&reopened)? };
        self.patches.clear();
        Ok(Some(backup_path))
    }

    /// Sucht ein Bytemuster ab `from` (blockweise, mit Überlappung).
    pub fn find(&self, pattern: &[u8], from: u64) -> Option<u64> {
        if pattern.is_empty() { return None; }
        const CHUNK: usize = 4 << 20; // 4 MiB
        let mut pos = from as usize;
        let n = self.mmap.len();
        while pos < n {
            let end = (pos + CHUNK + pattern.len()).min(n);
            let hay = &self.mmap[pos..end];
            if let Some(i) = hay.windows(pattern.len()).position(|w| w == pattern) {
                return Some((pos + i) as u64);
            }
            pos += CHUNK;
        }
        None
    }
}

fn timestamp() -> String {
    chrono::Local::now().format("%Y%m%d-%H%M%S").to_string()
}

pub fn format_row(row: &HexRow) -> (String, String, String) {
    let offset = format!("{:08X}", row.offset);
    let hex = row.bytes.iter().map(|b| format!("{b:02X}")).collect::<Vec<_>>().join(" ");
    let ascii = row.bytes.iter()
        .map(|&b| if (0x20..0x7F).contains(&b) { b as char } else { '.' })
        .collect();
    (offset, hex, ascii)
}
