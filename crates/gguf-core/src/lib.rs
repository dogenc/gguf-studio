//! # gguf-core
//! Memory-mapped, lazy GGUF parser. Es wird niemals die gesamte Datei in den
//! RAM geladen: der Header und die Metadaten werden über ein `Mmap`-Fenster
//! gelesen, Tensordaten bleiben auf der Platte und werden nur blockweise
//! über [`GgufFile::tensor_bytes`] angefasst (das OS pagt on demand).

pub mod dequant;
pub mod diff;
pub mod quant;
pub mod tokenizer;
pub mod value;

use byteorder::{ByteOrder, LittleEndian as LE};
use memmap2::Mmap;
use std::{collections::BTreeMap, fs::File, path::Path};
use thiserror::Error;
use value::GgufValue;

pub const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF"

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("Ungültige Magic-Number: {0:#x}")]
    BadMagic(u32),
    #[error("Nicht unterstützte GGUF-Version: {0}")]
    BadVersion(u32),
    #[error("Datei zu kurz / beschädigt bei Offset {0}")]
    Truncated(usize),
    #[error("Unbekannter Werttyp {0}")]
    BadValueType(u32),
    #[error("Unbekannter Tensortyp {0}")]
    BadTensorType(u32),
    #[error("Ungültiger UTF-8-String bei Offset {0}")]
    BadUtf8(usize),
}

pub type Result<T> = std::result::Result<T, GgufError>;

/// GGML-Tensordatentypen (ggml_type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum GgmlType {
    F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
    Q2K, Q3K, Q4K, Q5K, Q6K, Q8K,
    IQ2XXS, IQ2XS, IQ3XXS, IQ1S, IQ4NL, IQ3S, IQ2S, IQ4XS,
    I8, I16, I32, I64, F64, IQ1M, BF16,
    Unknown(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        use GgmlType::*;
        match v {
            0 => F32, 1 => F16, 2 => Q4_0, 3 => Q4_1, 6 => Q5_0, 7 => Q5_1,
            8 => Q8_0, 9 => Q8_1, 10 => Q2K, 11 => Q3K, 12 => Q4K, 13 => Q5K,
            14 => Q6K, 15 => Q8K, 16 => IQ2XXS, 17 => IQ2XS, 18 => IQ3XXS,
            19 => IQ1S, 20 => IQ4NL, 21 => IQ3S, 22 => IQ2S, 23 => IQ4XS,
            24 => I8, 25 => I16, 26 => I32, 27 => I64, 28 => F64, 29 => IQ1M,
            30 => BF16,
            other => Unknown(other),
        }
    }

    /// (Blockgröße in Elementen, Bytes pro Block)
    pub fn block_layout(self) -> (usize, usize) {
        use GgmlType::*;
        match self {
            F32 => (1, 4), F16 | BF16 => (1, 2), F64 => (1, 8),
            I8 => (1, 1), I16 => (1, 2), I32 => (1, 4), I64 => (1, 8),
            Q4_0 => (32, 18), Q4_1 => (32, 20), Q5_0 => (32, 22), Q5_1 => (32, 24),
            Q8_0 => (32, 34), Q8_1 => (32, 36),
            Q2K => (256, 84), Q3K => (256, 110), Q4K => (256, 144),
            Q5K => (256, 176), Q6K => (256, 210), Q8K => (256, 292),
            IQ2XXS => (256, 66), IQ2XS => (256, 74), IQ3XXS => (256, 98),
            IQ1S => (256, 50), IQ4NL => (32, 18), IQ3S => (256, 110),
            IQ2S => (256, 82), IQ4XS => (256, 136), IQ1M => (256, 56),
            Unknown(_) => (1, 1),
        }
    }

    pub fn bits_per_weight(self) -> f64 {
        let (n, b) = self.block_layout();
        (b as f64 * 8.0) / n as f64
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    /// Offset relativ zum Beginn des Datenbereichs.
    pub offset: u64,
    /// Berechnete Größe in Bytes.
    pub size_bytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product::<u64>().max(1)
    }
}

/// Geöffnete GGUF-Datei. Hält nur das Mapping — kein Kopieren der Daten.
pub struct GgufFile {
    pub path: std::path::PathBuf,
    pub version: u32,
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    pub data_offset: u64,
    pub file_size: u64,
    mmap: Mmap,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Truncated(self.pos))?;
        if end > self.buf.len() {
            return Err(GgufError::Truncated(self.pos));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> Result<u16> { Ok(LE::read_u16(self.take(2)?)) }
    fn u32(&mut self) -> Result<u32> { Ok(LE::read_u32(self.take(4)?)) }
    fn u64(&mut self) -> Result<u64> { Ok(LE::read_u64(self.take(8)?)) }
    fn i8(&mut self) -> Result<i8> { Ok(self.u8()? as i8) }
    fn i16(&mut self) -> Result<i16> { Ok(self.u16()? as i16) }
    fn i32(&mut self) -> Result<i32> { Ok(self.u32()? as i32) }
    fn i64(&mut self) -> Result<i64> { Ok(self.u64()? as i64) }
    fn f32(&mut self) -> Result<f32> { Ok(LE::read_f32(self.take(4)?)) }
    fn f64(&mut self) -> Result<f64> { Ok(LE::read_f64(self.take(8)?)) }
    fn string(&mut self) -> Result<String> {
        let len = self.u64()? as usize;
        let start = self.pos;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::BadUtf8(start))
    }
    fn value(&mut self, ty: u32) -> Result<GgufValue> {
        use GgufValue::*;
        Ok(match ty {
            0 => U8(self.u8()?),
            1 => I8(self.i8()?),
            2 => U16(self.u16()?),
            3 => I16(self.i16()?),
            4 => U32(self.u32()?),
            5 => I32(self.i32()?),
            6 => F32(self.f32()?),
            7 => Bool(self.u8()? != 0),
            8 => Str(self.string()?),
            9 => {
                let elem_ty = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(elem_ty)?);
                }
                Array(v)
            }
            10 => U64(self.u64()?),
            11 => I64(self.i64()?),
            12 => F64(self.f64()?),
            other => return Err(GgufError::BadValueType(other)),
        })
    }
}

impl GgufFile {
    /// Öffnet und parst Header/Metadaten/Tensortabelle. Tensordaten werden
    /// nicht gelesen — nur gemappt.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let file_size = file.metadata()?.len();
        // SAFETY: read-only Mapping; Datei wird nicht gleichzeitig geschrieben.
        let mmap = unsafe { Mmap::map(&file)? };

        let mut c = Cursor { buf: &mmap, pos: 0 };
        let magic = c.u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = c.u32()?;
        if !(1..=3).contains(&version) {
            return Err(GgufError::BadVersion(version));
        }
        let n_tensors = c.u64()?;
        let n_kv = c.u64()?;

        let mut metadata = BTreeMap::new();
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            metadata.insert(key, c.value(ty)?);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(GgufValue::as_u64)
            .unwrap_or(32);

        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(c.u64()?);
            }
            let dtype = GgmlType::from_u32(c.u32()?);
            let offset = c.u64()?;
            let n_elem: u64 = shape.iter().product::<u64>().max(1);
            let (bn, bb) = dtype.block_layout();
            let size_bytes = n_elem.div_ceil(bn as u64) * bb as u64;
            tensors.push(TensorInfo { name, shape, dtype, offset, size_bytes });
        }

        let data_offset = (c.pos as u64).div_ceil(alignment) * alignment;

        Ok(Self { path, version, metadata, tensors, alignment, data_offset, file_size, mmap })
    }

    /// Rohbytes eines Tensor-Ausschnitts — Zero-Copy-Slice ins Mapping.
    pub fn tensor_bytes(&self, t: &TensorInfo, rel_offset: u64, len: usize) -> Result<&[u8]> {
        let start = (self.data_offset + t.offset + rel_offset) as usize;
        let len = len.min(t.size_bytes.saturating_sub(rel_offset) as usize);
        self.mmap
            .get(start..start + len)
            .ok_or(GgufError::Truncated(start))
    }

    /// Beliebiger Dateiausschnitt (für Hex-Viewer). Zero-Copy.
    pub fn raw_bytes(&self, offset: u64, len: usize) -> &[u8] {
        let start = (offset as usize).min(self.mmap.len());
        let end = (start + len).min(self.mmap.len());
        &self.mmap[start..end]
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(GgufValue::as_str)
    }

    pub fn param_count(&self) -> u64 {
        self.tensors.iter().map(TensorInfo::n_elements).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_magic() {
        // Cursor direkt testen
        let buf = [0u8; 8];
        let mut c = Cursor { buf: &buf, pos: 0 };
        assert_eq!(c.u32().unwrap(), 0);
    }

    #[test]
    fn block_layouts_sane() {
        assert_eq!(GgmlType::F32.block_layout(), (1, 4));
        assert!((GgmlType::Q4K.bits_per_weight() - 4.5).abs() < 0.1);
    }
}
