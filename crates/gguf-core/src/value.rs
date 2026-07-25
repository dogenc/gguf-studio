//! Metadaten-Werttypen des GGUF-Formats.

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(untagged)]
pub enum GgufValue {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32),
    U64(u64), I64(i64), F32(f32), F64(f64), Bool(bool),
    Str(String), Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self { Self::Str(s) => Some(s), _ => None }
    }
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Self::U8(v) => Some(v as u64), Self::U16(v) => Some(v as u64),
            Self::U32(v) => Some(v as u64), Self::U64(v) => Some(v),
            Self::I8(v) if v >= 0 => Some(v as u64),
            Self::I16(v) if v >= 0 => Some(v as u64),
            Self::I32(v) if v >= 0 => Some(v as u64),
            Self::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(v as f64),
            Self::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }
    /// Kurzdarstellung für UI-Tabellen (Arrays werden gekürzt).
    pub fn display_short(&self, max: usize) -> String {
        match self {
            Self::Str(s) => {
                if s.chars().count() > max {
                    format!("{}…", s.chars().take(max).collect::<String>())
                } else {
                    s.clone()
                }
            }
            Self::Array(v) => format!("[{} Einträge]", v.len()),
            other => format!("{other:?}"),
        }
    }
}
