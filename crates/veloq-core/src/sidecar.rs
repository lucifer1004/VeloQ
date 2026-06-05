//! Versioned bincode sidecar cache inside a report artifact directory.
//!
//! Multiple veloq subsystems cache derived state under the report's
//! `<report>.veloq/` root. They all want the same primitive:
//! serialize a typed payload with a `u32` version header and a
//! `(mtime, size)` fingerprint of the source file, decode lazily on
//! warm calls, and rebuild on any of (version mismatch, source
//! changed, decode failure).
//!
//! [`SidecarCache<T>`] is that primitive. Each call site picks the
//! path suffix, the version constant, the human-readable label that
//! appears in `log::info!` lines, and a payload type that derives
//! `serde::{Serialize, DeserializeOwned}`. On-disk layout:
//!
//! ```text
//! [u32 version][i64 source_mtime_secs][u64 source_size][bincode payload]
//! ```
//!
//!

//! Caches with bespoke on-disk formats (Parquet's TOML manifest, NCU's
//! JSON disasm cache) stay on their own; this helper is for the
//! "bincode blob with a version byte" case only.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::fs;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// File-system fingerprint of the source artifact a sidecar covers.
/// Captures the two facts every cache invalidation depends on:
/// modification time (seconds since the Unix epoch) and size.
///
/// `mtime_secs == 0` on platforms where `Metadata::modified()` errors
/// out — the fallback is documented at the call site rather than
/// silently disabling invalidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceFingerprint {
    pub mtime_secs: i64,
    pub size: u64,
}

impl SourceFingerprint {
    /// Read the fingerprint from `source`'s filesystem metadata.
    pub fn of_path(source: &Path) -> std::io::Result<Self> {
        let meta = fs::metadata(source)?;
        Ok(Self::of_metadata(&meta))
    }

    /// Build from already-read metadata (avoids a second `stat` when
    /// the caller already has it on hand).
    pub fn of_metadata(meta: &fs::Metadata) -> Self {
        let mtime_secs = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        Self {
            mtime_secs,
            size: meta.len(),
        }
    }
}

/// Sidecar payload reader/writer for type `T`.
///
/// Construct once with the on-disk path, a version constant, and a
/// label (the noun that appears in `"<label> version mismatch …"` /
/// `"wrote <label>: …"` log lines). [`try_load`] checks the version
/// header and source fingerprint before decoding; [`write`] serializes
/// then atomically renames the result into place.
///
/// [`try_load`]: SidecarCache::try_load
/// [`write`]: SidecarCache::write
pub struct SidecarCache<T> {
    path: PathBuf,
    version: u32,
    label: &'static str,
    _phantom: PhantomData<fn() -> T>,
}

impl<T> SidecarCache<T> {
    pub fn new(path: PathBuf, version: u32, label: &'static str) -> Self {
        Self {
            path,
            version,
            label,
            _phantom: PhantomData,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// On-disk header peek: format version + source fingerprint, without
/// decoding the payload. Returned by [`SidecarCache::read_header`].
///
/// Useful for inspection verbs (`veloq prep --status`) that want to
/// report the on-disk format version next to the current expected
/// one, including for caches that fail fingerprint match — `try_load`
/// folds those down to `Ok(None)` so the version becomes invisible.
#[derive(Debug, Clone, Copy)]
pub struct SidecarHeader {
    pub version: u32,
    pub fingerprint: SourceFingerprint,
}

impl<T> SidecarCache<T> {
    /// Peek the on-disk header (version + source fingerprint) without
    /// reading the payload. `Ok(None)` for missing files; decode
    /// errors propagate so a corrupt sidecar is visible rather than
    /// silently treated as absent.
    pub fn read_header(&self) -> Result<Option<SidecarHeader>> {
        if !self.path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&self.path)
            .with_context(|| format!("reading {} at {}", self.label, self.path.display()))?;
        // The header struct mirrors the leading three fields of
        // `CacheFile<T>` exactly. bincode's positional encoding means
        // decoding just the header from the start of the buffer
        // works — `decode_from_slice` doesn't require full
        // consumption of the input.
        #[derive(serde::Deserialize)]
        struct HeaderOnly {
            version: u32,
            source_mtime_secs: i64,
            source_size: u64,
        }
        let (h, _read): (HeaderOnly, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .with_context(|| format!("decoding {} header", self.label))?;
        Ok(Some(SidecarHeader {
            version: h.version,
            fingerprint: SourceFingerprint {
                mtime_secs: h.source_mtime_secs,
                size: h.source_size,
            },
        }))
    }
}

impl<T: DeserializeOwned> SidecarCache<T> {
    /// Decode the sidecar if it exists, matches the configured
    /// version, and matches `source_fp`. Returns `Ok(None)` for any
    /// "skip + rebuild" condition (missing, version mismatch, source
    /// changed) with an info-level log line explaining which check
    /// failed. Decode/I/O errors propagate as `Err` so the caller
    /// can decide whether to rebuild or surface.
    pub fn try_load(&self, source_fp: SourceFingerprint) -> Result<Option<T>> {
        if !self.path.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&self.path)
            .with_context(|| format!("reading {} at {}", self.label, self.path.display()))?;
        let (file, _read): (CacheFile<T>, _) =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .with_context(|| format!("decoding {}", self.label))?;
        if file.version != self.version {
            log::info!(
                "{} version mismatch ({} vs {}); rebuilding",
                self.label,
                file.version,
                self.version
            );
            return Ok(None);
        }
        if file.source_mtime_secs != source_fp.mtime_secs || file.source_size != source_fp.size {
            log::info!(
                "trace file changed since {} was written; rebuilding",
                self.label
            );
            return Ok(None);
        }
        Ok(Some(file.payload))
    }
}

impl<T: Serialize> SidecarCache<T> {
    /// Encode `payload` and atomically replace the sidecar. The on-disk
    /// header records `source_fp`, which `try_load` will match against
    /// the *current* fingerprint of the source file on the next open.
    ///
    /// Writes via a `<path>.tmp` sibling + `rename(2)` so a crashed
    /// write never leaves a half-corrupt sidecar in place.
    pub fn write(&self, source_fp: SourceFingerprint, payload: &T) -> Result<()> {
        let file = CacheFileRef {
            version: self.version,
            source_mtime_secs: source_fp.mtime_secs,
            source_size: source_fp.size,
            payload,
        };
        let bytes = bincode::serde::encode_to_vec(&file, bincode::config::standard())
            .with_context(|| format!("encoding {}", self.label))?;
        let mut tmp = self.path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp_path = PathBuf::from(tmp);
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "creating {} parent directory at {}",
                    self.label,
                    parent.display()
                )
            })?;
        }
        fs::write(&tmp_path, &bytes).with_context(|| {
            format!("writing {} temp file at {}", self.label, tmp_path.display())
        })?;
        fs::rename(&tmp_path, &self.path).with_context(|| {
            format!(
                "renaming {} temp into place at {}",
                self.label,
                self.path.display()
            )
        })?;
        log::info!(
            "wrote {}: {} bytes → {}",
            self.label,
            bytes.len(),
            self.path.display()
        );
        Ok(())
    }
}

/// Owned on-disk shape used during decode. Field names are
/// load-bearing because bincode uses serde and serde tags field
/// indices on `T: Serialize` derived structs; renaming would break
/// compatibility with caches written before this helper landed.
#[derive(Serialize, Deserialize)]
struct CacheFile<T> {
    version: u32,
    source_mtime_secs: i64,
    source_size: u64,
    payload: T,
}

/// Borrowed-payload variant for write. Bincode-with-standard-config
/// produces byte-identical output to [`CacheFile<T>`] because the
/// fields are positional + serialized in declaration order. Lets the
/// caller pass `&T` without cloning.
#[derive(Serialize)]
struct CacheFileRef<'a, T> {
    version: u32,
    source_mtime_secs: i64,
    source_size: u64,
    payload: &'a T,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use std::fs;
    use std::path::PathBuf;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Demo {
        n: u32,
        s: String,
    }

    fn tmpdir() -> Result<PathBuf> {
        let d = std::env::temp_dir().join(format!(
            "veloq-sidecar-test-{}",
            std::process::id() as u64 * 1_000_000
                + std::time::SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0)
        ));
        fs::create_dir_all(&d).with_context(|| format!("creating tmp dir {}", d.display()))?;
        Ok(d)
    }

    fn fp(mtime: i64, size: u64) -> SourceFingerprint {
        SourceFingerprint {
            mtime_secs: mtime,
            size,
        }
    }

    #[test]
    fn round_trip_load_returns_written_payload() -> Result<()> {
        let dir = tmpdir()?;
        let path = dir.join("demo.cache");
        let cache: SidecarCache<Demo> = SidecarCache::new(path, 7, "demo cache");
        let payload = Demo {
            n: 42,
            s: "hello".into(),
        };
        cache.write(fp(1234, 999), &payload)?;
        let back = cache
            .try_load(fp(1234, 999))?
            .ok_or_else(|| anyhow::anyhow!("just-written cache should load"))?;
        assert_eq!(back, payload);
        Ok(())
    }

    #[test]
    fn try_load_missing_returns_none() -> Result<()> {
        let dir = tmpdir()?;
        let path = dir.join("does-not-exist.cache");
        let cache: SidecarCache<Demo> = SidecarCache::new(path, 1, "demo cache");
        assert!(cache.try_load(fp(0, 0))?.is_none());
        Ok(())
    }

    #[test]
    fn version_mismatch_rebuilds() -> Result<()> {
        let dir = tmpdir()?;
        let path = dir.join("demo.cache");
        let writer: SidecarCache<Demo> = SidecarCache::new(path.clone(), 1, "demo cache");
        writer.write(
            fp(1, 1),
            &Demo {
                n: 1,
                s: "x".into(),
            },
        )?;
        let reader: SidecarCache<Demo> = SidecarCache::new(path, 2, "demo cache");
        assert!(reader.try_load(fp(1, 1))?.is_none());
        Ok(())
    }

    #[test]
    fn source_changed_rebuilds() -> Result<()> {
        let dir = tmpdir()?;
        let path = dir.join("demo.cache");
        let cache: SidecarCache<Demo> = SidecarCache::new(path, 1, "demo cache");
        cache.write(
            fp(1, 100),
            &Demo {
                n: 1,
                s: "x".into(),
            },
        )?;
        assert!(cache.try_load(fp(2, 100))?.is_none(), "mtime change");
        assert!(cache.try_load(fp(1, 200))?.is_none(), "size change");
        assert!(cache.try_load(fp(1, 100))?.is_some(), "matching fp");
        Ok(())
    }

    #[test]
    fn fingerprint_from_real_file_round_trips() -> Result<()> {
        let dir = tmpdir()?;
        let src = dir.join("source.bin");
        fs::write(&src, b"hello world")?;
        let fp1 = SourceFingerprint::of_path(&src)?;
        assert_eq!(fp1.size, 11);
        // mtime_secs varies — just check it isn't an obvious sentinel.
        assert!(fp1.mtime_secs > 1_000_000_000);
        Ok(())
    }
}
