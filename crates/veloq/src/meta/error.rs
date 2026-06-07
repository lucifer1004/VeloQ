use std::borrow::Cow;
use thiserror::Error;
use veloq_core::{ErrorCode, VeloqDiagnostic};

pub type MetaResult<T> = Result<T, MetaError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaIoOperation {
    CreateDir,
    CreateFile,
    ReadDir,
    ReadDirEntry,
    ReadFileType,
    RemoveDir,
    CopyFile,
}

impl MetaIoOperation {
    fn code(self) -> ErrorCode {
        match self {
            Self::CreateDir => ErrorCode::new("meta.self-update.io-create-dir"),
            Self::CreateFile => ErrorCode::new("meta.self-update.io-create-file"),
            Self::ReadDir => ErrorCode::new("meta.self-update.io-read-dir"),
            Self::ReadDirEntry => ErrorCode::new("meta.self-update.io-read-dir-entry"),
            Self::ReadFileType => ErrorCode::new("meta.self-update.io-read-file-type"),
            Self::RemoveDir => ErrorCode::new("meta.self-update.io-remove-dir"),
            Self::CopyFile => ErrorCode::new("meta.self-update.io-copy-file"),
        }
    }
}

impl std::fmt::Display for MetaIoOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateDir => f.write_str("create directory"),
            Self::CreateFile => f.write_str("create file"),
            Self::ReadDir => f.write_str("read directory"),
            Self::ReadDirEntry => f.write_str("read directory entry"),
            Self::ReadFileType => f.write_str("read file type"),
            Self::RemoveDir => f.write_str("remove directory"),
            Self::CopyFile => f.write_str("copy file"),
        }
    }
}

#[derive(Debug, Error)]
pub enum MetaError {
    #[error("unknown meta verb `{verb}`")]
    UnknownVerb { verb: String },

    #[error("`<{argument}>` argument is required")]
    MissingArgument { argument: &'static str },

    #[error("failed to serialize meta envelope")]
    SerializeEnvelope { source: serde_json::Error },

    #[error("no recipe with id `{id}` (run `veloq recipes` to list registered ids)")]
    UnknownRecipe { id: String },

    #[error("stat cache root {path}")]
    StatCacheRoot {
        path: String,
        source: std::io::Error,
    },

    #[error("cache root is a symlink; refusing to clean {path}")]
    CacheRootSymlink { path: String },

    #[error("cache root is not a directory; refusing to clean {path}")]
    CacheRootNotDirectory { path: String },

    #[error("removing cache root {path}")]
    RemoveCacheRoot {
        path: String,
        source: std::io::Error,
    },

    #[error("stat artifact {path}")]
    StatArtifact {
        path: String,
        source: std::io::Error,
    },

    #[error("reading {path}")]
    ReadDir {
        path: String,
        source: std::io::Error,
    },

    #[error("reading entry under {path}")]
    ReadDirEntry {
        path: String,
        source: std::io::Error,
    },

    #[error("self-update release lookup could not be configured")]
    SelfUpdateReleaseLookupConfig { source: self_update::errors::Error },

    #[error("self-update releases could not be fetched from GitHub")]
    SelfUpdateReleaseFetch { source: self_update::errors::Error },

    #[error("no releases found on GitHub")]
    SelfUpdateReleaseMissing,

    #[error("binary self-update could not be configured")]
    SelfUpdateBinaryConfig { source: self_update::errors::Error },

    #[error("latest binary could not be downloaded and installed")]
    SelfUpdateBinaryInstall { source: self_update::errors::Error },

    #[error("self-update {area} failed to {operation}: {target}")]
    SelfUpdateIo {
        area: &'static str,
        operation: MetaIoOperation,
        target: String,
        source: std::io::Error,
    },

    #[error("{asset} could not be downloaded from {url}")]
    SelfUpdateSkillsDownload {
        asset: &'static str,
        url: String,
        source: self_update::errors::Error,
    },

    #[error("skills archive could not be extracted: {path}")]
    SelfUpdateSkillsExtract {
        path: String,
        source: self_update::errors::Error,
    },

    #[error("skills archive is missing the expected .claude/skills layout at {path}")]
    SelfUpdateSkillsLayoutMissing { path: String },

    #[error("cannot locate home directory (set HOME, VELOQ_SKILLS_DIR, or --skills-dir)")]
    SelfUpdateHomeMissing,

    #[error("{label} version `{value}` is not valid semver")]
    SelfUpdateVersionParse {
        label: &'static str,
        value: String,
        source: semver::Error,
    },
}

impl MetaError {
    pub fn missing_argument(argument: &'static str) -> Self {
        Self::MissingArgument { argument }
    }

    pub fn self_update_release_lookup_config(source: self_update::errors::Error) -> Self {
        Self::SelfUpdateReleaseLookupConfig { source }
    }

    pub fn self_update_release_fetch(source: self_update::errors::Error) -> Self {
        Self::SelfUpdateReleaseFetch { source }
    }

    pub fn self_update_binary_config(source: self_update::errors::Error) -> Self {
        Self::SelfUpdateBinaryConfig { source }
    }

    pub fn self_update_binary_install(source: self_update::errors::Error) -> Self {
        Self::SelfUpdateBinaryInstall { source }
    }

    fn self_update_io(
        area: &'static str,
        operation: MetaIoOperation,
        target: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::SelfUpdateIo {
            area,
            operation,
            target: target.to_string(),
            source,
        }
    }

    pub fn self_update_skills_temp_dir(source: std::io::Error) -> Self {
        Self::self_update_io(
            "skills archive temp dir",
            MetaIoOperation::CreateDir,
            "temporary directory",
            source,
        )
    }

    pub fn self_update_skills_archive_create(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::self_update_io(
            "temp skills archive",
            MetaIoOperation::CreateFile,
            path,
            source,
        )
    }

    pub fn self_update_skills_download(
        asset: &'static str,
        url: impl Into<String>,
        source: self_update::errors::Error,
    ) -> Self {
        Self::SelfUpdateSkillsDownload {
            asset,
            url: url.into(),
            source,
        }
    }

    pub fn self_update_skills_extract(
        path: impl std::fmt::Display,
        source: self_update::errors::Error,
    ) -> Self {
        Self::SelfUpdateSkillsExtract {
            path: path.to_string(),
            source,
        }
    }

    pub fn self_update_skills_dir_create(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::self_update_io("skills directory", MetaIoOperation::CreateDir, path, source)
    }

    pub fn self_update_skills_staging_read(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::self_update_io(
            "staged skills directory",
            MetaIoOperation::ReadDir,
            path,
            source,
        )
    }

    pub fn self_update_skills_entry_read(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::self_update_io(
            "staged skills directory entry",
            MetaIoOperation::ReadDirEntry,
            path,
            source,
        )
    }

    pub fn self_update_skills_entry_file_type(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::self_update_io(
            "staged skills entry",
            MetaIoOperation::ReadFileType,
            path,
            source,
        )
    }

    pub fn self_update_skill_remove_stale(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::self_update_io(
            "stale skill directory",
            MetaIoOperation::RemoveDir,
            path,
            source,
        )
    }

    pub fn self_update_copy_dir_create(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::self_update_io(
            "skills copy directory",
            MetaIoOperation::CreateDir,
            path,
            source,
        )
    }

    pub fn self_update_copy_dir_read(path: impl std::fmt::Display, source: std::io::Error) -> Self {
        Self::self_update_io(
            "skills copy source directory",
            MetaIoOperation::ReadDir,
            path,
            source,
        )
    }

    pub fn self_update_copy_dir_entry_read(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::self_update_io(
            "skills copy source entry",
            MetaIoOperation::ReadDirEntry,
            path,
            source,
        )
    }

    pub fn self_update_copy_dir_entry_file_type(
        path: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::self_update_io(
            "skills copy source entry",
            MetaIoOperation::ReadFileType,
            path,
            source,
        )
    }

    pub fn self_update_copy_file(
        from: impl std::fmt::Display,
        to: impl std::fmt::Display,
        source: std::io::Error,
    ) -> Self {
        Self::self_update_io(
            "skill file",
            MetaIoOperation::CopyFile,
            format!("{from} -> {to}"),
            source,
        )
    }

    pub fn self_update_current_version_parse(value: &str, source: semver::Error) -> Self {
        Self::SelfUpdateVersionParse {
            label: "current",
            value: value.to_string(),
            source,
        }
    }

    pub fn self_update_latest_version_parse(value: &str, source: semver::Error) -> Self {
        Self::SelfUpdateVersionParse {
            label: "latest",
            value: value.to_string(),
            source,
        }
    }
}

impl VeloqDiagnostic for MetaError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::UnknownVerb { .. } => ErrorCode::new("meta.unknown-verb"),
            Self::MissingArgument { .. } => ErrorCode::new("meta.missing-argument"),
            Self::SerializeEnvelope { .. } => ErrorCode::new("meta.serialize-envelope"),
            Self::UnknownRecipe { .. } => ErrorCode::new("meta.unknown-recipe"),
            Self::StatCacheRoot { .. } | Self::StatArtifact { .. } => ErrorCode::IO_STAT,
            Self::CacheRootSymlink { .. } => ErrorCode::new("meta.clean.symlink-cache-root"),
            Self::CacheRootNotDirectory { .. } => {
                ErrorCode::new("meta.clean.cache-root-not-directory")
            }
            Self::RemoveCacheRoot { .. } => ErrorCode::IO_REMOVE,
            Self::ReadDir { .. } | Self::ReadDirEntry { .. } => ErrorCode::IO_READ,
            Self::SelfUpdateReleaseLookupConfig { .. } => {
                ErrorCode::new("meta.self-update.release-lookup-config")
            }
            Self::SelfUpdateReleaseFetch { .. } => ErrorCode::new("meta.self-update.release-fetch"),
            Self::SelfUpdateReleaseMissing => ErrorCode::new("meta.self-update.release-missing"),
            Self::SelfUpdateBinaryConfig { .. } => ErrorCode::new("meta.self-update.binary-config"),
            Self::SelfUpdateBinaryInstall { .. } => {
                ErrorCode::new("meta.self-update.binary-install")
            }
            Self::SelfUpdateIo { operation, .. } => operation.code(),
            Self::SelfUpdateSkillsDownload { .. } => {
                ErrorCode::new("meta.self-update.skills-download")
            }
            Self::SelfUpdateSkillsExtract { .. } => {
                ErrorCode::new("meta.self-update.skills-extract")
            }
            Self::SelfUpdateSkillsLayoutMissing { .. } => {
                ErrorCode::new("meta.self-update.skills-layout-missing")
            }
            Self::SelfUpdateHomeMissing => ErrorCode::new("meta.self-update.home-missing"),
            Self::SelfUpdateVersionParse { .. } => ErrorCode::new("meta.self-update.version-parse"),
        }
    }

    fn hint(&self) -> Option<Cow<'_, str>> {
        match self {
            Self::UnknownRecipe { .. } => {
                Some(Cow::Borrowed("run `veloq recipes` to list registered ids"))
            }
            Self::MissingArgument { argument } => Some(Cow::Owned(format!(
                "pass the required `<{argument}>` positional argument"
            ))),
            _ => None,
        }
    }
}
