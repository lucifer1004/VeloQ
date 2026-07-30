use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(unix)]
use interprocess::local_socket::GenericFilePath;
#[cfg(windows)]
use interprocess::local_socket::GenericNamespaced;
use interprocess::local_socket::{Name, prelude::*};
use serde::{Deserialize, Serialize};
use sysinfo::{Pid, ProcessesToUpdate, System};

use super::config::DaemonLimits;
use super::protocol::PROTOCOL_VERSION;
use super::{DaemonError, DaemonResult};

const OWNER_FILE: &str = "owner-v1.json";
const READY_FILE_PREFIX: &str = ".owner-ready-v1-";
const STOPPING_FILE_PREFIX: &str = ".owner-stopping-v1-";
const SOCKET_FILE_PREFIX: &str = "daemon-v1-";
const MIN_OWNER_TOKEN_BYTES: usize = 32;
const MAX_OWNER_TOKEN_BYTES: usize = 128;

#[derive(Debug, Clone)]
pub struct RuntimePaths {
    pub root: PathBuf,
    pub owner: PathBuf,
}

impl RuntimePaths {
    pub fn discover() -> DaemonResult<Self> {
        let root = runtime_root()?;
        ensure_private_dir(&root)?;
        Ok(Self {
            owner: root.join(OWNER_FILE),
            root,
        })
    }

    pub fn socket_path(&self, owner_token: &str) -> DaemonResult<PathBuf> {
        validate_owner_token(owner_token)?;
        Ok(self
            .root
            .join(format!("{SOCKET_FILE_PREFIX}{owner_token}.sock")))
    }

    pub fn socket_name(&self, owner_token: &str) -> DaemonResult<Name<'static>> {
        validate_owner_token(owner_token)?;
        #[cfg(unix)]
        {
            let socket = self.socket_path(owner_token)?;
            socket
                .as_os_str()
                .to_fs_name::<GenericFilePath>()
                .map(Name::into_owned)
                .map_err(|source| {
                    DaemonError::lifecycle(format!(
                        "cannot map local daemon endpoint {}: {source}",
                        socket.display()
                    ))
                })
        }
        #[cfg(windows)]
        {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            self.root.hash(&mut hasher);
            owner_token.hash(&mut hasher);
            let name = format!("veloq-daemon-{:016x}", hasher.finish());
            name.to_ns_name::<GenericNamespaced>()
                .map(Name::into_owned)
                .map_err(|source| {
                    DaemonError::lifecycle(format!(
                        "cannot map current-user daemon endpoint: {source}"
                    ))
                })
        }
        #[cfg(not(any(unix, windows)))]
        Err(DaemonError::lifecycle(
            "this operating system has no supported local daemon endpoint",
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OwnerPhase {
    Starting,
    Ready,
    Stopping,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OwnerRecord {
    pub owner_format: u32,
    pub token: String,
    pub phase: OwnerPhase,
    pub process_id: u32,
    pub process_start_time: u64,
    pub veloq_version: String,
    pub protocol_version: String,
    pub limits: DaemonLimits,
}

impl OwnerRecord {
    pub fn starting(token: String, limits: DaemonLimits) -> DaemonResult<Self> {
        Ok(Self {
            owner_format: 1,
            token,
            phase: OwnerPhase::Starting,
            process_id: std::process::id(),
            process_start_time: current_process_start_time()?,
            veloq_version: env!("CARGO_PKG_VERSION").to_string(),
            protocol_version: PROTOCOL_VERSION.to_string(),
            limits,
        })
    }

    pub fn for_daemon(mut self) -> DaemonResult<Self> {
        self.phase = OwnerPhase::Ready;
        self.process_id = std::process::id();
        self.process_start_time = current_process_start_time()?;
        Ok(self)
    }
}

pub fn create_owner(paths: &RuntimePaths, record: &OwnerRecord) -> DaemonResult<()> {
    validate_owner_parent(paths)?;
    validate_owner_token(&record.token)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&paths.owner).map_err(|source| {
        if source.kind() == io::ErrorKind::AlreadyExists {
            DaemonError::owner_exists()
        } else {
            DaemonError::lifecycle(format!(
                "cannot acquire daemon singleton ownership: {source}"
            ))
        }
    })?;
    write_record(&mut file, record)?;
    Ok(())
}

pub fn read_owner(paths: &RuntimePaths) -> DaemonResult<Option<OwnerRecord>> {
    validate_owner_parent(paths)?;
    let Some(claim) = read_record(&paths.owner, "daemon owner record")? else {
        return Ok(None);
    };
    validate_owner_record(&claim)?;

    for phase in [OwnerPhase::Stopping, OwnerPhase::Ready] {
        let state_path = owner_state_path(paths, &claim.token, phase)?;
        let Some(state) = read_record(&state_path, "daemon owner state")? else {
            continue;
        };
        validate_owner_record(&state)?;
        if state.token != claim.token || state.phase != phase {
            return Err(DaemonError::lifecycle(
                "daemon owner state does not match its immutable ownership claim",
            ));
        }
        return Ok(Some(state));
    }
    Ok(Some(claim))
}

fn read_claim(paths: &RuntimePaths) -> DaemonResult<Option<OwnerRecord>> {
    validate_owner_parent(paths)?;
    let Some(claim) = read_record(&paths.owner, "daemon owner record")? else {
        return Ok(None);
    };
    validate_owner_record(&claim)?;
    Ok(Some(claim))
}

fn read_record(path: &Path, description: &str) -> DaemonResult<Option<OwnerRecord>> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(source) if source.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(DaemonError::lifecycle(format!(
                "cannot inspect {description}: {source}"
            )));
        }
    };
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(DaemonError::lifecycle(format!(
            "{description} is not a regular current-user file"
        )));
    }
    validate_private_file(&metadata)?;
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|source| DaemonError::lifecycle(format!("cannot read {description}: {source}")))?;
    let record: OwnerRecord = serde_json::from_slice(&bytes)
        .map_err(|source| DaemonError::lifecycle(format!("{description} is invalid: {source}")))?;
    Ok(Some(record))
}

fn validate_owner_record(record: &OwnerRecord) -> DaemonResult<()> {
    if record.owner_format != 1 {
        return Err(DaemonError::lifecycle(
            "daemon owner record has an unsupported ownership format",
        ));
    }
    validate_owner_token(&record.token)
}

pub fn replace_owner(
    paths: &RuntimePaths,
    expected_token: &str,
    record: &OwnerRecord,
) -> DaemonResult<()> {
    validate_owner_token(expected_token)?;
    validate_owner_record(record)?;
    if record.token != expected_token {
        return Err(DaemonError::lifecycle(
            "daemon owner state token does not match the ownership claim",
        ));
    }
    if record.phase == OwnerPhase::Starting {
        return Err(DaemonError::lifecycle(
            "daemon starting state belongs in the immutable ownership claim",
        ));
    }
    let current = read_claim(paths)?.ok_or_else(|| {
        DaemonError::lifecycle("daemon singleton ownership disappeared during transition")
    })?;
    if current.token != expected_token {
        return Err(DaemonError::lifecycle(
            "daemon singleton ownership changed during transition",
        ));
    }
    let state = owner_state_path(paths, expected_token, record.phase)?;
    let temp = paths.root.join(format!(
        ".owner-state-v1-{}-{}.tmp",
        record.token,
        std::process::id()
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(&temp).map_err(|source| {
        DaemonError::lifecycle(format!("cannot stage daemon owner record: {source}"))
    })?;
    if let Err(error) = write_record(&mut file, record) {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    fs::rename(&temp, &state).map_err(|source| {
        let _ = fs::remove_file(&temp);
        DaemonError::lifecycle(format!("cannot publish daemon owner record: {source}"))
    })?;
    let published = read_claim(paths)?.ok_or_else(|| {
        DaemonError::lifecycle("daemon singleton ownership disappeared during transition")
    })?;
    if published.token != expected_token {
        let _ = fs::remove_file(&state);
        return Err(DaemonError::lifecycle(
            "daemon singleton ownership changed during transition",
        ));
    }
    Ok(())
}

pub fn remove_owner(paths: &RuntimePaths, expected_token: &str) -> DaemonResult<()> {
    validate_owner_token(expected_token)?;
    let Some(current) = read_claim(paths)? else {
        return Ok(());
    };
    if current.token != expected_token {
        return Err(DaemonError::lifecycle(
            "refusing to remove daemon ownership that changed identity",
        ));
    }
    fs::remove_file(&paths.owner).map_err(|source| {
        DaemonError::lifecycle(format!(
            "cannot release daemon singleton ownership: {source}"
        ))
    })?;
    for phase in [OwnerPhase::Ready, OwnerPhase::Stopping] {
        let state = owner_state_path(paths, expected_token, phase)?;
        match fs::remove_file(state) {
            Ok(()) => {}
            Err(source) if source.kind() == io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(DaemonError::lifecycle(format!(
                    "cannot remove released daemon owner state: {source}"
                )));
            }
        }
    }
    Ok(())
}

pub fn remove_stale_endpoint(paths: &RuntimePaths, owner_token: &str) -> DaemonResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileTypeExt;
        let socket = paths.socket_path(owner_token)?;
        match fs::symlink_metadata(&socket) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.file_type().is_socket() {
                    return Err(DaemonError::lifecycle(
                        "refusing to remove a daemon endpoint that is not a socket",
                    ));
                }
                validate_private_file(&metadata)?;
                fs::remove_file(&socket).map_err(|source| {
                    DaemonError::lifecycle(format!(
                        "cannot remove safely identified stale daemon endpoint: {source}"
                    ))
                })?;
            }
            Err(source) if source.kind() == io::ErrorKind::NotFound => {}
            Err(source) => {
                return Err(DaemonError::lifecycle(format!(
                    "cannot inspect daemon endpoint: {source}"
                )));
            }
        }
    }
    #[cfg(not(unix))]
    let _ = (paths, owner_token);
    Ok(())
}

pub fn process_matches(record: &OwnerRecord) -> bool {
    let pid = Pid::from_u32(record.process_id);
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    system
        .process(pid)
        .is_some_and(|process| process.start_time() == record.process_start_time)
}

pub fn new_owner_token() -> DaemonResult<String> {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|source| {
            DaemonError::lifecycle(format!("cannot create daemon ownership token: {source}"))
        })?;
    Ok(format!(
        "{:08x}{:016x}{:08x}",
        std::process::id(),
        elapsed.as_nanos(),
        current_process_start_time()?
    ))
}

fn current_process_start_time() -> DaemonResult<u64> {
    let pid = Pid::from_u32(std::process::id());
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    system
        .process(pid)
        .map(|process| process.start_time())
        .ok_or_else(|| DaemonError::lifecycle("cannot establish current process identity"))
}

fn write_record(file: &mut File, record: &OwnerRecord) -> DaemonResult<()> {
    serde_json::to_writer(&mut *file, record).map_err(|source| {
        DaemonError::lifecycle(format!("cannot serialize daemon owner record: {source}"))
    })?;
    file.write_all(b"\n")
        .and_then(|_| file.sync_all())
        .map_err(|source| {
            DaemonError::lifecycle(format!("cannot persist daemon owner record: {source}"))
        })
}

fn validate_owner_parent(paths: &RuntimePaths) -> DaemonResult<()> {
    ensure_private_dir(&paths.root)
}

#[cfg(unix)]
fn runtime_root() -> DaemonResult<PathBuf> {
    let uid = unsafe { libc::geteuid() };
    if let Some(base) = env::var_os("XDG_RUNTIME_DIR") {
        let base = PathBuf::from(base);
        validate_existing_private_dir(&base)?;
        return Ok(base.join("veloq"));
    }
    Ok(PathBuf::from(format!("/tmp/veloq-{uid}")))
}

#[cfg(windows)]
fn runtime_root() -> DaemonResult<PathBuf> {
    env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .map(|base| base.join("veloq").join("runtime"))
        .ok_or_else(|| {
            DaemonError::lifecycle(
                "cannot locate the current-user local application data directory",
            )
        })
}

#[cfg(not(any(unix, windows)))]
fn runtime_root() -> DaemonResult<PathBuf> {
    Err(DaemonError::lifecycle(
        "this operating system has no supported local daemon runtime directory",
    ))
}

fn ensure_private_dir(path: &Path) -> DaemonResult<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => return validate_existing_private_dir(path),
        Err(source) if source.kind() == io::ErrorKind::NotFound => {}
        Err(source) => {
            return Err(DaemonError::lifecycle(format!(
                "cannot inspect current-user daemon runtime directory: {source}"
            )));
        }
    }
    if let Err(source) = fs::create_dir_all(path) {
        return Err(DaemonError::lifecycle(format!(
            "cannot create current-user daemon runtime directory: {source}"
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|source| {
            DaemonError::lifecycle(format!(
                "cannot restrict daemon runtime directory permissions: {source}"
            ))
        })?;
    }
    validate_existing_private_dir(path)
}

fn owner_state_path(
    paths: &RuntimePaths,
    owner_token: &str,
    phase: OwnerPhase,
) -> DaemonResult<PathBuf> {
    validate_owner_token(owner_token)?;
    let prefix = match phase {
        OwnerPhase::Ready => READY_FILE_PREFIX,
        OwnerPhase::Stopping => STOPPING_FILE_PREFIX,
        OwnerPhase::Starting => {
            return Err(DaemonError::lifecycle(
                "daemon starting state has no mutable owner-state path",
            ));
        }
    };
    Ok(paths.root.join(format!("{prefix}{owner_token}.json")))
}

fn validate_owner_token(token: &str) -> DaemonResult<()> {
    if !(MIN_OWNER_TOKEN_BYTES..=MAX_OWNER_TOKEN_BYTES).contains(&token.len())
        || !token
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(DaemonError::lifecycle(
            "daemon ownership token has an invalid format",
        ));
    }
    Ok(())
}

fn validate_existing_private_dir(path: &Path) -> DaemonResult<()> {
    let metadata = fs::symlink_metadata(path).map_err(|source| {
        DaemonError::lifecycle(format!(
            "cannot inspect current-user daemon runtime directory: {source}"
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(DaemonError::lifecycle(
            "daemon runtime path is not a private directory",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let uid = unsafe { libc::geteuid() };
        if metadata.uid() != uid || metadata.permissions().mode() & 0o077 != 0 {
            return Err(DaemonError::lifecycle(
                "daemon runtime directory is not owned and restricted to the current user",
            ));
        }
    }
    Ok(())
}

fn validate_private_file(metadata: &fs::Metadata) -> DaemonResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let uid = unsafe { libc::geteuid() };
        if metadata.uid() != uid || metadata.permissions().mode() & 0o077 != 0 {
            return Err(DaemonError::lifecycle(
                "daemon state is not owned and restricted to the current user",
            ));
        }
    }
    Ok(())
}
