//! Path helpers mirroring `packages/daemon-client/src/paths.ts`.

use std::path::PathBuf;

use sha2::{Digest, Sha256};

/// Matches Node `os.userInfo()` scope hashing used by `@rayvan/daemon-client`.
pub fn user_scope_id() -> String {
    let raw = match user_scope_raw() {
        Some(value) => value,
        None => dirs_home_fallback(),
    };
    let digest = Sha256::digest(raw.as_bytes());
    hex::encode(&digest[..6])
}

fn user_scope_raw() -> Option<String> {
    #[cfg(windows)]
    {
        // Node `os.userInfo().uid` is `-1` on Windows.
        let username = std::env::var("USERNAME").ok()?;
        Some(format!("-1:{username}"))
    }
    #[cfg(unix)]
    {
        let username = std::env::var("USER")
            .or_else(|_| std::env::var("LOGNAME"))
            .ok()?;
        let uid = unsafe { libc::getuid() };
        Some(format!("{uid}:{username}"))
    }
    #[cfg(not(any(windows, unix)))]
    {
        None
    }
}

fn dirs_home_fallback() -> String {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".into())
}

pub fn default_rayvan_data_dir() -> PathBuf {
    if let Ok(override_dir) = std::env::var("RAYVAN_DATA_DIR") {
        return PathBuf::from(override_dir);
    }
    if cfg!(windows) {
        let base = std::env::var("APPDATA").unwrap_or_else(|_| {
            let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".into());
            format!("{home}\\AppData\\Roaming")
        });
        return PathBuf::from(base).join("com.rayvan.desktop");
    }
    if cfg!(target_os = "macos") {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        return PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("com.rayvan.desktop");
    }
    let xdg = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        format!("{home}/.local/share")
    });
    PathBuf::from(xdg).join("rayvan")
}

pub fn default_rayvan_runtime_dir() -> PathBuf {
    if let Ok(override_dir) = std::env::var("RAYVAN_RUNTIME_DIR") {
        return PathBuf::from(override_dir);
    }
    if cfg!(windows) {
        let base = std::env::var("LOCALAPPDATA").unwrap_or_else(|_| {
            let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".into());
            format!("{home}\\AppData\\Local")
        });
        return PathBuf::from(base).join("rayvan").join("run");
    }
    if cfg!(target_os = "macos") {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        return PathBuf::from(home)
            .join("Library")
            .join("Caches")
            .join("rayvan")
            .join("run");
    }
    let xdg = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| {
        let tmp = std::env::temp_dir();
        tmp.join(format!("rayvan-{}", user_scope_id()))
            .to_string_lossy()
            .into_owned()
    });
    PathBuf::from(xdg).join("rayvan")
}

pub fn daemon_endpoint_path(runtime_dir: Option<&std::path::Path>) -> String {
    if let Ok(endpoint) = std::env::var("RAYVAN_DAEMON_ENDPOINT") {
        return endpoint;
    }
    let scope = user_scope_id();
    if cfg!(windows) {
        format!(r"\\.\pipe\rayvan-{scope}")
    } else {
        let dir = runtime_dir
            .map(PathBuf::from)
            .unwrap_or_else(default_rayvan_runtime_dir);
        dir.join("rayvand.sock").to_string_lossy().into_owned()
    }
}

pub fn daemon_lock_path(runtime_dir: Option<&std::path::Path>) -> PathBuf {
    let dir = runtime_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_rayvan_runtime_dir);
    dir.join("rayvand.lock")
}

pub fn daemon_pid_path(runtime_dir: Option<&std::path::Path>) -> PathBuf {
    let dir = runtime_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_rayvan_runtime_dir);
    dir.join("rayvand.pid")
}

pub fn daemon_credential_store_path(data_dir: Option<&std::path::Path>) -> PathBuf {
    let dir = data_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_rayvan_data_dir);
    dir.join("credentials").join("local-clients.json")
}

pub fn database_path(data_dir: Option<&std::path::Path>) -> PathBuf {
    let dir = data_dir
        .map(PathBuf::from)
        .unwrap_or_else(default_rayvan_data_dir);
    dir.join("rayvan.db")
}
