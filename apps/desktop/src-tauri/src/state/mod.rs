use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rayvan_daemon_ipc::{
    daemon_endpoint_path, event_channel, wait_for_desktop_credential, DaemonConnection,
    DaemonConnectionError, DaemonStatusSnapshot, HandshakeResponse,
};
use serde_json::Value;
use tauri::{AppHandle, Emitter};
#[cfg(not(debug_assertions))]
use tauri_plugin_shell::ShellExt;

use crate::prefs::SessionPrefs;

pub struct AppState {
    pub daemon: Arc<Mutex<DaemonSession>>,
    pub prefs: Arc<Mutex<SessionPrefs>>,
}

pub struct DaemonSession {
    pub connection: Option<Arc<DaemonConnection>>,
    pub handshake: Option<HandshakeResponse>,
    pub endpoint: String,
    pub spawned: bool,
    pub last_error: Option<String>,
}

impl DaemonSession {
    pub fn status_snapshot(&self) -> DaemonStatusSnapshot {
        DaemonStatusSnapshot {
            connected: self.connection.is_some(),
            endpoint: self.endpoint.clone(),
            spawned: self.spawned,
            session_id: self.handshake.as_ref().map(|h| h.session_id.clone()),
            daemon_version: self.handshake.as_ref().map(|h| h.daemon_version.clone()),
            authenticated_client_id: self
                .handshake
                .as_ref()
                .and_then(|h| h.authenticated_client_id.clone()),
            last_error: self.last_error.clone(),
            status: None,
        }
    }
}

pub fn initialize_app_state(app: &AppHandle) -> AppState {
    let data_dir = rayvan_native_core::paths::app_data_dir(app)
        .unwrap_or_else(|_| rayvan_daemon_ipc::default_rayvan_data_dir());
    std::fs::create_dir_all(&data_dir).expect("application data directory should be writable");

    let prefs = SessionPrefs::load(&data_dir.join("session-prefs.json"));
    let endpoint = daemon_endpoint_path(None);

    let mut session = DaemonSession {
        connection: None,
        handshake: None,
        endpoint: endpoint.clone(),
        spawned: false,
        last_error: None,
    };

    match launch_or_attach(app, &endpoint) {
        Ok((connection, handshake, spawned)) => {
            let connection = Arc::new(connection);
            forward_daemon_events(app.clone(), Arc::clone(&connection));
            if let Err(error) = connection.subscribe_all() {
                tracing::warn!(error = %error, "failed to subscribe to daemon events");
            }
            session.connection = Some(connection);
            session.handshake = Some(handshake);
            session.spawned = spawned;
            tracing::info!(endpoint = %endpoint, spawned, "connected to rayvand");
        }
        Err(error) => {
            tracing::error!(error = %error, "failed to launch or attach to rayvand");
            session.last_error = Some(error.to_string());
        }
    }

    AppState {
        daemon: Arc::new(Mutex::new(session)),
        prefs: Arc::new(Mutex::new(prefs)),
    }
}

fn launch_or_attach(
    app: &AppHandle,
    endpoint: &str,
) -> Result<(DaemonConnection, HandshakeResponse, bool), DaemonConnectionError> {
    // Prefer attaching to an already-running daemon.
    if let Ok((connection, handshake)) = try_connect(endpoint, Duration::from_millis(400)) {
        return Ok((connection, handshake, false));
    }

    spawn_daemon(app, endpoint)?;

    let credential = wait_for_desktop_credential(Duration::from_secs(10))?;
    let deadline = std::time::Instant::now() + Duration::from_secs(15);
    let mut last_error = None;
    while std::time::Instant::now() < deadline {
        match DaemonConnection::connect(
            Some(endpoint.to_string()),
            Some(credential.clone()),
            Duration::from_millis(500),
            Duration::from_secs(10),
        ) {
            Ok((connection, handshake)) => return Ok((connection, handshake, true)),
            Err(error) => {
                last_error = Some(error);
                std::thread::sleep(Duration::from_millis(150));
            }
        }
    }

    Err(last_error.unwrap_or_else(|| {
        DaemonConnectionError::Unavailable(format!(
            "failed to attach to rayvand at {endpoint} after spawn"
        ))
    }))
}

fn try_connect(
    endpoint: &str,
    timeout: Duration,
) -> Result<(DaemonConnection, HandshakeResponse), DaemonConnectionError> {
    let credential = match wait_for_desktop_credential(Duration::from_millis(200)) {
        Ok(value) => Some(value),
        Err(DaemonConnectionError::Unauthenticated(_)) => None,
        Err(error) => return Err(error),
    };
    match credential {
        Some(credential) => DaemonConnection::connect(
            Some(endpoint.to_string()),
            Some(credential),
            timeout,
            Duration::from_secs(10),
        ),
        None => Err(DaemonConnectionError::Unauthenticated(
            "desktop credential not yet provisioned".into(),
        )),
    }
}

fn spawn_daemon(app: &AppHandle, endpoint: &str) -> Result<(), DaemonConnectionError> {
    // Packaged builds use the externalBin sidecar. Dev builds resolve the
    // workspace daemon instead of empty packaging placeholders.
    #[cfg(not(debug_assertions))]
    {
        if let Ok(sidecar) = app.shell().sidecar("binaries/rayvand") {
            tracing::info!("spawning packaged rayvand sidecar");
            let command = sidecar
                .args(["serve"])
                .env("RAYVAN_DAEMON_ENDPOINT", endpoint);
            command
                .spawn()
                .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?;
            return Ok(());
        }
    }
    #[cfg(debug_assertions)]
    {
        let _ = app;
    }

    let binary = resolve_dev_daemon_binary()?;
    tracing::info!(binary = %binary.display(), "spawning development rayvand");
    let mut command = std::process::Command::new(&binary);
    command
        .arg("serve")
        .env("RAYVAN_DAEMON_ENDPOINT", endpoint)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        command.creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);
    }
    command
        .spawn()
        .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?;
    Ok(())
}

fn resolve_dev_daemon_binary() -> Result<PathBuf, DaemonConnectionError> {
    if let Ok(path) = std::env::var("RAYVAN_DAEMON_BIN") {
        return Ok(PathBuf::from(path));
    }

    // Prefer a built dist entry when present.
    let workspace_candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../daemon/dist/main.js"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../../apps/daemon/dist/main.js"),
    ];
    for candidate in workspace_candidates {
        if candidate.exists() {
            // Wrap with node: return a small launcher script path is awkward on Windows.
            // Instead resolve to `node` and rely on spawn using node + script via a helper.
            // Callers expect an executable — use a platform-specific shell wrapper approach:
            return Ok(create_node_daemon_launcher(&candidate)?);
        }
    }

    // Fall back to pnpm workspace filter entry via a launcher script.
    let daemon_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../daemon/src/main.ts");
    let daemon_src = if daemon_src.exists() {
        daemon_src
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../apps/daemon/src/main.ts")
    };
    if daemon_src.exists() {
        return Ok(create_tsx_daemon_launcher(&daemon_src)?);
    }

    Err(DaemonConnectionError::Unavailable(
        "rayvand binary not found. Set RAYVAN_DAEMON_BIN to a rayvand executable, or build apps/daemon (pnpm --filter @rayvan/daemon build). Packaged builds should embed binaries/rayvand via externalBin.".into(),
    ))
}

fn create_node_daemon_launcher(main_js: &std::path::Path) -> Result<PathBuf, DaemonConnectionError> {
    let runtime_dir = rayvan_daemon_ipc::default_rayvan_runtime_dir();
    std::fs::create_dir_all(&runtime_dir)
        .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?;

    #[cfg(windows)]
    {
        let launcher = runtime_dir.join("rayvand-dev-launcher.cmd");
        let contents = format!(
            "@echo off\r\nnode \"{}\" %*\r\n",
            main_js.display()
        );
        std::fs::write(&launcher, contents)
            .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?;
        Ok(launcher)
    }
    #[cfg(not(windows))]
    {
        let launcher = runtime_dir.join("rayvand-dev-launcher.sh");
        let contents = format!("#!/bin/sh\nexec node \"{}\" \"$@\"\n", main_js.display());
        std::fs::write(&launcher, contents)
            .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?;
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&launcher)
            .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&launcher, perms)
            .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?;
        Ok(launcher)
    }
}

fn create_tsx_daemon_launcher(main_ts: &std::path::Path) -> Result<PathBuf, DaemonConnectionError> {
    let runtime_dir = rayvan_daemon_ipc::default_rayvan_runtime_dir();
    std::fs::create_dir_all(&runtime_dir)
        .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?;
    let daemon_dir = main_ts
        .parent()
        .and_then(|p| p.parent())
        .ok_or_else(|| DaemonConnectionError::Unavailable("invalid daemon path".into()))?;

    #[cfg(windows)]
    {
        let launcher = runtime_dir.join("rayvand-dev-launcher.cmd");
        // Use pnpm exec tsx from the daemon package so Windows resolves workspace tooling.
        let contents = format!(
            "@echo off\r\ncd /d \"{}\"\r\npnpm exec tsx --tsconfig tests/tsconfig.typecheck.json src/main.ts %*\r\n",
            daemon_dir.display()
        );
        std::fs::write(&launcher, contents)
            .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?;
        Ok(launcher)
    }
    #[cfg(not(windows))]
    {
        let launcher = runtime_dir.join("rayvand-dev-launcher.sh");
        let contents = format!(
            "#!/bin/sh\ncd \"{}\" || exit 1\nexec pnpm exec tsx --tsconfig tests/tsconfig.typecheck.json src/main.ts \"$@\"\n",
            daemon_dir.display()
        );
        std::fs::write(&launcher, contents)
            .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?;
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&launcher)
            .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&launcher, perms)
            .map_err(|error| DaemonConnectionError::Unavailable(error.to_string()))?;
        Ok(launcher)
    }
}

fn forward_daemon_events(app: AppHandle, connection: Arc<DaemonConnection>) {
    let (tx, rx) = event_channel();
    connection.set_event_sender(tx);
    std::thread::spawn(move || {
        while let Ok(event) = rx.recv() {
            if let Err(error) = app.emit("daemon://event", event) {
                tracing::debug!(error = %error, "failed to emit daemon event to webview");
            }
        }
    });
}

pub fn with_connection<T>(
    state: &AppState,
    f: impl FnOnce(&DaemonConnection) -> Result<T, DaemonConnectionError>,
) -> Result<T, DaemonConnectionError> {
    let session = state.daemon.lock().unwrap();
    let connection = session
        .connection
        .as_ref()
        .ok_or_else(|| {
            DaemonConnectionError::Unavailable(
                session
                    .last_error
                    .clone()
                    .unwrap_or_else(|| "daemon is not connected".into()),
            )
        })?;
    f(connection)
}

pub fn reconnect(app: &AppHandle, state: &AppState) -> Result<DaemonStatusSnapshot, DaemonConnectionError> {
    let endpoint = daemon_endpoint_path(None);
    let (connection, handshake, spawned) = launch_or_attach(app, &endpoint)?;
    let connection = Arc::new(connection);
    forward_daemon_events(app.clone(), Arc::clone(&connection));
    let _ = connection.subscribe_all();
    let mut session = state.daemon.lock().unwrap();
    session.connection = Some(connection);
    session.handshake = Some(handshake);
    session.spawned = spawned;
    session.endpoint = endpoint;
    session.last_error = None;
    Ok(session.status_snapshot())
}

pub fn daemon_request(
    state: &AppState,
    method: &str,
    params: Value,
) -> Result<Value, DaemonConnectionError> {
    with_connection(state, |connection| connection.request(method, params))
}
