use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ActiveExperiment {
    pub run_id: String,
    pub run_directory: String,
    pub cancellation_requested: bool,
}

#[derive(Default)]
pub struct ExperimentProcessState(pub Mutex<Option<ActiveExperiment>>);

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StartRequest {
    pub config: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StartResponse {
    pub run_id: String,
    pub run_directory: String,
}

#[tauri::command]
pub async fn get_research_schema() -> Result<Value, String> {
    run_python_json(&["schema"], None)
}

#[tauri::command]
pub async fn estimate_experiment(config: Value) -> Result<Value, String> {
    let temporary = std::env::temp_dir().join(format!(
        "rayvan-estimate-{}.json",
        Uuid::new_v4().simple()
    ));
    write_json(&temporary, &config)?;
    let result = run_python_json(
        &["estimate", temporary.to_string_lossy().as_ref()],
        None,
    );
    let _ = fs::remove_file(temporary);
    result
}

#[tauri::command]
pub async fn start_experiment(
    app: AppHandle,
    state: State<'_, ExperimentProcessState>,
    request: StartRequest,
) -> Result<StartResponse, String> {
    {
        let active = state.0.lock().map_err(|_| "experiment state is unavailable")?;
        if let Some(run) = active.as_ref() {
            return Err(format!("experiment {} is already running", run.run_id));
        }
    }

    let run_id = format!(
        "{}-{}",
        chrono_stamp(),
        &Uuid::new_v4().simple().to_string()[..8]
    );
    let runs_directory = research_runs_directory(&app)?;
    fs::create_dir_all(&runs_directory).map_err(|error| error.to_string())?;
    let run_directory = runs_directory.join(&run_id);
    let launch_config = runs_directory.join(format!(".launch-{run_id}.json"));
    write_json(&launch_config, &request.config)?;

    let python = python_executable();
    let emc_root = emc_root()?;
    let mut command = Command::new(&python);
    command
        .arg("-m")
        .arg("rayvan_emc.research")
        .arg("run")
        .arg(&launch_config)
        .arg("--runs-dir")
        .arg(&runs_directory)
        .arg("--run-id")
        .arg(&run_id)
        .current_dir(&emc_root)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONPATH", python_path(&emc_root))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|error| {
        format!(
            "could not start Python ({python:?}): {error}. Set RAYVAN_PYTHON to the EMC environment interpreter"
        )
    })?;
    let stdout = child.stdout.take().ok_or("Python stdout was not captured")?;
    let stderr = child.stderr.take().ok_or("Python stderr was not captured")?;

    let active = ActiveExperiment {
        run_id: run_id.clone(),
        run_directory: run_directory.to_string_lossy().into_owned(),
        cancellation_requested: false,
    };
    *state.0.lock().map_err(|_| "experiment state is unavailable")? = Some(active);

    let stdout_app = app.clone();
    let stdout_run = run_id.clone();
    let stdout_log = run_directory.join("logs.txt");
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            match serde_json::from_str::<Value>(&line) {
                Ok(event) => {
                    let _ = stdout_app.emit("research-event", event);
                }
                Err(_) => {
                    append_log(&stdout_log, &format!("[stdout] {line}"));
                    let _ = stdout_app.emit(
                        "research-log",
                        json!({"runId": stdout_run, "stream": "stdout", "line": line}),
                    );
                }
            }
        }
    });
    let stderr_app = app.clone();
    let stderr_run = run_id.clone();
    let stderr_log = run_directory.join("logs.txt");
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            append_log(&stderr_log, &format!("[stderr] {line}"));
            let _ = stderr_app.emit(
                "research-log",
                json!({"runId": stderr_run, "stream": "stderr", "line": line}),
            );
        }
    });
    let wait_app = app.clone();
    let wait_run = run_id.clone();
    std::thread::spawn(move || {
        let result = child.wait();
        let _ = fs::remove_file(launch_config);
        if let Some(state) = wait_app.try_state::<ExperimentProcessState>() {
            if let Ok(mut slot) = state.0.lock() {
                if slot.as_ref().map(|active| active.run_id.as_str()) == Some(wait_run.as_str()) {
                    *slot = None;
                }
            }
        }
        if let Ok(status) = result {
            let _ = wait_app.emit(
                "research-process-exit",
                json!({"runId": wait_run, "exitCode": status.code()}),
            );
        }
    });

    Ok(StartResponse {
        run_id,
        run_directory: run_directory.to_string_lossy().into_owned(),
    })
}

#[tauri::command]
pub async fn cancel_experiment(
    state: State<'_, ExperimentProcessState>,
) -> Result<(), String> {
    let mut active = state.0.lock().map_err(|_| "experiment state is unavailable")?;
    let run = active.as_mut().ok_or("no experiment is currently running")?;
    let directory = PathBuf::from(&run.run_directory);
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    fs::write(directory.join("cancel.requested"), b"requested\n")
        .map_err(|error| error.to_string())?;
    run.cancellation_requested = true;
    Ok(())
}

#[tauri::command]
pub fn get_active_experiment(
    state: State<'_, ExperimentProcessState>,
) -> Result<Option<ActiveExperiment>, String> {
    state
        .0
        .lock()
        .map(|active| active.clone())
        .map_err(|_| "experiment state is unavailable".to_owned())
}

#[tauri::command]
pub fn list_experiments(app: AppHandle) -> Result<Vec<Value>, String> {
    let directory = research_runs_directory(&app)?;
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let active_id = app
        .try_state::<ExperimentProcessState>()
        .and_then(|state| state.0.lock().ok()?.as_ref().map(|run| run.run_id.clone()));
    let mut rows = Vec::new();
    for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        if !entry.file_type().map_err(|error| error.to_string())?.is_dir() {
            continue;
        }
        let path = entry.path();
        let run_id = entry.file_name().to_string_lossy().into_owned();
        let summary = read_optional_json(&path.join("summary.json"));
        let status = read_optional_json(&path.join("status.json"));
        let config = read_optional_json(&path.join("config.json"));
        let mut row = summary.unwrap_or_else(|| {
            json!({
                "run_id": run_id,
                "name": config.as_ref().and_then(|v| v.get("name")).and_then(Value::as_str).unwrap_or(&run_id),
                "suite": config.as_ref().and_then(|v| v.get("suite")),
                "architecture": config.as_ref().and_then(|v| v.get("architecture")),
                "started_at": read_optional_json(&path.join("metadata.json")).and_then(|v| v.get("started_at").cloned()),
            })
        });
        let raw_status = status.as_ref().and_then(|v| v.get("status")).and_then(Value::as_str).unwrap_or("partial");
        let resolved_status = if raw_status == "running" && active_id.as_deref() != Some(&run_id) { "interrupted" } else { raw_status };
        if let Some(object) = row.as_object_mut() {
            object.insert("status".to_owned(), Value::String(resolved_status.to_owned()));
            object.insert("runDirectory".to_owned(), Value::String(path.to_string_lossy().into_owned()));
        }
        rows.push(row);
    }
    rows.sort_by(|left, right| right.get("started_at").and_then(Value::as_str).cmp(&left.get("started_at").and_then(Value::as_str)));
    Ok(rows)
}

#[tauri::command]
pub fn get_experiment(app: AppHandle, run_id: String) -> Result<Value, String> {
    validate_run_id(&run_id)?;
    let directory = research_runs_directory(&app)?.join(&run_id);
    if !directory.is_dir() {
        return Err(format!("run {run_id} does not exist"));
    }
    let metrics = read_jsonl(&directory.join("metrics.jsonl"))?;
    Ok(json!({
        "runId": run_id,
        "runDirectory": directory,
        "config": read_optional_json(&directory.join("config.json")),
        "metadata": read_optional_json(&directory.join("metadata.json")),
        "model": read_optional_json(&directory.join("model.json")),
        "summary": read_optional_json(&directory.join("summary.json")),
        "projections": read_optional_json(&directory.join("projections.json")),
        "diagnostics": read_optional_json(&directory.join("diagnostics").join("report.json")),
        "events": metrics,
        "logs": fs::read_to_string(directory.join("logs.txt")).unwrap_or_default(),
    }))
}

fn run_python_json(arguments: &[&str], working_directory: Option<&Path>) -> Result<Value, String> {
    let emc_root = emc_root()?;
    let output = Command::new(python_executable())
        .arg("-m")
        .arg("rayvan_emc.research")
        .args(arguments)
        .current_dir(working_directory.unwrap_or(&emc_root))
        .env("PYTHONPATH", python_path(&emc_root))
        .output()
        .map_err(|error| format!("could not start the EMC Python environment: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    serde_json::from_slice(&output.stdout).map_err(|error| format!("Python returned malformed JSON: {error}"))
}

fn research_runs_directory(app: &AppHandle) -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("RAYVAN_RUNS_DIR") {
        return Ok(PathBuf::from(path));
    }
    app.path()
        .app_local_data_dir()
        .map(|path| path.join("research-runs"))
        .map_err(|error| error.to_string())
}

fn emc_root() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("RAYVAN_EMC_ROOT") {
        return Ok(PathBuf::from(path));
    }
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = source.parent().unwrap_or(&source).join("emc");
    if candidate.join("rayvan_emc").is_dir() {
        return Ok(candidate);
    }
    Err("the rayvan_emc Python package could not be located; set RAYVAN_EMC_ROOT".to_owned())
}

fn python_executable() -> PathBuf {
    if let Ok(path) = std::env::var("RAYVAN_PYTHON") {
        return PathBuf::from(path);
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    for candidate in [root.join(".venv/Scripts/python.exe"), root.join("emc/.venv/Scripts/python.exe")] {
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("python")
}

fn python_path(emc_root: &Path) -> String {
    let separator = if cfg!(windows) { ";" } else { ":" };
    match std::env::var("PYTHONPATH") {
        Ok(existing) if !existing.is_empty() => format!("{}{separator}{existing}", emc_root.display()),
        _ => emc_root.to_string_lossy().into_owned(),
    }
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn read_optional_json(path: &Path) -> Option<Value> {
    fs::read(path).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>, String> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = fs::File::open(path).map_err(|error| error.to_string())?;
    Ok(BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|line| serde_json::from_str(&line).ok())
        .collect())
}

fn validate_run_id(run_id: &str) -> Result<(), String> {
    if run_id.is_empty() || !run_id.chars().all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_')) {
        return Err("invalid run identifier".to_owned());
    }
    Ok(())
}

fn chrono_stamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs().to_string()
}

fn append_log(path: &Path, line: &str) {
    if let Ok(mut stream) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(stream, "{line}");
    }
}
