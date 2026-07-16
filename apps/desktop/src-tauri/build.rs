use std::fs;
use std::path::PathBuf;

fn main() {
    ensure_sidecar_placeholders();
    tauri_build::build()
}

/// Tauri `externalBin` requires host-triple files to exist at build time.
/// Dev/packaging scaffolding creates empty placeholders when real SEA binaries
/// are not present yet.
fn ensure_sidecar_placeholders() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let binaries = manifest_dir.join("binaries");
    let _ = fs::create_dir_all(&binaries);

    let triple = std::env::var("TAURI_ENV_TARGET_TRIPLE")
        .or_else(|_| std::env::var("TARGET"))
        .unwrap_or_else(|_| {
            if cfg!(windows) {
                "x86_64-pc-windows-msvc".into()
            } else if cfg!(target_os = "macos") {
                if cfg!(target_arch = "aarch64") {
                    "aarch64-apple-darwin".into()
                } else {
                    "x86_64-apple-darwin".into()
                }
            } else {
                "x86_64-unknown-linux-gnu".into()
            }
        });

    let suffix = if cfg!(windows) { ".exe" } else { "" };
    for name in ["rayvand", "rayvan-mcp"] {
        let path = binaries.join(format!("{name}-{triple}{suffix}"));
        if !path.exists() {
            let _ = fs::write(&path, []);
            println!("cargo:warning=created sidecar placeholder {}", path.display());
        }
    }
}
