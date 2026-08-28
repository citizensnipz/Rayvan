mod application;
pub mod domain;
pub mod runtime;
pub mod simulation;
pub use application::status::NetworkStatus;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            application::status::get_application_status
        ])
        .run(tauri::generate_context!())
        .expect("failed to run the Rayvan desktop application");
}
