mod commands;
mod state;

use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            let app_state = state::initialize_app_state(app.handle());
            app.manage(app_state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::greet,
            commands::get_app_info,
            commands::list_projects,
            commands::get_project,
            commands::create_project,
            commands::update_project,
            commands::set_project_archived,
            commands::delete_project,
            commands::get_current_project_id,
            commands::set_current_project_id,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Rayvan desktop application");
}
