mod application;
pub mod domain;
pub mod network;
pub mod runtime;
pub mod simulation;
pub mod research;
pub use application::status::NetworkStatus;

use application::status::NetworkStatusStore;
use network::client::{MembershipClient, MembershipClientConfig};
use network::identity::{NodeIdentityStore, NODE_ID_FILE_NAME};
use std::net::SocketAddr;
use tauri::Manager;

pub const DEFAULT_NETWORK_ADDRESS: &str = "127.0.0.1:7878";

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let network_status = NetworkStatusStore::default();
    let client_status = network_status.clone();

    tauri::Builder::default()
        .manage(network_status)
        .manage(research::ExperimentProcessState::default())
        .setup(move |app| {
            let identity_path = app.path().app_local_data_dir()?.join(NODE_ID_FILE_NAME);
            let node_id = NodeIdentityStore::new(identity_path).load_or_create()?;
            let service_address: SocketAddr = std::env::var("RAYVAN_NETWORK_ADDR")
                .unwrap_or_else(|_| DEFAULT_NETWORK_ADDRESS.to_owned())
                .parse()?;
            let client =
                MembershipClient::new(MembershipClientConfig::for_service(service_address));
            let status = client_status.clone();

            tauri::async_runtime::spawn(async move {
                client.run(node_id, move |next| status.set(next)).await;
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            application::status::get_application_status,
            research::get_research_schema,
            research::estimate_experiment,
            research::start_experiment,
            research::cancel_experiment,
            research::get_active_experiment,
            research::list_experiments,
            research::get_experiment
        ])
        .run(tauri::generate_context!())
        .expect("failed to run the Rayvan desktop application");
}
