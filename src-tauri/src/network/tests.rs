use super::client::{MembershipClient, MembershipClientConfig};
use super::identity::{NodeIdentityStore, NODE_ID_FILE_NAME};
use super::protocol::{read_message, write_message, MembershipMessage, PROTOCOL_VERSION};
use super::service::{BootstrapServiceConfig, MembershipRegistry, RunningBootstrapService};
use crate::application::status::NetworkStatusStore;
use crate::NetworkStatus;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tempfile::tempdir;
use tokio::net::TcpStream;
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout};

const TEST_TIMEOUT: Duration = Duration::from_secs(3);

fn service_config() -> BootstrapServiceConfig {
    BootstrapServiceConfig {
        handshake_timeout: Duration::from_millis(250),
        heartbeat_timeout: Duration::from_millis(150),
    }
}

async fn start_service() -> RunningBootstrapService {
    RunningBootstrapService::start(
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
        service_config(),
    )
    .await
    .expect("the test service should bind to localhost")
}

fn client_config(address: SocketAddr) -> MembershipClientConfig {
    MembershipClientConfig {
        service_address: address,
        connect_timeout: Duration::from_millis(200),
        heartbeat_interval: Duration::from_millis(30),
        heartbeat_response_timeout: Duration::from_millis(100),
        reconnect_delay: Duration::from_millis(50),
        protocol_version: PROTOCOL_VERSION,
    }
}

fn start_client(
    address: SocketAddr,
    node_id: &str,
) -> (
    NetworkStatusStore,
    Arc<Mutex<Vec<NetworkStatus>>>,
    JoinHandle<()>,
) {
    let status = NetworkStatusStore::default();
    let task_status = status.clone();
    let history = Arc::new(Mutex::new(Vec::new()));
    let task_history = history.clone();
    let client = MembershipClient::new(client_config(address));
    let node_id = node_id.to_owned();
    let task = tokio::spawn(async move {
        client
            .run(node_id, move |next| {
                task_status.set(next);
                task_history
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(next);
            })
            .await;
    });
    (status, history, task)
}

async fn wait_for_status(status: &NetworkStatusStore, expected: NetworkStatus) {
    timeout(TEST_TIMEOUT, async {
        loop {
            if status.get() == expected {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("status did not become {expected:?}"));
}

async fn wait_for_member(registry: &MembershipRegistry, node_id: &str) {
    timeout(TEST_TIMEOUT, async {
        loop {
            if registry.contains(node_id).await {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("node {node_id} was not registered"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn node_successfully_joins_local_service() {
    let service = start_service().await;
    let registry = service.registry();
    let (status, history, client_task) = start_client(service.address(), "node-success");

    wait_for_status(&status, NetworkStatus::Connected).await;
    wait_for_member(&registry, "node-success").await;
    assert_eq!(registry.count().await, 1);
    assert!(history
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .windows(2)
        .any(|states| states == [NetworkStatus::Connecting, NetworkStatus::Connected]));

    client_task.abort();
    let _ = client_task.await;
    service
        .shutdown()
        .await
        .expect("service should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_protocol_version_is_rejected() {
    let service = start_service().await;
    let mut stream = TcpStream::connect(service.address())
        .await
        .expect("client should connect to the service");

    write_message(
        &mut stream,
        &MembershipMessage::RegisterNode {
            protocol_version: PROTOCOL_VERSION + 1,
            node_id: "wrong-version".to_owned(),
        },
    )
    .await
    .expect("registration frame should be sent");

    let response = read_message(&mut stream)
        .await
        .expect("service should send a rejection");
    assert!(matches!(response, MembershipMessage::JoinRejected { .. }));
    assert_eq!(service.registry().count().await, 0);

    service
        .shutdown()
        .await
        .expect("service should stop cleanly");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn connection_loss_returns_node_to_not_connected() {
    let service = start_service().await;
    let (status, history, client_task) = start_client(service.address(), "node-loss");
    wait_for_status(&status, NetworkStatus::Connected).await;

    service
        .shutdown()
        .await
        .expect("service should stop cleanly");
    timeout(TEST_TIMEOUT, async {
        loop {
            let saw_disconnect = history
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .windows(2)
                .any(|states| states == [NetworkStatus::Connected, NetworkStatus::NotConnected]);
            if saw_disconnect {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("client should observe the stopped service");

    client_task.abort();
    let _ = client_task.await;
}

#[test]
fn node_id_survives_local_reload() {
    let directory = tempdir().expect("temporary identity directory should be created");
    let identity_path = directory.path().join(NODE_ID_FILE_NAME);

    let first = NodeIdentityStore::new(&identity_path)
        .load_or_create()
        .expect("first load should create the node ID");
    let second = NodeIdentityStore::new(&identity_path)
        .load_or_create()
        .expect("second load should reuse the node ID");

    assert_eq!(first, second);
    assert_eq!(
        std::fs::read_to_string(identity_path)
            .expect("identity file should remain readable")
            .trim(),
        first
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn heartbeats_keep_active_node_registered() {
    let service = start_service().await;
    let registry = service.registry();
    let (status, _, client_task) = start_client(service.address(), "node-heartbeat");
    wait_for_status(&status, NetworkStatus::Connected).await;
    wait_for_member(&registry, "node-heartbeat").await;

    sleep(Duration::from_millis(500)).await;
    assert!(registry.contains("node-heartbeat").await);
    assert_eq!(registry.count().await, 1);

    client_task.abort();
    let _ = client_task.await;
    service
        .shutdown()
        .await
        .expect("service should stop cleanly");
}
