use crate::network::protocol::{
    read_message, write_message, FrameError, MembershipMessage, PROTOCOL_VERSION,
};
use std::collections::HashMap;
use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{watch, RwLock};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::{timeout, Instant};

#[derive(Debug, Clone)]
pub struct BootstrapServiceConfig {
    pub handshake_timeout: Duration,
    pub heartbeat_timeout: Duration,
}

impl Default for BootstrapServiceConfig {
    fn default() -> Self {
        Self {
            handshake_timeout: Duration::from_secs(5),
            heartbeat_timeout: Duration::from_secs(10),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MembershipRegistry {
    members: Arc<RwLock<HashMap<String, Membership>>>,
}

impl MembershipRegistry {
    pub async fn contains(&self, node_id: &str) -> bool {
        self.members.read().await.contains_key(node_id)
    }

    pub async fn count(&self) -> usize {
        self.members.read().await.len()
    }

    pub async fn node_ids(&self) -> Vec<String> {
        self.members.read().await.keys().cloned().collect()
    }

    async fn register(&self, node_id: String, connection_id: u64) {
        self.members.write().await.insert(
            node_id,
            Membership {
                connection_id,
                last_heartbeat: Instant::now(),
            },
        );
    }

    async fn heartbeat(&self, node_id: &str, connection_id: u64) -> bool {
        let mut members = self.members.write().await;
        let Some(membership) = members.get_mut(node_id) else {
            return false;
        };
        if membership.connection_id != connection_id {
            return false;
        }

        membership.last_heartbeat = Instant::now();
        true
    }

    async fn disconnect(&self, node_id: &str, connection_id: u64) {
        let mut members = self.members.write().await;
        if members
            .get(node_id)
            .is_some_and(|membership| membership.connection_id == connection_id)
        {
            members.remove(node_id);
        }
    }
}

#[derive(Debug)]
struct Membership {
    connection_id: u64,
    last_heartbeat: Instant,
}

pub struct RunningBootstrapService {
    address: SocketAddr,
    registry: MembershipRegistry,
    shutdown_sender: Option<watch::Sender<bool>>,
    task: Option<JoinHandle<io::Result<()>>>,
}

impl RunningBootstrapService {
    pub async fn start(address: SocketAddr, config: BootstrapServiceConfig) -> io::Result<Self> {
        let listener = TcpListener::bind(address).await?;
        let address = listener.local_addr()?;
        let registry = MembershipRegistry::default();
        let (shutdown_sender, shutdown_receiver) = watch::channel(false);
        let service_registry = registry.clone();
        let task = tokio::spawn(run_service(
            listener,
            service_registry,
            config,
            shutdown_receiver,
        ));

        Ok(Self {
            address,
            registry,
            shutdown_sender: Some(shutdown_sender),
            task: Some(task),
        })
    }

    pub fn address(&self) -> SocketAddr {
        self.address
    }

    pub fn registry(&self) -> MembershipRegistry {
        self.registry.clone()
    }

    pub async fn shutdown(mut self) -> io::Result<()> {
        if let Some(sender) = self.shutdown_sender.take() {
            let _ = sender.send(true);
        }
        match self.task.take() {
            Some(task) => task.await.map_err(join_error_to_io)?,
            None => Ok(()),
        }
    }
}

impl Drop for RunningBootstrapService {
    fn drop(&mut self) {
        if let Some(sender) = self.shutdown_sender.take() {
            let _ = sender.send(true);
        }
    }
}

fn join_error_to_io(error: tokio::task::JoinError) -> io::Error {
    io::Error::other(format!("service task failed: {error}"))
}

async fn run_service(
    listener: TcpListener,
    registry: MembershipRegistry,
    config: BootstrapServiceConfig,
    mut shutdown: watch::Receiver<bool>,
) -> io::Result<()> {
    let connection_ids = Arc::new(AtomicU64::new(1));
    let mut connections = JoinSet::new();

    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_ok() && *shutdown.borrow() {
                    break;
                }
            }
            accepted = listener.accept() => {
                let (stream, _) = accepted?;
                let connection_id = connection_ids.fetch_add(1, Ordering::Relaxed);
                connections.spawn(handle_connection(
                    stream,
                    connection_id,
                    registry.clone(),
                    config.clone(),
                    shutdown.clone(),
                ));
            }
            completed = connections.join_next(), if !connections.is_empty() => {
                if let Some(Err(error)) = completed {
                    eprintln!("Rayvan membership connection task failed: {error}");
                }
            }
        }
    }

    while connections.join_next().await.is_some() {}
    Ok(())
}

async fn handle_connection(
    mut stream: TcpStream,
    connection_id: u64,
    registry: MembershipRegistry,
    config: BootstrapServiceConfig,
    mut shutdown: watch::Receiver<bool>,
) -> Result<(), ConnectionError> {
    let registration = timeout(config.handshake_timeout, read_message(&mut stream))
        .await
        .map_err(|_| ConnectionError::HandshakeTimeout)??;

    let (protocol_version, node_id) = match registration {
        MembershipMessage::RegisterNode {
            protocol_version,
            node_id,
        } => (protocol_version, node_id),
        _ => {
            reject(&mut stream, "the first message must register a node").await?;
            return Ok(());
        }
    };

    if protocol_version != PROTOCOL_VERSION {
        reject(
            &mut stream,
            &format!(
                "unsupported protocol version {protocol_version}; expected {PROTOCOL_VERSION}"
            ),
        )
        .await?;
        return Ok(());
    }
    if node_id.is_empty() || node_id.len() > 128 {
        reject(&mut stream, "node ID must contain 1 to 128 bytes").await?;
        return Ok(());
    }

    registry.register(node_id.clone(), connection_id).await;
    if let Err(error) = write_message(
        &mut stream,
        &MembershipMessage::JoinAccepted {
            protocol_version: PROTOCOL_VERSION,
        },
    )
    .await
    {
        registry.disconnect(&node_id, connection_id).await;
        return Err(error.into());
    }
    println!("Rayvan node joined: {node_id}");

    let result = heartbeat_loop(
        &mut stream,
        &node_id,
        connection_id,
        &registry,
        config.heartbeat_timeout,
        &mut shutdown,
    )
    .await;

    registry.disconnect(&node_id, connection_id).await;
    println!("Rayvan node disconnected: {node_id}");
    result
}

async fn heartbeat_loop(
    stream: &mut TcpStream,
    node_id: &str,
    connection_id: u64,
    registry: &MembershipRegistry,
    heartbeat_timeout: Duration,
    shutdown: &mut watch::Receiver<bool>,
) -> Result<(), ConnectionError> {
    loop {
        let message = tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_ok() && *shutdown.borrow() {
                    return Ok(());
                }
                continue;
            }
            result = timeout(heartbeat_timeout, read_message(stream)) => {
                result.map_err(|_| ConnectionError::HeartbeatTimeout)??
            }
        };

        match message {
            MembershipMessage::Heartbeat {
                node_id: heartbeat_node_id,
            } if heartbeat_node_id == node_id => {
                if !registry.heartbeat(node_id, connection_id).await {
                    return Err(ConnectionError::ReplacedConnection);
                }
                write_message(
                    stream,
                    &MembershipMessage::Heartbeat {
                        node_id: node_id.to_owned(),
                    },
                )
                .await?;
            }
            MembershipMessage::Heartbeat { .. } => {
                return Err(ConnectionError::WrongHeartbeatNode);
            }
            _ => return Err(ConnectionError::UnexpectedMessage),
        }
    }
}

async fn reject(stream: &mut TcpStream, reason: &str) -> Result<(), FrameError> {
    write_message(
        stream,
        &MembershipMessage::JoinRejected {
            reason: reason.to_owned(),
        },
    )
    .await
}

#[derive(Debug, Error)]
enum ConnectionError {
    #[error(transparent)]
    Frame(#[from] FrameError),
    #[error("node registration timed out")]
    HandshakeTimeout,
    #[error("node heartbeat timed out")]
    HeartbeatTimeout,
    #[error("heartbeat named a different node")]
    WrongHeartbeatNode,
    #[error("received an unexpected membership message")]
    UnexpectedMessage,
    #[error("another connection replaced this node membership")]
    ReplacedConnection,
}
