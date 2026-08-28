use crate::application::status::NetworkStatus;
use crate::network::protocol::{
    read_message, write_message, FrameError, MembershipMessage, PROTOCOL_VERSION,
};
use std::io;
use std::net::SocketAddr;
use std::time::Duration;
use thiserror::Error;
use tokio::net::TcpStream;
use tokio::time::{interval, sleep, timeout, MissedTickBehavior};

#[derive(Debug, Clone)]
pub struct MembershipClientConfig {
    pub service_address: SocketAddr,
    pub connect_timeout: Duration,
    pub heartbeat_interval: Duration,
    pub heartbeat_response_timeout: Duration,
    pub reconnect_delay: Duration,
    pub protocol_version: u16,
}

impl MembershipClientConfig {
    pub fn for_service(service_address: SocketAddr) -> Self {
        Self {
            service_address,
            connect_timeout: Duration::from_secs(3),
            heartbeat_interval: Duration::from_secs(2),
            heartbeat_response_timeout: Duration::from_secs(3),
            reconnect_delay: Duration::from_secs(2),
            protocol_version: PROTOCOL_VERSION,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MembershipClient {
    config: MembershipClientConfig,
}

impl MembershipClient {
    pub fn new(config: MembershipClientConfig) -> Self {
        Self { config }
    }

    pub async fn run<F>(self, node_id: String, status_changed: F)
    where
        F: Fn(NetworkStatus) + Send + Sync + 'static,
    {
        loop {
            status_changed(NetworkStatus::Connecting);
            if let Ok(stream) = self.join(&node_id).await {
                status_changed(NetworkStatus::Connected);
                let _ = self.heartbeat_loop(stream, &node_id).await;
            }
            status_changed(NetworkStatus::NotConnected);
            sleep(self.config.reconnect_delay).await;
        }
    }

    async fn join(&self, node_id: &str) -> Result<TcpStream, MembershipClientError> {
        let mut stream = timeout(
            self.config.connect_timeout,
            TcpStream::connect(self.config.service_address),
        )
        .await
        .map_err(|_| MembershipClientError::ConnectTimeout)??;

        write_message(
            &mut stream,
            &MembershipMessage::RegisterNode {
                protocol_version: self.config.protocol_version,
                node_id: node_id.to_owned(),
            },
        )
        .await?;

        let response = timeout(
            self.config.heartbeat_response_timeout,
            read_message(&mut stream),
        )
        .await
        .map_err(|_| MembershipClientError::JoinResponseTimeout)??;

        match response {
            MembershipMessage::JoinAccepted { protocol_version }
                if protocol_version == self.config.protocol_version => {}
            MembershipMessage::JoinAccepted { protocol_version } => {
                return Err(MembershipClientError::WrongAcceptedVersion(
                    protocol_version,
                ));
            }
            MembershipMessage::JoinRejected { reason } => {
                return Err(MembershipClientError::JoinRejected(reason));
            }
            _ => return Err(MembershipClientError::UnexpectedMessage),
        }

        Ok(stream)
    }

    async fn heartbeat_loop(
        &self,
        mut stream: TcpStream,
        node_id: &str,
    ) -> Result<(), MembershipClientError> {
        let mut heartbeats = interval(self.config.heartbeat_interval);
        heartbeats.set_missed_tick_behavior(MissedTickBehavior::Delay);

        loop {
            heartbeats.tick().await;
            write_message(
                &mut stream,
                &MembershipMessage::Heartbeat {
                    node_id: node_id.to_owned(),
                },
            )
            .await?;

            let response = timeout(
                self.config.heartbeat_response_timeout,
                read_message(&mut stream),
            )
            .await
            .map_err(|_| MembershipClientError::HeartbeatResponseTimeout)??;
            match response {
                MembershipMessage::Heartbeat {
                    node_id: heartbeat_node_id,
                } if heartbeat_node_id == node_id => {}
                _ => return Err(MembershipClientError::UnexpectedMessage),
            }
        }
    }
}

#[derive(Debug, Error)]
enum MembershipClientError {
    #[error("timed out connecting to the membership service")]
    ConnectTimeout,
    #[error("timed out waiting for the join response")]
    JoinResponseTimeout,
    #[error("timed out waiting for the heartbeat response")]
    HeartbeatResponseTimeout,
    #[error("the service rejected membership: {0}")]
    JoinRejected(String),
    #[error("the service accepted a different protocol version: {0}")]
    WrongAcceptedVersion(u16),
    #[error("the service sent an unexpected membership message")]
    UnexpectedMessage,
    #[error(transparent)]
    Frame(#[from] FrameError),
    #[error("membership connection failed: {0}")]
    Io(#[from] io::Error),
}
