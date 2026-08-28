use rayvan_lib::network::service::{BootstrapServiceConfig, RunningBootstrapService};
use rayvan_lib::DEFAULT_NETWORK_ADDRESS;
use std::error::Error;
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let address: SocketAddr = std::env::var("RAYVAN_NETWORK_BIND")
        .unwrap_or_else(|_| DEFAULT_NETWORK_ADDRESS.to_owned())
        .parse()?;
    let service =
        RunningBootstrapService::start(address, BootstrapServiceConfig::default()).await?;

    println!("Rayvan network service listening on {}", service.address());
    tokio::signal::ctrl_c().await?;
    service.shutdown().await?;
    Ok(())
}
