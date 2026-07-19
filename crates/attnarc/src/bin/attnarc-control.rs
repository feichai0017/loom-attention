use attnarc::catalog::{HoltCatalog, PersistentCatalog};
use axum::{extract::State, routing::get, Json, Router};
use clap::Parser;
use serde::Serialize;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, Parser)]
#[command(name = "attnarc-control")]
#[command(about = "Slow-path catalog and scheduler for AttnArc workers")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
    #[arg(long, default_value = ".attnarc/catalog")]
    catalog_path: PathBuf,
}

#[derive(Debug)]
struct AppState {
    catalog_backend: String,
    catalog_path: PathBuf,
    _catalog: Mutex<HoltCatalog>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    service: &'static str,
    architecture_version: u32,
}

#[derive(Debug, Serialize)]
struct StateResponse {
    service: &'static str,
    role: &'static str,
    catalog_backend: String,
    catalog_path: PathBuf,
    fast_path: &'static str,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "attnarc=info".to_owned()))
        .init();
    let args = Args::parse();
    let catalog = HoltCatalog::open(&args.catalog_path)?;
    let state = Arc::new(AppState {
        catalog_backend: catalog.name().to_owned(),
        catalog_path: args.catalog_path,
        _catalog: Mutex::new(catalog),
    });

    let app = Router::new()
        .route("/healthz", get(health))
        .route("/v1/state", get(service_state))
        .with_state(state);
    tracing::info!(bind = %args.bind, "starting AttnArc control service");
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "attnarc-control",
        architecture_version: 3,
    })
}

async fn service_state(State(state): State<Arc<AppState>>) -> Json<StateResponse> {
    Json(StateResponse {
        service: "attnarc-control",
        role: "slow-path catalog and scheduler",
        catalog_backend: state.catalog_backend.clone(),
        catalog_path: state.catalog_path.clone(),
        fast_path: "node-local runtime only",
    })
}
