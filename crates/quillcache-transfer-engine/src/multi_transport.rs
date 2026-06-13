//! MultiTransport (Mooncake's `multi_transport.h`) — the registry of installed
//! transports plus per-request backend selection. The engine installs one
//! backend per protocol (`"tcp"`, `"rdma"`, …) and selects by the target
//! segment's protocol. Mooncake's `MultiTransport::selectTransport` also weighs
//! topology; that hook lands with the RDMA backend.

use crate::transport::Transport;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct MultiTransport {
    transports: HashMap<String, Arc<dyn Transport>>,
}

impl MultiTransport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn install(&mut self, protocol: impl Into<String>, transport: Arc<dyn Transport>) {
        self.transports.insert(protocol.into(), transport);
    }

    /// Pick the backend for a target segment's protocol (exact match).
    pub fn select(&self, protocol: &str) -> Option<Arc<dyn Transport>> {
        self.transports.get(protocol).cloned()
    }

    /// Topology-aware selection (Mooncake's `selectTransport` weighing the link):
    /// among the `protocols` a target supports, pick the installed backend with
    /// the fastest link class — so an RDMA-capable peer uses RDMA and falls back
    /// to TCP when RDMA isn't installed. `None` if none of them are installed.
    pub fn select_best(&self, protocols: &[&str]) -> Option<Arc<dyn Transport>> {
        protocols
            .iter()
            .filter_map(|p| self.transports.get(*p))
            .min_by_key(|t| t.link_class().rank())
            .cloned()
    }

    pub fn installed(&self) -> Vec<String> {
        self.transports.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::rdma::RdmaTransport;
    use crate::transport::tcp::TcpTransport;

    #[test]
    fn select_best_prefers_rdma_then_falls_back_to_tcp() {
        let mut mt = MultiTransport::new();
        mt.install("tcp", Arc::new(TcpTransport));
        mt.install("rdma", Arc::new(RdmaTransport::default()));

        // Both available for the target → RDMA wins (faster link class).
        assert_eq!(mt.select_best(&["tcp", "rdma"]).unwrap().name(), "rdma");
        // Only TCP supported → TCP.
        assert_eq!(mt.select_best(&["tcp"]).unwrap().name(), "tcp");
        // Exact-protocol select still works.
        assert_eq!(mt.select("rdma").unwrap().name(), "rdma");
        // Nothing installed for these protocols → None.
        assert!(mt.select_best(&["nvme-of"]).is_none());
    }
}
