//! HA: etcd-backed master leader election — Mooncake's high-availability mode,
//! where "multiple master nodes [are] coordinated through an etcd cluster … etcd
//! [elects] a leader … if the current leader fails … the remaining master nodes
//! automatically perform a new leader election".
//!
//! N masters [`MasterElection::campaign`] on a shared election key, each backed by
//! an etcd **lease**: etcd serializes the campaigns so exactly one is leader at a
//! time; the rest block (stand by). If the leader's process dies, its lease lapses
//! and a standby wins — the new leader rebuilds state from the latest
//! [`crate::MasterSnapshot`]. This is the orchestration on top of snapshot/recovery;
//! object metadata still lives in-memory in the [`crate::MasterService`].
//!
//! Built only with `--features etcd` (pulls `etcd-client`); the integration test
//! is `#[ignore]` since it needs a running etcd.

use etcd_client::{Client, Error, LeaderKey, ResignOptions};

/// One master's participation in the leader election.
pub struct MasterElection {
    client: Client,
    /// The shared election key every master campaigns on.
    election: String,
    /// This master's identity — the value stored when it wins (the leader value).
    node_id: String,
    /// The lease backing this master's leadership; lapses if the process dies.
    lease_id: i64,
}

/// Proof that this master currently holds leadership — the fencing key to resign.
pub struct Leadership {
    leader_key: LeaderKey,
}

impl MasterElection {
    /// Connect to the etcd cluster and grant a `lease_ttl_secs` lease that backs
    /// this master's leadership (the leader must keep it alive; if its process
    /// dies the lease lapses and a standby is elected).
    pub async fn join(
        endpoints: Vec<String>,
        election: impl Into<String>,
        node_id: impl Into<String>,
        lease_ttl_secs: i64,
    ) -> Result<Self, Error> {
        let mut client = Client::connect(endpoints, None).await?;
        let lease = client.lease_grant(lease_ttl_secs, None).await?;
        Ok(Self {
            client,
            election: election.into(),
            node_id: node_id.into(),
            lease_id: lease.id(),
        })
    }

    /// The lease id backing this master's leadership.
    pub fn lease_id(&self) -> i64 {
        self.lease_id
    }

    /// Campaign for leadership; resolves once this master IS the leader (etcd
    /// blocks here while another master holds it).
    pub async fn campaign(&mut self) -> Result<Leadership, Error> {
        let resp = self
            .client
            .campaign(
                self.election.as_bytes(),
                self.node_id.as_bytes(),
                self.lease_id,
            )
            .await?;
        let leader_key = resp
            .leader()
            .cloned()
            .expect("a campaign winner always has a leader key");
        Ok(Leadership { leader_key })
    }

    /// The current leader's node id, or `None` if there is no leader yet.
    pub async fn leader(&mut self) -> Result<Option<String>, Error> {
        match self.client.leader(self.election.as_bytes()).await {
            Ok(resp) => Ok(resp
                .kv()
                .map(|kv| String::from_utf8_lossy(kv.value()).into_owned())),
            // etcd returns an error when the election has no leader yet.
            Err(Error::GRpcStatus(status)) if status.message().contains("no leader") => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Keep this master's lease alive (call on an interval shorter than the TTL)
    /// so its leadership persists.
    pub async fn keep_alive(&mut self) -> Result<(), Error> {
        let (mut keeper, _stream) = self.client.lease_keep_alive(self.lease_id).await?;
        keeper.keep_alive().await
    }

    /// Step down from leadership; a standby master is then elected.
    pub async fn resign(&mut self, leadership: Leadership) -> Result<(), Error> {
        self.client
            .resign(Some(
                ResignOptions::new().with_leader(leadership.leader_key),
            ))
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Needs a running etcd:
    //   docker run -d -p 2379:2379 quay.io/coreos/etcd:v3.5.13 etcd \
    //     --advertise-client-urls http://0.0.0.0:2379 --listen-client-urls http://0.0.0.0:2379
    // then: cargo test -p quillcache-store --features etcd -- --ignored
    #[tokio::test]
    #[ignore = "requires a running etcd on 127.0.0.1:2379"]
    async fn two_masters_elect_one_leader_then_fail_over() {
        let endpoints = vec!["http://127.0.0.1:2379".to_string()];
        let election = format!("qc-master-election-{}", std::process::id());

        // Master A joins and campaigns → becomes the leader.
        let mut a = MasterElection::join(endpoints.clone(), election.clone(), "master-a", 10)
            .await
            .expect("a joins");
        let lead_a = a.campaign().await.expect("a campaigns");
        assert_eq!(a.leader().await.unwrap().as_deref(), Some("master-a"));

        // Master B joins; A is still leader.
        let mut b = MasterElection::join(endpoints.clone(), election.clone(), "master-b", 10)
            .await
            .expect("b joins");
        assert_eq!(b.leader().await.unwrap().as_deref(), Some("master-a"));

        // A steps down → B wins the next election (failover). The new leader would
        // rebuild its state from the latest MasterSnapshot.
        a.resign(lead_a).await.expect("a resigns");
        let _lead_b = b.campaign().await.expect("b campaigns");
        assert_eq!(b.leader().await.unwrap().as_deref(), Some("master-b"));
    }
}
