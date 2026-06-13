# Real engine ↔ the KV store (the data plane)

The faithful Mooncake **Store** data path: a real engine offloads/loads KV bytes
to/from the distributed pool via the store `MasterService` + the Transfer Engine.
(For the **control plane** — the gateway routing requests cache-aware — see
[m3-real-vllm.md](m3-real-vllm.md).)

The pieces (all in this repo):

| piece | what it is |
| --- | --- |
| `quillcache store-master` | the `MasterService` over HTTP — two-phase Put, identity-guarded Get, Mount (`src/store_master_http.rs`) |
| `quillcache transfer-node` | a Transfer Engine storage node serving one named RAM segment over the `(segment, offset)` wire (`src/transfer_node.rs`) |
| `bridge/quillcache_store_client.py` | stdlib client: the store master (HTTP) + the transfer wire (TCP) |
| `bridge/quillcache_v1_connector.py` | the **real** vLLM 0.22.1 `KVConnectorBase_V1` on the store (GPU-verified) |
| `bridge/quillcache_store_demo.py` | a no-GPU demo of the store offload/load path |
| `quillcache pd-proxy` | orchestrates prefill → store → decode across two engines (`src/pd_proxy.rs`) |

The flow is Mooncake's: **put** = `put_start` (master allocates replica buffers)
→ WRITE the bytes to each `(segment, offset)` over the transfer engine →
`put_end`; **get** = `get_replica_list` (identity-guarded — refused *before* any
byte moves) → READ a replica over the transfer engine. No object bytes ever flow
through the master.

## Local dry run — no GPU, all real (verified)

Three terminals. This moves **real bytes** through the **real** store; only the
"KV" payload is fake (a GPU supplies the real tensor bytes).

```bash
# 1) a storage node — a Transfer Engine segment served over TCP
cargo run -- transfer-node --addr 127.0.0.1:8100 --segment seg-0

# 2) the store master — metadata, two-phase Put, the identity guard
cargo run -- store-master --addr 127.0.0.1:7777

# 3) the store demo — offload a block, load it back through the store (no GPU)
python3 bridge/quillcache_store_demo.py          # run from the bridge/ dir
#   -> loaded back: b'fake-kv-bytes-over-the-faithful-store'
curl -s http://127.0.0.1:7777/v1/state
#   -> {"objects":1,"segments":1,"capacity":...,"allocated":37}
```

Add more storage nodes (`--addr ... --segment seg-1`, register them in the
connector's `segment_endpoints` + `--replica-num 2`) and a Put replicates across
distinct segments. A Get under a different `tenant_id` is refused with HTTP 403
(the identity guard, over the wire).

## On a GPU (Modal) — the connector is real and verified

The vLLM `KVConnectorBase_V1` implementation (`bridge/quillcache_v1_connector.py`)
is written against the deployed vLLM 0.22.1 API and **verified on a Modal L4**:

- `deploy/modal_vllm_connector_check.py` — 5/5 conformance: vLLM's own
  `KVConnectorFactory` loads the connector via `kv_connector_module_path`.
- `deploy/modal_vllm_connector.py` — single-engine e2e: request-1 offloads a
  496-token prefix to the store (`QC committed`), request-2 reuses it
  (`QC loading`); prefix caching off, so the reuse can only come from the store.
- `deploy/modal_vllm_pd.py` — disaggregated P/D on a 2-GPU box: one request
  through `quillcache pd-proxy` warms the store on GPU 0 (prefill) and reuses it
  on GPU 1 (decode). KV computed on one instance, reused by another, via the store.

The connector keeps vLLM's paged-KV slot-mapping extract/inject verbatim (correct
per attention backend) and swaps disk-safetensors for the identity-guarded store.

## Status — what's real vs what needs hardware

| piece | status |
| --- | --- |
| `store-master` (MasterService over HTTP; **HA**: snapshot recovery + heartbeat + etcd leader election) | **real, tested** — HA verified locally + vs Docker etcd |
| `transfer-node` (Transfer Engine segment server) | **real, tested** (the engine's TCP round-trip) |
| `quillcache_store_client.py` (master HTTP + transfer wire) | **real, tested** (the local e2e above) |
| connector offload / load (two-phase Put, identity-guarded Get) | **real, tested** (the local e2e above) |
| vLLM `KVConnectorBase_V1` connector + KV-tensor (de)serialization | **real, L4-verified** (`deploy/modal_vllm_connector.py`) |
| disaggregated P/D (prefill → store → decode via `pd-proxy`) | **real, 2×L4-verified** (`deploy/modal_vllm_pd.py`) |
| multi-node (store on a separate node) | **code-ready** — `master_url` / `segment_endpoints` are arbitrary `host:port`, and the TCP byte path is identical to localhost; only Modal cross-container plumbing (tunnels) remains |
| RDMA / GPUDirect transfer (vs the TCP wire) | **reserved** — `--features rdma/nvlink`, needs a NIC / multi-GPU |

This mirrors the project's honesty rule: the store, the byte-moving transfer
engine, the connector, master HA, and disaggregated P/D are real and verified
(laptop / Docker etcd / Modal L4); zero-copy RDMA/GPUDirect is the clearly-marked
seam that needs a NIC or multi-GPU.
