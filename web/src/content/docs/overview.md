---
title: Overview
description: What QuillCache is, and what's wired online vs a tested unit vs reserved.
---

QuillCache is a **faithful Rust port of [Mooncake](https://github.com/kvcache-ai/Mooncake)'s
distributed KV cache store** (the KVCache-centric data plane from Moonshot / Kimi,
FAST'25) — its component decomposition, code layout, and API mirrored module for
module — **plus two properties the production data planes leave implicit:
identity-governed safe reuse and a crash-consistent persistent tier.** It sits
beside real inference engines (vLLM, SGLang) and owns the KV cache as a resource:

- a **Transfer Engine** (`quillcache-transfer-engine`) — moves bytes one-sidedly
  between *registered memory* by `(segment, offset)`, exactly like Mooncake (TCP
  today; RDMA / GPUDirect reserved behind the same trait);
- a **Store** (`quillcache-store`) — a two-phase-Put `Client`, a `MasterService`
  (object metadata, replica allocation, lease eviction), a buffer allocator, the
  replica model, and a crash-consistent durable `DiskTier`;
- a **Gateway / Conductor** — an OpenAI-compatible proxy that routes cache-aware
  (the Dynamo KV-router cost function), governs reuse, and meters SLO, backed by a
  persistent residency index.

:::note[It does not run models]
No transformer kernels, no attention. The CUDA tier moves and quantizes KV
*bytes* (the data path), not inference compute.
:::

## Status — wired online vs tested unit vs reserved

Everything here is real code — there is no simulation. The honest distinction is
how far each piece is integrated:

- **✅ wired online & measured** — the gateway, control plane, Dynamo-cost
  routing, persistent residency index, `StoreDataPlane` moving real bytes across
  HBM/DRAM/SSD, live SLO goodput, and the ART-vs-LSM storage study.
- **◑ GPU-verified on Modal (the engine ⟷ store path)** — a real vLLM 0.22.1
  `KVConnectorBase_V1` (`bridge/quillcache_v1_connector.py`) offloads a prompt's KV
  to the store and **reuses** it on a later request (L4); **disaggregated
  prefill/decode** runs across two engines sharing one store via `quillcache
  pd-proxy` — both content-addressed reuse *and* true vLLM-native P/D
  (`kv_producer`/`kv_consumer` with a `transfer_id` handshake, output matching a
  monolithic run token-for-token) on 2×L4. This exercises the faithful store
  (`Client` two-phase Put, `MasterService` + identity guard, transfer engine) and
  master **HA** (etcd leader election, verified vs Docker etcd) end to end.
- **⊙ reserved / needs hardware** — `RdmaTransport` / GPUDirect zero-copy (behind
  the `rdma` / `nvlink` features): real interfaces, stubbed so the default build
  stays hardware-free. (The CUDA device tier is also GPU-verified —
  `quillcache-cuda --features cuda` on an L4 — but stays out of the default
  workspace.)

`cargo test` — 79 tests pass; `cargo fmt --check` and `cargo clippy` are clean.

## Differentiation

The reference designs (Mooncake / Dynamo / LMCache / KVBM) key reuse on a
block's **content hash** (Mooncake adds only a tenant scope) and leave byte-tier
crash-integrity to the caller. QuillCache adds:

1. **Identity-governed safe reuse** — a block is served only when the requester's
   model · tokenizer · adapter · tenant matches, so cross-tenant leaks and
   cross-adapter/model errors are refused. The same guard runs at every serving
   point — `LocalKvStore::get` and `DiskTier::get` (the byte tiers) and
   `MasterService::get_replica_list` (the metadata layer, before any byte moves).
   See [Identity-safe reuse](/identity-safe-reuse/).
2. **A crash-consistent persistent tier** — a durable `DiskTier` survives a
   restart with object-first atomic publish + a WAL, so durable blocks are
   immediately reusable and corrupt/half-written ones are never served (Mooncake's
   byte tier trusts on-disk files by size on recovery). See [Crash-consistent tier](/crash-consistency/).
