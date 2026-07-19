# QuillCache

QuillCache is an external-KV-pool-native distributed core-attention runtime for
LLM serving. vLLM or SGLang keeps model execution; Mooncake, LMCache, or another
`KvPool` owns sealed KV objects; QuillCache schedules attention near those
objects and merges exact partial-softmax results.

This repository made an intentional breaking architecture change at `v2`.
The previous built-in KV store, byte-oriented transfer engine, OpenAI proxy,
P/D demos, and storage connector were removed. They mixed storage ownership
with attention execution and remain available in Git history.

## Boundary

| Component | Owns |
| --- | --- |
| vLLM / SGLang | batching, weights, QKV projection, RoPE, FFN, sampling |
| QuillCache | sequence page tables, read leases, attention plans, partial attention and merge |
| external `KvPool` | sealed KV allocation, placement, replication, eviction, durability |
| Holt catalog | persistent `prefix -> PoolObjectRef` metadata, revalidated after recovery |

The control service is a slow path. Per-layer execution uses node-local state and
never synchronously queries the gateway or Holt.

## Workspace

| Package | Responsibility |
| --- | --- |
| `quillcache-core` | `types`, `pool`, `scheduler`, `attention`, `runtime`, and `transport` modules |
| `quillcache-catalog` | control-plane hot directory and Holt persistent catalog |
| `quillcache-control` | global catalog/scheduler service |
| `quillcache-attention-worker` | node attention-worker control endpoint |
| `python/quillcache_engine` | out-of-tree vLLM attention backend adapters |

## Implemented

- dependency-clean v2 workspace and service split;
- external `KvPool` API with object generations, ordered events, and read leases;
- Holt-backed persistent object catalog plus worker-epoch-aware hot directory;
- SLO cost comparison for local, route-query, KV-stage, and sharded execution;
- exact split-KV online-softmax merge with reference correctness tests;
- node-local single-writer step state, lease validation, and active-tail sealing;
- handle-based tensor transport API with registered-region bounds checks;
- control-only service endpoints for the controller and attention worker;
- vLLM `CUSTOM` backend that validates the local tensor contract and delegates
  unchanged execution to vLLM FlashAttention;
- vLLM metadata-builder wrapper that records generation-checked paged-KV tensor
  descriptors without reading device tensor contents.

## Not Implemented Yet

- SGLang and remote-execution attention backend adapters;
- CUDA partial-attention and merge kernels;
- CUDA IPC, NCCL, NIXL, or GPUDirect RDMA transports;
- a production Mooncake adapter;
- remote GPU end-to-end latency or throughput claims.

## Build

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
PYTHONPATH=python python3 -m unittest discover -s python/tests -v
```

Run the control endpoints:

```bash
cargo run -p quillcache-control -- --bind 127.0.0.1:8080
cargo run -p quillcache-attention-worker -- --bind 127.0.0.1:8090
```

See [architecture](docs/architecture.md), [platform plan](docs/platform-plan.md),
[vLLM local backend](docs/vllm-local-backend.md), and
[protocol invariants](docs/protocols.md).

## License

MIT
