# Runtime Control Plane

This document separates runtime control-plane features from simulator-only
experiments.

## What Is Runtime Now

These pieces run in the online gateway path:

- `EngineRole`: each engine can be `aggregated`, `prefill`, or `decode`.
- `ControlPlane::plan`: every request becomes a `RequestPlan`, not just a route.
- `RequestPlan`: records serving mode, execution worker, optional prefill worker,
  decode worker, and per-block planner actions.
- `TieredDataPlane`: an in-process HBM/DRAM/SSD data-plane adapter that performs
  admission, hits, promotion, demotion, eviction, and remove/clear correction.
- `/v1/state`: exposes index metrics, data-plane metrics, data-plane residency,
  action-sink config, engine roles, and index residency.
- `action_sink`: optional synchronous HTTP delivery of planner/data-plane
  actions to a real adapter.
- `x-quillcache-*` response headers: include planner mode, prefill/decode engine
  ids, planner action count, cache action count, local hits, transfers,
  recomputes, estimated TTFT, reuse refusals, and action-sink delivery status.

The simulator still exists for experiments, but PD, planner, and tiered cache are
no longer only simulator concepts.

## PD Serving Roles

`EngineEndpoint.role` defaults to `Aggregated`, so old configs keep working.

```yaml
engines:
  - id: prefill-a
    kind: Vllm
    role: Prefill
    base_url: http://127.0.0.1:8001
    model_id: Qwen/Qwen2.5-7B-Instruct
    tokenizer_id: Qwen/Qwen2.5-7B-Instruct
    tenant_id: default
    locality_domain: rack-a
  - id: decode-a
    kind: Vllm
    role: Decode
    base_url: http://127.0.0.1:8002
    model_id: Qwen/Qwen2.5-7B-Instruct
    tokenizer_id: Qwen/Qwen2.5-7B-Instruct
    tenant_id: default
    locality_domain: rack-a
```

The planner filters decode candidates to `Decode` or `Aggregated` workers. If a
decode-only worker needs recompute work and a prefill-capable worker exists, the
plan becomes `Disaggregated` and includes `RunPrefill` actions. The gateway still
proxies the OpenAI request to `execution_worker_id`; a real vLLM/SGLang
disaggregated connector can consume the prefill/decode plan later.

## Planner Actions

Planner actions are runtime records:

- `UseLocal`: block is already in the target worker's HBM.
- `Fetch`: block should be fetched from another worker/tier.
- `Recompute`: aggregated worker should compute the block.
- `RunPrefill`: prefill worker should compute the block for a decode worker.
- `Decode`: decode worker runs generation.

This is the control-plane surface that LMCache, KVBM, NIXL, or a vLLM
`kv_transfer` adapter should consume.

## Action Sink

The gateway can synchronously POST each request's runtime plan to an external
adapter:

```yaml
action_sink:
  kind: http
  url: http://127.0.0.1:9090/v1/quillcache/actions
  fail_open: true
  timeout_ms: 250
```

The sink receives two event phases:

- `planned`: sent before the OpenAI request is proxied to the selected engine.
  It contains `RequestShape`, route summary, and full `PlanAction` records.
- `committed`: sent after a successful upstream response and after
  `observe_placement` updates the tiered data plane. It includes any
  `DataPlaneAction` records such as `Admit`, `Promote`, `Demote`, and `Evict`.

`fail_open: true` makes the gateway continue when the adapter is down and marks
the response with `x-quillcache-action-sink: failed`. `fail_open: false` turns a
sink failure into `502`, which is the mode for an adapter that must complete
prefill/fetch before decode can run.

For local inspection:

```bash
python3 tools/action_sink_mock.py --port 9090
```

The mock prints a compact line for every planned/committed event. A production
adapter should map `Fetch` to tensor transfer, `RunPrefill` to prefill-side
execution, `Decode` to decode-side scheduling, and committed cache actions to
admission/eviction calls in LMCache, KVBM, or another KV data plane.

## Tiered Data Plane

Enable the in-process tiered data plane in gateway config:

```yaml
data_plane:
  kind: tiered
  hbm_capacity_bytes: 85899345920
  cpu_dram_capacity_bytes: 549755813888
  local_ssd_capacity_bytes: 1099511627776
```

When a request succeeds, the gateway calls `ControlPlane::observe_placement`.
With `data_plane: tiered`, that call admits/touches blocks in the tiered data
plane, performs promotion/demotion/eviction, and mirrors the final tier state
back into the `IndexBackend`.

When real KV events arrive:

- `BlockStored` updates both the index and the tiered data plane.
- `BlockRemoved` removes stale inferred/tiered state.
- `AllBlocksCleared` clears the worker.

The index remains the metadata catalog. The data plane is the runtime tier
manager. A production adapter can replace `TieredDataPlane` with LMCache,
Dynamo KVBM, FlexKV, or vLLM offload semantics.

## Current Boundary

The online gateway now has real planner, tiered-cache state transitions, and a
synchronous action-sink API. It does not move vLLM/SGLang KV tensors inside the
Rust gateway. Tensor movement and engine-specific prefill/decode handoff remain
adapter responsibilities, attached through the action-sink event stream and
corrected through `/v1/kv-events`.
