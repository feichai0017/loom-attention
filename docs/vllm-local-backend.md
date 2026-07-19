# vLLM Local Attention Backend

## Purpose

M1 proves that QuillCache can enter the real vLLM V1 attention call path without
owning model execution or replacing its optimized kernel. The adapter registers
as an out-of-tree `CUSTOM` backend, validates the local tensor contract, and
delegates to vLLM `FlashAttentionImpl` with the same arguments and output tensor.

```text
vLLM model runner
  -> QuillCacheFlashAttentionMetadataBuilder.build
       -> CPU query boundary validation
       -> opaque paged-KV tensor descriptors
       -> generation-checked step snapshot
  -> QuillCacheFlashAttentionImpl.forward
       -> first-call Q/K/V/output layout and device validation
       -> process-local timing and failure accounting
       -> vLLM FlashAttentionImpl.forward
            -> GPU FlashAttention kernel
```

No Q/K/V payload enters the Rust control service. The only tensor values read by
the adapter are `query_start_loc_cpu`, which vLLM already maintains on CPU.
Device-side block tables, slot mappings, sequence lengths, and query offsets
remain opaque. The adapter adds the engine boundary that later route-Q and
split-KV executors will implement.

## Install And Run

The package currently targets vLLM 0.25.x and its V1 attention backend registry.
Install it in the same Python environment as vLLM:

```bash
python3 -m pip install -e './python[vllm]'
```

Load only the QuillCache plugin and select the registered backend:

```bash
VLLM_PLUGINS=quillcache \
  vllm serve MODEL \
  --attention-backend CUSTOM \
  --enforce-eager
```

`--enforce-eager` keeps the M1 validation path outside CUDA Graph capture. Graph
support is intentionally deferred until the remote execution contract is fixed.

## Contract

The first forward checks:

- positive MHA/GQA head counts and valid GQA divisibility;
- flattened or explicit-head Q/K/V/output layouts;
- one device for all non-empty attention tensors;
- a deterministic layout identity from attention type, head counts, head size,
  and KV-cache dtype.

Set `QUILLCACHE_VALIDATE_EVERY_FORWARD=1` for debugging dynamic layouts. The
default validates once per attention implementation to avoid repeated Python
shape walks in vLLM's per-layer critical path. Call count, failures, elapsed
time, layout identity, and last validated device are process-local telemetry;
they are not distributed scheduler evidence.

## Paged-KV Step Snapshot

The custom backend returns a subclass of vLLM's
`FlashAttentionMetadataBuilder`. After the native builder finishes, QuillCache
attaches one immutable `StepMetadataSnapshot` to the resulting metadata. It
contains:

- request count, actual/padded token bounds, maximum query and sequence lengths;
- the existing CPU query-start offsets;
- block size, layer group, head layout, KV dtype, and a layout digest;
- shape, dtype, device, item count, byte bounds, and process-local data pointer
  for the contiguous block table, slot mapping, sequence lengths, and device
  query offsets;
- a monotonically increasing generation, including `update_block_table` calls.

The snapshot deliberately does not contain physical block-table values. Reading
those values in Python would synchronize the GPU. Prefix identity and
`PoolObjectRef` mappings come from QuillCache's page table and pool events; a
later node-local bridge will join those control-plane identities with these
device tensor descriptors.

## Current Validation Boundary

CI tests registration, forwarding, metadata building, block-table replacement,
zero device readback, layout rejection, error propagation, and plugin
idempotence with fake tensor and vLLM modules. A real vLLM installation, CUDA
device, model decode, output-equality check, and performance measurement are
still required before M1 can be declared complete.
