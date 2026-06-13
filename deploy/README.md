# Modal deploy + verification scripts

GPU / multi-process runs that verify the parts of QuillCache a laptop can't.
Each is `modal run deploy/<script>.py` (needs the Modal CLI authed).

| script | what it runs | verifies |
| --- | --- | --- |
| `modal_vllm.py` | a single vLLM engine on an L4 (`modal deploy`) | a real OpenAI-compatible engine for the gateway to route to |
| `modal_vllm_introspect.py` | dumps vLLM 0.22.1's `KVConnectorBase_V1` source (CPU) | the exact connector API we implement against (no guessing) |
| `modal_vllm_connector_check.py` | loads `QuillCacheV1Connector` via vLLM's factory (CPU) | 5/5 conformance — vLLM will load our connector |
| `modal_vllm_connector.py` | vLLM + co-located store + the connector (1×L4) | offload → reuse: request-1 commits a prefix, request-2 loads it from the store |
| `modal_vllm_pd.py` | 2 vLLM (GPU0 prefill / GPU1 decode) + store + `pd-proxy` (2×L4) | disaggregated P/D: one request through the proxy warms the store on GPU0, reuses it on GPU1 |
| `modal_cuda_verify.py` | builds `quillcache-cuda` / transfer-engine `--features cuda` (1×L4) | real CUDA: device-tier H2D/D2H + the GPU HBM device segment over the one-sided wire |

The Rust device tier + HBM segment (cudarc 0.19, `dynamic-loading`) compile with
no CUDA toolkit, so CI compile-checks them; the GPU round-trip tests are
`#[ignore]` and run here. The store, transfer engine, connector, master HA, and
P/D are real and verified across laptop / Docker etcd / these Modal runs; only
zero-copy RDMA/GPUDirect (needs a NIC / multi-GPU) stays reserved.
