# Loom Kernels documentation

A guided map of the Rust contracts, CUDA backend, framework integrations, and
the evidence used to qualify them.

[Project overview](../README.md) · [Website](https://feichai0017.github.io/loom-kernels/) · [Operator catalog](operator-catalog.md) · [Compatibility](compatibility.md) · [Evidence index](results/README.md)

---

## Choose a path

| Goal | Start here | Then read |
| --- | --- | --- |
| Understand the project | [Operator-library design](design/operator-library.md) | [Operator catalog](operator-catalog.md) |
| Use Rust or CUDA | [Project quick start](../README.md#quick-start) | [Implementation status](status.md) |
| Integrate PyTorch or vLLM | [Python adapter README](../python/README.md) | [vLLM provider guide](guides/vllm-ir-provider.md) |
| Check supported versions | [Compatibility matrix](compatibility.md) | [Implementation status](status.md) |
| Work on paged decode | [Paged-decode design](design/paged-decode-attention.md) | [Paged-decode evidence](results/README.md#paged-decode-attention) |
| Evaluate performance | [Evidence index](results/README.md) | Raw JSON under [`results/`](results/) |
| Pick the next operator | [Roadmap](roadmap.md) | [Catalog implementation order](operator-catalog.md#implementation-order) |

## Core reference

### Product and architecture

- [Operator-library design](design/operator-library.md) defines what Loom owns,
  what remains vendor-backed, and the six admission gates.
- [Operator catalog](operator-catalog.md) is the complete intended inference
  surface with explicit status and priority.
- [Implementation status](status.md) separates implemented code, validation,
  and unresolved work.
- [Compatibility matrix](compatibility.md) separates qualified source,
  framework, GPU, and binary distribution boundaries.
- [Roadmap](roadmap.md) orders work by engine value rather than operator count.
- [Contributing guide](../CONTRIBUTING.md) defines proposal and acceptance
  requirements for new operators.

### Integration

- [Python adapter README](../python/README.md) covers installation, native
  library discovery, and direct PyTorch use.
- [vLLM provider guide](guides/vllm-ir-provider.md) contains the complete vLLM
  0.24/0.25 build, opt-in, fallback, test, and benchmark contracts.
- [Paged-decode design](design/paged-decode-attention.md) documents native KV
  layouts, GQA packing, local split-K/LSE, and routing exclusions.

## What a status means

| State | Meaning |
| --- | --- |
| `supported` | Contract, oracle, CUDA, framework adapter, and H20 evidence exist |
| `next` | Admitted to the immediate implementation queue |
| `planned` | Has a named engine consumer but is ordered later |
| `profile-gated` | Becomes public only after a real workload shows material cost |
| `vendor-backed` | Loom owns dispatch or fusion around a qualified vendor primitive |

## How evidence is read

Loom keeps these claims separate:

1. contract and CPU-oracle correctness;
2. accelerator correctness across edge and representative shapes;
3. warmed measurement against a named operator baseline;
4. framework dispatch and CUDA Graph behavior;
5. invocation from a real engine path;
6. model or serving benefit such as TTFT, TPOT, throughput, memory, or goodput.

Passing one level never implies the next. The [evidence index](results/README.md)
groups accepted, parity, fallback, and rejected experiments without hiding
negative results.

> [!NOTE]
> Only JSON artifacts under [`docs/results`](results/) count as performance
> evidence. A CPU test, successful CUDA launch, or isolated number without a
> named baseline is not a speedup claim.
