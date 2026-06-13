"""Stage A — prove QuillCacheV1Connector conforms to the DEPLOYED vllm 0.22.1.

Runs the exact code path `--kv-transfer-config` triggers: import the external
module, resolve the class, run the factory's external-connector checks. CPU-only,
no GPU, no model download, no store — just "will vllm 0.22.1 load this connector?"

    modal run deploy/modal_vllm_connector_check.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm")
    # the connector + its store client, importable as top-level modules
    .add_local_file("bridge/quillcache_v1_connector.py", "/root/quillcache_v1_connector.py", copy=True)
    .add_local_file("bridge/quillcache_store_client.py", "/root/quillcache_store_client.py", copy=True)
)
app = modal.App("quillcache-vllm-connector-check")


@app.function(image=image, timeout=600)
def check():
    import sys

    sys.path.insert(0, "/root")
    results = []

    def step(name, fn):
        try:
            detail = fn()
            results.append((name, True, detail))
        except Exception as e:
            import traceback

            results.append((name, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))

    # 1) The connector module imports under the real vllm 0.22.1 (every vllm
    #    symbol the connector references must resolve in this version).
    def s1():
        import quillcache_v1_connector as m

        return f"imported; class={m.QuillCacheV1Connector.__name__}"

    step("import connector module under vllm 0.22.1", s1)

    # 2) It is a CONCRETE subclass of the real base — no abstract method left out.
    def s2():
        import quillcache_v1_connector as m
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

        cls = m.QuillCacheV1Connector
        assert issubclass(cls, KVConnectorBase_V1), "not a KVConnectorBase_V1 subclass"
        missing = sorted(getattr(cls, "__abstractmethods__", frozenset()))
        assert not missing, f"unimplemented abstract methods: {missing}"
        return "concrete subclass; all abstract methods implemented"

    step("concrete KVConnectorBase_V1 (no missing abstract methods)", s2)

    # 3) The factory's external-connector contract: 3-arg ctor incl. kv_cache_config.
    def s3():
        import quillcache_v1_connector as m
        from vllm.utils.func_utils import supports_kw

        assert supports_kw(m.QuillCacheV1Connector, "kv_cache_config"), "ctor missing kv_cache_config"
        return "constructor accepts kv_cache_config (external v1 contract)"

    step("constructor signature accepted by the factory", s3)

    # 4) The EXACT resolution path `--kv-transfer-config` uses: build a real
    #    KVTransferConfig and ask the factory for the class by module path.
    def s4():
        import quillcache_v1_connector as m
        from vllm.config.kv_transfer import KVTransferConfig
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

        kt = KVTransferConfig(
            kv_connector="QuillCacheV1Connector",
            kv_connector_module_path="quillcache_v1_connector",
            kv_role="kv_both",
        )
        resolved = KVConnectorFactory.get_connector_class(kt)
        assert resolved is m.QuillCacheV1Connector, f"factory resolved {resolved}"
        return "vllm factory resolves QuillCacheV1Connector via kv_connector_module_path"

    step("vllm KVConnectorFactory resolves it (the --kv-transfer-config path)", s4)

    # 5) The paged-KV slot-mapping math (pure torch, no GPU) produces the right shape.
    def s5():
        import quillcache_v1_connector as m

        rm = m.ReqMeta.make_meta(
            token_ids=list(range(40)),
            block_ids=[0, 1, 2],
            block_size=16,
            is_store=True,
            prefix_hash="deadbeef",
        )
        # align_to_block_size(40,16) = (39//16)*16 = 32 tokens covered.
        assert rm.slot_mapping.numel() == 32, f"slot_mapping len={rm.slot_mapping.numel()}"
        # block 0 -> slots 0..15, block 1 -> 16..31 (contiguous within block).
        assert rm.slot_mapping[0].item() == 0, f"slot[0]={rm.slot_mapping[0].item()}"
        assert rm.slot_mapping[16].item() == 16, f"slot[16]={rm.slot_mapping[16].item()}"
        h, aligned = m._prefix_hash(list(range(40)), 16, [])
        assert aligned == 32, f"aligned={aligned}"
        assert isinstance(h, str) and len(h) >= 16, f"hash={h!r}"
        return f"slot_mapping=32 tokens, prefix_hash ok ({h[:12]}… len={len(h)})"

    step("paged-KV slot-mapping + prefix-hash math", s5)

    return results


@app.local_entrypoint()
def main():
    results = check.remote()
    print("\n=== QuillCacheV1Connector vs vllm 0.22.1 — conformance ===")
    ok = 0
    for name, passed, detail in results:
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}")
        if passed:
            print(f"       {detail}")
            ok += 1
        else:
            # show only the first 2 lines of the traceback inline
            for line in str(detail).splitlines()[:6]:
                print(f"       {line}")
    print(f"\n{ok}/{len(results)} checks passed")
