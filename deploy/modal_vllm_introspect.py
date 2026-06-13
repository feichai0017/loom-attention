"""Dump the EXACT KV-connector API of the vLLM version deployed on Modal.

The connector must subclass vLLM's real `KVConnectorBase_V1`, whose signatures
are version-specific and (for vllm 0.22.1) postdate our knowledge. So instead of
guessing, pull the ground-truth source from the same image the engines run:

    modal run deploy/modal_vllm_introspect.py

CPU-only, no GPU, no model download — it just reads the installed .py files
(never imports torch/cuda), so it's cheap and independent of the live engines.
"""

import modal

# Same vLLM as the serving image (unpinned -> latest, currently 0.22.1).
image = modal.Image.debian_slim(python_version="3.12").pip_install("vllm")
app = modal.App("quillcache-vllm-introspect")


@app.function(image=image, timeout=600)
def dump():
    import glob
    import os
    import re
    import sysconfig

    sp = sysconfig.get_paths()["purelib"]
    root = os.path.join(sp, "vllm")

    # The whole kv_transfer subtree (base class + shipped connectors as references).
    files = sorted(
        glob.glob(os.path.join(root, "distributed/kv_transfer/**/*.py"), recursive=True)
    )
    # Anything else in the tree that defines a KVConnector class (in case the
    # path moved in 0.22.1).
    extra = []
    for f in glob.glob(os.path.join(root, "**/*.py"), recursive=True):
        if f in files:
            continue
        try:
            head = open(f, "r", errors="replace").read(4000)
        except OSError:
            continue
        if re.search(r"class \w*KVConnector", head):
            extra.append(f)

    version = "unknown"
    try:
        version = open(os.path.join(root, "version.py")).read()
    except OSError:
        pass

    out = {"__version__": version}
    for f in sorted(set(files + extra)):
        try:
            out[f.replace(sp + "/", "")] = open(f, "r", errors="replace").read()
        except OSError as e:
            out[f.replace(sp + "/", "")] = f"<unreadable: {e}>"
    return out


@app.local_entrypoint()
def main():
    res = dump.remote()
    # version banner first
    print("#### vllm/version.py ####")
    print(res.pop("__version__", "?"))
    # then every connector source file, base.py first
    def sort_key(name):
        return (0 if name.endswith("/v1/base.py") else 1, name)

    for name in sorted(res, key=sort_key):
        print("\n" + "=" * 100)
        print(name)
        print("=" * 100)
        print(res[name])
