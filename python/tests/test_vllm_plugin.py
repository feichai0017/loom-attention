from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from quillcache_engine import vllm_plugin
from test_local_delegate import FakeTensor


class FakeFlashAttentionImpl:
    def __init__(self, *args, **kwargs) -> None:
        self.delegate_calls = 0

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        self.delegate_calls += 1
        if layer == "fail":
            raise RuntimeError("delegate failure")
        return output


class FakeFlashAttentionBackend:
    @staticmethod
    def get_builder_cls():
        return object


class VllmPluginTest(unittest.TestCase):
    def setUp(self) -> None:
        vllm_plugin._REGISTERED = False
        for name in (
            "QuillCacheFlashAttentionBackend",
            "QuillCacheFlashAttentionImpl",
        ):
            vllm_plugin.__dict__.pop(name, None)

    def fake_modules(self, registrations: list[tuple[object, str]]) -> dict[str, ModuleType]:
        packages = {
            name: ModuleType(name)
            for name in (
                "vllm",
                "vllm.v1",
                "vllm.v1.attention",
                "vllm.v1.attention.backends",
            )
        }
        flash = ModuleType("vllm.v1.attention.backends.flash_attn")
        flash.FlashAttentionBackend = FakeFlashAttentionBackend
        flash.FlashAttentionImpl = FakeFlashAttentionImpl

        registry = ModuleType("vllm.v1.attention.backends.registry")
        custom = object()
        registry.AttentionBackendEnum = SimpleNamespace(CUSTOM=custom)

        def register_backend(backend, class_path=None):
            registrations.append((backend, class_path))
            return lambda value: value

        registry.register_backend = register_backend
        packages[flash.__name__] = flash
        packages[registry.__name__] = registry
        return packages

    def tensors(self) -> dict[str, FakeTensor]:
        return {
            "query": FakeTensor((4, 8 * 64)),
            "key": FakeTensor((4, 2 * 64)),
            "value": FakeTensor((4, 2 * 64)),
            "kv_cache": FakeTensor((2, 32, 16, 2, 64)),
            "output": FakeTensor((4, 8 * 64)),
        }

    def test_registers_custom_backend_and_delegates_unchanged_output(self) -> None:
        registrations = []
        with patch.dict(sys.modules, self.fake_modules(registrations), clear=False):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("QUILLCACHE_VLLM_DELEGATE", None)
                vllm_plugin.register()

        self.assertEqual(len(registrations), 1)
        self.assertEqual(
            registrations[0][1],
            "quillcache_engine.vllm_plugin.QuillCacheFlashAttentionBackend",
        )
        implementation = vllm_plugin.QuillCacheFlashAttentionBackend.get_impl_cls()(
            num_heads=8,
            head_size=64,
            scale=0.125,
            num_kv_heads=2,
            kv_cache_dtype="bfloat16",
            attn_type="decoder",
        )
        tensors = self.tensors()
        result = implementation.forward(
            "layer", attn_metadata=object(), **tensors
        )
        self.assertIs(result, tensors["output"])
        self.assertEqual(implementation.delegate_calls, 1)
        self.assertEqual(implementation.quillcache_observer.snapshot().calls, 1)
        self.assertFalse(implementation.quillcache_observer.validate_every_call)

    def test_registration_is_idempotent(self) -> None:
        registrations = []
        with patch.dict(sys.modules, self.fake_modules(registrations), clear=False):
            vllm_plugin.register()
            vllm_plugin.register()
        self.assertEqual(len(registrations), 1)

    def test_delegate_failure_is_recorded_and_propagated(self) -> None:
        registrations = []
        with patch.dict(sys.modules, self.fake_modules(registrations), clear=False):
            vllm_plugin.register()
        implementation = vllm_plugin.QuillCacheFlashAttentionBackend.get_impl_cls()(
            8, 64, 0.125, 2
        )
        with self.assertRaisesRegex(RuntimeError, "delegate failure"):
            implementation.forward("fail", attn_metadata=object(), **self.tensors())
        snapshot = implementation.quillcache_observer.snapshot()
        self.assertEqual(snapshot.calls, 1)
        self.assertEqual(snapshot.failures, 1)

    def test_rejects_unimplemented_delegate(self) -> None:
        registrations = []
        with patch.dict(sys.modules, self.fake_modules(registrations), clear=False):
            with patch.dict(
                os.environ, {"QUILLCACHE_VLLM_DELEGATE": "triton"}, clear=False
            ):
                with self.assertRaisesRegex(RuntimeError, "supports only"):
                    vllm_plugin.register()

    def test_rejects_invalid_validation_mode(self) -> None:
        registrations = []
        with patch.dict(sys.modules, self.fake_modules(registrations), clear=False):
            with patch.dict(
                os.environ,
                {"QUILLCACHE_VALIDATE_EVERY_FORWARD": "sometimes"},
                clear=False,
            ):
                vllm_plugin.register()
                implementation_class = (
                    vllm_plugin.QuillCacheFlashAttentionBackend.get_impl_cls()
                )
                with self.assertRaisesRegex(RuntimeError, "must be a boolean"):
                    implementation_class(8, 64, 0.125, 2)


if __name__ == "__main__":
    unittest.main()
