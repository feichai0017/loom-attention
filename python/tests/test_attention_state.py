import unittest
from unittest.mock import patch

from loom_attention.attention_state import (
    compute_attention_state,
    merge_attention_states,
)


class FakeTensor:
    def __init__(self, name: str) -> None:
        self.name = name

    def contiguous(self):
        return self

    def float(self):
        return self


class FakeTorch:
    def __init__(self) -> None:
        self.stack_dimensions = []

    def stack(self, values, dim=0):
        self.stack_dimensions.append((len(values), dim))
        return FakeTensor("stack")


class FakeFlashInfer:
    def __init__(self) -> None:
        self.decode_calls = []
        self.merge_calls = []

    def single_decode_with_kv_cache(self, query, key, value, **kwargs):
        self.decode_calls.append((query, key, value, kwargs))
        return FakeTensor("output"), FakeTensor("lse")

    def merge_states(self, output, logsumexp):
        self.merge_calls.append((output, logsumexp))
        return FakeTensor("merged-output"), FakeTensor("merged-lse")


class AttentionStateContractTest(unittest.TestCase):
    def test_flashinfer_uses_native_state_and_merge_contract(self) -> None:
        torch = FakeTorch()
        flashinfer = FakeFlashInfer()
        query = [FakeTensor("q0"), FakeTensor("q1")]
        key = FakeTensor("key")
        value = FakeTensor("value")
        with patch(
            "loom_attention.attention_state._load_flashinfer",
            return_value=flashinfer,
        ):
            state = compute_attention_state(
                torch,
                query,
                key,
                value,
                kv_heads=1,
                scale=0.125,
                backend="flashinfer",
            )
            merged = merge_attention_states(
                torch, [state, state], backend="flashinfer"
            )

        self.assertEqual(len(flashinfer.decode_calls), 2)
        self.assertEqual(
            flashinfer.decode_calls[0][3],
            {
                "kv_layout": "NHD",
                "pos_encoding_mode": "NONE",
                "sm_scale": 0.125,
                "return_lse": True,
            },
        )
        self.assertEqual(
            torch.stack_dimensions,
            [(2, 0), (2, 0), (2, 1), (2, 1)],
        )
        self.assertEqual(len(flashinfer.merge_calls), 1)
        self.assertEqual(merged[0].name, "merged-output")
        self.assertEqual(merged[1].name, "merged-lse")


if __name__ == "__main__":
    unittest.main()
