import unittest

from loom_attention.paged_executor import (
    FlashInferPagedExecutor,
    PagedKvContractError,
    PagedKvView,
)


class FakeTensor:
    def __init__(
        self,
        shape,
        *,
        dtype="torch.float16",
        device="cuda:0",
        contiguous=True,
        name="tensor",
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._contiguous = contiguous
        self.name = name

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        self._contiguous = True
        return self

    def float(self):
        self.dtype = "torch.float32"
        return self


class FakeTorch:
    uint8 = "torch.uint8"

    def __init__(self) -> None:
        self.zeros_calls = []

    def zeros(self, size, *, dtype, device):
        self.zeros_calls.append((size, dtype, device))
        return FakeTensor((size,), dtype=dtype, device=device, name="workspace")


class FakeWrapper:
    def __init__(self, workspace, layout, **kwargs) -> None:
        self.workspace = workspace
        self.layout = layout
        self.options = kwargs
        self.plan_calls = []
        self.run_calls = []

    def plan(self, *args, **kwargs):
        self.plan_calls.append((args, kwargs))

    def run(self, query, cache, **kwargs):
        self.run_calls.append((query, cache, kwargs))
        return (
            FakeTensor(query.shape, device=query.device, name="output"),
            FakeTensor(
                query.shape[:2],
                dtype="torch.float32",
                device=query.device,
                name="lse",
            ),
        )


class FakeDecode:
    def __init__(self) -> None:
        self.wrappers = []

    def BatchDecodeWithPagedKVCacheWrapper(self, *args, **kwargs):
        wrapper = FakeWrapper(*args, **kwargs)
        self.wrappers.append(wrapper)
        return wrapper


class FakeFlashInfer:
    def __init__(self) -> None:
        self.decode = FakeDecode()


def make_view(*, generation=1, table_id="request-1/layer-0", device="cuda:0"):
    return PagedKvView(
        table_id=table_id,
        page_table_generation=generation,
        lease_ids=("lease-1",),
        indptr=FakeTensor((3,), dtype="torch.int32", device=device),
        indices=FakeTensor((8,), dtype="torch.int32", device=device),
        last_page_len=FakeTensor((2,), dtype="torch.int32", device=device),
        page_size=16,
    )


class PagedExecutorContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.torch = FakeTorch()
        self.flashinfer = FakeFlashInfer()
        self.executor = FlashInferPagedExecutor(
            workspace_bytes=1024,
            torch_module=self.torch,
            flashinfer_module=self.flashinfer,
        )
        self.query = FakeTensor((2, 32, 128))
        self.cache = FakeTensor((8, 2, 16, 8, 128))

    def execute(self, view=None):
        return self.executor.execute(
            self.query,
            self.cache,
            view or make_view(),
            kv_heads=8,
            scale=0.125,
        )

    def test_reuses_workspace_wrapper_and_plan_for_same_generation(self) -> None:
        first = self.execute()
        second = self.execute()

        self.assertEqual(self.torch.zeros_calls, [(1024, "torch.uint8", "cuda:0")])
        self.assertEqual(len(self.flashinfer.decode.wrappers), 1)
        wrapper = self.flashinfer.decode.wrappers[0]
        self.assertEqual(len(wrapper.plan_calls), 1)
        self.assertEqual(len(wrapper.run_calls), 2)
        self.assertEqual(wrapper.run_calls[0][2], {"return_lse": True})
        self.assertEqual(first[0].name, "output")
        self.assertEqual(second[1].dtype, "torch.float32")

    def test_generation_and_table_identity_invalidate_the_plan(self) -> None:
        self.execute(make_view(generation=1))
        self.execute(make_view(generation=2))
        self.execute(make_view(generation=2, table_id="request-2/layer-0"))

        wrapper = self.flashinfer.decode.wrappers[0]
        self.assertEqual(len(wrapper.plan_calls), 3)
        self.assertEqual(self.executor.plan_count, 3)

    def test_plan_uses_current_flashinfer_keyword_contract(self) -> None:
        self.execute()
        args, kwargs = self.flashinfer.decode.wrappers[0].plan_calls[0]
        self.assertEqual(args[3:7], (32, 8, 128, 16))
        self.assertEqual(
            kwargs,
            {
                "pos_encoding_mode": "NONE",
                "q_data_type": "torch.float16",
                "kv_data_type": "torch.float16",
                "sm_scale": 0.125,
            },
        )

    def test_rejects_unpinned_or_unleased_view(self) -> None:
        for view in (
            make_view(generation=0),
            PagedKvView(
                table_id="request-1/layer-0",
                page_table_generation=1,
                lease_ids=(),
                indptr=FakeTensor((3,), dtype="torch.int32"),
                indices=FakeTensor((8,), dtype="torch.int32"),
                last_page_len=FakeTensor((2,), dtype="torch.int32"),
                page_size=16,
            ),
        ):
            with self.assertRaises(PagedKvContractError):
                self.execute(view)
        self.assertEqual(len(self.flashinfer.decode.wrappers), 0)

    def test_rejects_bad_page_table_dtype_and_device(self) -> None:
        bad_dtype = make_view()
        object.__setattr__(bad_dtype, "indices", FakeTensor((8,), dtype="torch.int64"))
        with self.assertRaisesRegex(PagedKvContractError, "indices must use"):
            self.execute(bad_dtype)

        bad_device = make_view(device="cuda:1")
        with self.assertRaisesRegex(PagedKvContractError, "query device"):
            self.execute(bad_device)

    def test_rejects_cache_shape_mismatch(self) -> None:
        self.cache = FakeTensor((8, 2, 8, 16, 128))
        with self.assertRaisesRegex(PagedKvContractError, "invalid NHD shape"):
            self.execute()


if __name__ == "__main__":
    unittest.main()
