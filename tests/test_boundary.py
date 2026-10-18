import ast
import importlib
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class Vec:
    def __init__(self, values):
        self.values = list(values)

    def __eq__(self, other):
        return BoolVec(value == other for value in self.values)

    def __ne__(self, other):
        return BoolVec(value != other for value in self.values)


class BoolVec(Vec):
    def __or__(self, other):
        return BoolVec(a or b for a, b in zip(self.values, other.values))

    def __and__(self, other):
        return BoolVec(a and b for a, b in zip(self.values, other.values))

    def __invert__(self):
        return BoolVec(not value for value in self.values)


class SlotArray:
    def __init__(self, size):
        self.values = [0] * size

    def __getitem__(self, indices):
        return Vec(self.values[index] for index in indices)


class FakeStagedWriteTensor:
    def __init__(self, size, dtype, device):
        self.gpu = SlotArray(size)
        self.pending = []
        self.apply_calls = 0

    def stage_write_elem(self, index, value):
        self.pending.append((index, value))

    def apply_write(self):
        for index, value in self.pending:
            self.gpu.values[index] = value
        self.pending.clear()
        self.apply_calls += 1


class FakeTensor:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    def argmax(self, dim):
        self.assert_dim(dim)
        return Vec(max(range(len(row)), key=row.__getitem__) for row in self.rows)

    def assert_dim(self, dim):
        if dim != -1:
            raise AssertionError(dim)

    def __setitem__(self, key, value):
        mask, column = key
        for enabled, row in zip(mask.values, self.rows):
            if enabled:
                row[column] = value


class FakeLogitsProcessor:
    pass


class BoundaryV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vllm = types.ModuleType("vllm")
        modules = {
            "vllm": vllm,
            "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
            "vllm.v1.worker.gpu": types.ModuleType("vllm.v1.worker.gpu"),
            "vllm.v1.worker.gpu.sample": types.ModuleType("vllm.v1.worker.gpu.sample"),
            "vllm.v1.worker.gpu.sample.logits_processor": types.ModuleType(
                "vllm.v1.worker.gpu.sample.logits_processor"
            ),
            "vllm.v1.worker.gpu.buffer_utils": types.ModuleType(
                "vllm.v1.worker.gpu.buffer_utils"
            ),
            "torch": types.ModuleType("torch"),
        }
        modules["vllm.v1.worker.gpu.sample.logits_processor"].LogitsProcessor = (
            FakeLogitsProcessor
        )
        modules["vllm.v1.worker.gpu.buffer_utils"].StagedWriteTensor = (
            FakeStagedWriteTensor
        )
        modules["torch"].int32 = "int32"

        cls.module_patch = patch.dict(sys.modules, modules)
        cls.module_patch.start()
        sys.modules.pop("simulst_mm.model.boundary", None)
        cls.boundary = importlib.import_module("simulst_mm.model.boundary")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("simulst_mm.model.boundary", None)
        cls.module_patch.stop()

    def config(self, architectures=None, turn_end=3, eos=4):
        hf_config = SimpleNamespace(
            architectures=architectures or [self.boundary.ARCHITECTURE],
            turn_end_token_id=turn_end,
            eos_token_id=eos,
        )
        model_config = SimpleNamespace(hf_config=hf_config)
        return SimpleNamespace(model_config=model_config)

    def req_states(self, vocab_size=6):
        return SimpleNamespace(device="cpu", max_num_reqs=4, vocab_size=vocab_size)

    def test_class_implements_v2_interface(self):
        self.assertTrue(
            issubclass(
                self.boundary.BoundaryGreedyLogitsProcessor,
                FakeLogitsProcessor,
            )
        )
        self.assertFalse(hasattr(self.boundary.BoundaryGreedyLogitsProcessor, "update_state"))

    def test_sampling_slots_and_expanded_rows(self):
        processor = self.boundary.BoundaryGreedyLogitsProcessor(
            self.config(), self.req_states()
        )
        self.assertTrue(processor.add_request(1, SimpleNamespace(temperature=0.7)))
        self.assertFalse(processor.add_request(2, SimpleNamespace(temperature=0.0)))
        processor.apply_staged_writes()

        original = [
            [10, 1, 1, 2, 3, 0],
            [1, 1, 1, 10, 3, 0],
            [10, 1, 1, 2, 3, 0],
            [1, 1, 1, 2, 10, 0],
        ]
        logits = FakeTensor(original)
        ctx = SimpleNamespace(expanded_idx_mapping=[1, 1, 2, 1])
        processor.apply(logits, ctx)

        self.assertEqual(logits.rows[0][3:5], [float("-inf"), float("-inf")])
        self.assertEqual(logits.rows[1], original[1])
        self.assertEqual(logits.rows[2], original[2])
        self.assertEqual(logits.rows[3], original[3])

    def test_recycled_slot_is_cleared(self):
        processor = self.boundary.BoundaryGreedyLogitsProcessor(
            self.config(), self.req_states()
        )
        processor.add_request(1, SimpleNamespace(temperature=0.7))
        processor.apply_staged_writes()
        processor.add_request(1, SimpleNamespace(temperature=0.0))
        processor.apply_staged_writes()

        original = [[10, 1, 1, 2, 3, 0]]
        logits = FakeTensor(original)
        processor.apply(logits, SimpleNamespace(expanded_idx_mapping=[1]))
        self.assertEqual(logits.rows, original)

    def test_foreign_model_is_disabled(self):
        processor = self.boundary.BoundaryGreedyLogitsProcessor(
            self.config(["OtherForCausalLM"]), self.req_states()
        )
        self.assertFalse(processor.add_request(0, SimpleNamespace(temperature=1.0)))
        logits = FakeTensor([[10, 1, 1, 2, 3, 0]])
        self.assertIs(
            processor.apply(logits, SimpleNamespace(expanded_idx_mapping=[0])),
            logits,
        )

    def test_boundary_ids_must_fit_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "outside vocabulary size"):
            self.boundary.BoundaryGreedyLogitsProcessor(
                self.config(turn_end=9), self.req_states(vocab_size=6)
            )

    def test_source_is_frontend_light_and_v2_only(self):
        source_path = (
            Path(__file__).parents[1] / "simulst_mm" / "model" / "boundary.py"
        )
        source = source_path.read_text()
        tree = ast.parse(source)
        top_imports = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        self.assertNotIn("torch", top_imports)
        self.assertNotIn("vllm.v1.sample.logits_processor", source)
        self.assertNotIn("def update_state", source)


if __name__ == "__main__":
    unittest.main()
