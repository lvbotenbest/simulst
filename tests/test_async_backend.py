import concurrent.futures
import sys
import threading
import time
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simulst_mm.serving.async_backend import AsyncTurnRunner


class FakeOutput:
    def __init__(self, request_id, value, *, finished=True):
        self.request_id = request_id
        self.value = value
        self.finished = finished


class FakeEngine:
    def __init__(self, *, abort_error=None, shutdown_error=None):
        self.abort_error = abort_error
        self.shutdown_error = shutdown_error
        self.calls = []
        self.aborted = []
        self.shutdown_calls = 0
        self.started = threading.Event()

    async def generate(self, prompt, params, request_id):
        import asyncio

        self.calls.append((prompt, params, request_id))
        self.started.set()
        await asyncio.sleep(prompt.get("delay", 0))
        if error := prompt.get("error"):
            raise error
        output_id = prompt.get("output_id", request_id)
        yield FakeOutput(output_id, prompt.get("value"))

    async def abort(self, request_id):
        self.aborted.append(request_id)
        if self.abort_error is not None:
            raise self.abort_error

    def shutdown(self):
        self.shutdown_calls += 1
        if self.shutdown_error is not None:
            raise self.shutdown_error


class AsyncTurnRunnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vllm = types.ModuleType("vllm")
        sampling = types.ModuleType("vllm.sampling_params")

        class RequestOutputKind:
            FINAL_ONLY = "final-only"

        sampling.RequestOutputKind = RequestOutputKind
        cls.modules = {"vllm": vllm, "vllm.sampling_params": sampling}
        cls.RequestOutputKind = RequestOutputKind

    def runner(self, engine, **kwargs):
        patcher = patch.dict(sys.modules, self.modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        runner = AsyncTurnRunner(lambda: engine, **kwargs)
        self.addCleanup(self.close_quietly, runner)
        return runner

    @staticmethod
    def close_quietly(runner):
        try:
            runner.close()
        except Exception:
            pass

    def test_independent_requests_can_finish_out_of_order(self):
        engine = FakeEngine()
        runner = self.runner(engine)
        params = SimpleNamespace(output_kind="original", temperature=0)
        completed = []

        def submit(prompt):
            result = runner.submit(prompt, params)
            completed.append(result.value)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            slow = pool.submit(submit, {"delay": 0.05, "value": "slow"})
            fast = pool.submit(submit, {"delay": 0.0, "value": "fast"})
            self.assertEqual(fast.result().value, "fast")
            self.assertEqual(slow.result().value, "slow")

        self.assertEqual(completed, ["fast", "slow"])
        self.assertEqual(params.output_kind, "original")
        self.assertTrue(
            all(call[1].output_kind == self.RequestOutputKind.FINAL_ONLY for call in engine.calls)
        )
        self.assertEqual(len({call[2] for call in engine.calls}), 2)
        self.assertEqual(
            runner.stats(),
            {
                "async_turn_submitted": 2,
                "async_turn_active": 0,
                "async_turn_completed": 2,
                "async_turn_failed": 0,
                "async_turn_cancelled": 0,
            },
        )

    def test_wrong_request_id_is_isolated_and_aborted(self):
        engine = FakeEngine()
        runner = self.runner(engine)
        with self.assertRaisesRegex(RuntimeError, "different request ID"):
            runner.submit(
                {"output_id": "wrong"}, SimpleNamespace(output_kind=None)
            )
        self.assertEqual(len(engine.aborted), 1)
        self.assertEqual(runner.stats()["async_turn_failed"], 1)

    def test_abort_failure_does_not_mask_original_error(self):
        engine = FakeEngine(abort_error=RuntimeError("abort failed"))
        runner = self.runner(engine)
        with self.assertRaisesRegex(ValueError, "model failed"):
            runner.submit(
                {"error": ValueError("model failed")},
                SimpleNamespace(output_kind=None),
            )
        self.assertEqual(len(engine.aborted), 1)

    def test_timeout_aborts_only_that_request(self):
        engine = FakeEngine()
        runner = self.runner(engine, timeout_seconds=0.01)
        with self.assertRaises(TimeoutError):
            runner.submit({"delay": 1}, SimpleNamespace(output_kind=None))
        self.assertEqual(len(engine.aborted), 1)
        self.assertEqual(runner.stats()["async_turn_active"], 0)

    def test_cancel_owner_does_not_cancel_other_owner(self):
        engine = FakeEngine()
        runner = self.runner(engine)
        params = SimpleNamespace(output_kind=None)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                runner.submit, {"delay": 1, "value": "a"}, params, owner="a"
            )
            engine.started.wait(timeout=1)
            second = pool.submit(
                runner.submit, {"delay": 0, "value": "b"}, params, owner="b"
            )
            runner.cancel_owner("a")
            self.assertEqual(second.result().value, "b")
            with self.assertRaises(concurrent.futures.CancelledError):
                first.result()
        self.assertEqual(runner.stats()["async_turn_cancelled"], 1)

    def test_pre_cancelled_request_never_reaches_engine(self):
        engine = FakeEngine()
        runner = self.runner(engine)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(concurrent.futures.CancelledError):
            runner.submit(
                {}, SimpleNamespace(output_kind=None), cancel_event=cancelled
            )
        self.assertEqual(engine.calls, [])

    def test_close_rejects_new_submissions_and_stops_thread(self):
        engine = FakeEngine()
        runner = self.runner(engine)
        params = SimpleNamespace(output_kind=None)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            request = pool.submit(runner.submit, {"delay": 1}, params)
            engine.started.wait(timeout=1)
            closing = pool.submit(runner.close)
            for _ in range(1000):
                with runner._lifecycle:
                    if runner._state != "open":
                        break
            with self.assertRaisesRegex(RuntimeError, "runner is closed"):
                runner.submit({}, params)
            closing.result()
            with self.assertRaises(concurrent.futures.CancelledError):
                request.result()
        self.assertFalse(runner.thread.is_alive())
        self.assertEqual(engine.shutdown_calls, 1)

    def test_concurrent_close_shuts_engine_down_once(self):
        engine = FakeEngine()
        runner = self.runner(engine)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _: runner.close(), range(2)))
        self.assertEqual(engine.shutdown_calls, 1)
        self.assertFalse(runner.thread.is_alive())

    def test_shutdown_error_is_reported_after_thread_cleanup(self):
        engine = FakeEngine(shutdown_error=RuntimeError("shutdown failed"))
        runner = self.runner(engine)
        with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
            runner.close()
        self.assertFalse(runner.thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
            runner.close()

    def test_factory_failure_does_not_leave_thread(self):
        def fail():
            raise ValueError("factory failed")

        with self.assertRaisesRegex(ValueError, "factory failed"):
            AsyncTurnRunner(fail)
        self.assertFalse(
            any(thread.name == "simulst-async-vllm" and thread.is_alive() for thread in threading.enumerate())
        )


if __name__ == "__main__":
    unittest.main()
