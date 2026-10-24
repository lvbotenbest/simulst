"""Run independent turns concurrently on one vLLM ``AsyncLLM`` instance.

``submit`` is intentionally synchronous: each caller blocks until its own final
output, while calls from different threads are scheduled together by vLLM's
single asyncio engine loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import inspect
import logging
import threading
import uuid

logger = logging.getLogger(__name__)


class _TurnTimedOut(Exception):
    pass


class AsyncTurnRunner:
    """Thread-safe synchronous facade over a shared asynchronous vLLM engine."""

    def __init__(self, factory, *, timeout_seconds: float = 900.0):
        if timeout_seconds <= 0:
            raise ValueError("async turn timeout must be positive")

        self.timeout_seconds = timeout_seconds
        self._factory = factory
        self._ready = concurrent.futures.Future()
        self._lifecycle = threading.Lock()
        self._state = "open"
        self._closed_event = threading.Event()
        self._shutdown_error: BaseException | None = None

        self._stats_lock = threading.Lock()
        self._counts = {
            "submitted": 0,
            "active": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
        }
        # Accessed only on the event-loop thread.
        self._owners: dict[str, object] = {}
        self._tasks: dict[str, asyncio.Task] = {}

        self.thread = threading.Thread(
            target=self._thread_main,
            name="simulst-async-vllm",
            daemon=True,
        )
        self.thread.start()
        try:
            self._ready.result()
        except BaseException:
            self.thread.join(timeout=30)
            with self._lifecycle:
                self._state = "closed"
            raise

    async def _maybe_await(self, result):
        if inspect.isawaitable(result):
            return await result
        return result

    def _thread_main(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        async def initialize():
            self.engine = self._factory()
            if inspect.isawaitable(self.engine):
                self.engine = await self.engine

        try:
            self.loop.run_until_complete(initialize())
        except BaseException as error:
            self._ready.set_exception(error)
            self.loop.close()
            self._closed_event.set()
            return

        self._ready.set_result(True)
        try:
            self.loop.run_forever()
        finally:
            remaining = asyncio.all_tasks(self.loop)
            for task in remaining:
                task.cancel()
            if remaining:
                self.loop.run_until_complete(
                    asyncio.gather(*remaining, return_exceptions=True)
                )
            try:
                self.loop.run_until_complete(
                    self._maybe_await(self.engine.shutdown())
                )
            except BaseException as error:
                self._shutdown_error = error
            finally:
                self.loop.close()
                self._closed_event.set()

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                f"async_turn_{key}": value for key, value in self._counts.items()
            }

    async def _abort_best_effort(self, request_id: str) -> None:
        try:
            await self._maybe_await(self.engine.abort(request_id))
        except BaseException:
            logger.warning("failed to abort vLLM request %s", request_id, exc_info=True)

    async def _generate(self, prompt, params, request_id, owner, cancel_event):
        from vllm.sampling_params import RequestOutputKind

        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("session cancelled before submit")

        params = copy.copy(params)
        params.output_kind = RequestOutputKind.FINAL_ONLY
        self._owners[request_id] = owner
        self._tasks[request_id] = asyncio.current_task()
        with self._stats_lock:
            self._counts["submitted"] += 1
            self._counts["active"] += 1

        async def consume():
            final = None
            async for output in self.engine.generate(prompt, params, request_id):
                if output.request_id != request_id:
                    raise RuntimeError("vLLM returned a different request ID")
                if output.finished:
                    final = output
            return final

        consume_task = asyncio.create_task(consume())
        try:
            try:
                done, _ = await asyncio.wait(
                    (consume_task,), timeout=self.timeout_seconds
                )
                if not done:
                    raise _TurnTimedOut
                final = consume_task.result()
            finally:
                if not consume_task.done():
                    consume_task.cancel()
                    await asyncio.gather(consume_task, return_exceptions=True)

            if final is None:
                raise RuntimeError("vLLM ended without a final output")
            with self._stats_lock:
                self._counts["completed"] += 1
            return final
        except (asyncio.CancelledError, concurrent.futures.CancelledError):
            with self._stats_lock:
                self._counts["cancelled"] += 1
            await self._abort_best_effort(request_id)
            raise
        except BaseException:
            with self._stats_lock:
                self._counts["failed"] += 1
            await self._abort_best_effort(request_id)
            raise
        finally:
            self._owners.pop(request_id, None)
            self._tasks.pop(request_id, None)
            with self._stats_lock:
                self._counts["active"] -= 1

    def submit(self, prompt, params, *, owner=None, cancel_event=None):
        if threading.current_thread() is self.thread:
            raise RuntimeError("cannot block the vLLM event loop")

        request_id = "simulst-" + uuid.uuid4().hex
        with self._lifecycle:
            if self._state != "open":
                raise RuntimeError("async vLLM runner is closed")
            future = asyncio.run_coroutine_threadsafe(
                self._generate(
                    prompt,
                    params,
                    request_id,
                    owner,
                    cancel_event,
                ),
                self.loop,
            )
        try:
            return future.result()
        except _TurnTimedOut as error:
            raise TimeoutError(
                f"async vLLM turn timed out after {self.timeout_seconds:g} seconds"
            ) from error

    def cancel_owner(self, owner) -> None:
        if threading.current_thread() is self.thread:
            raise RuntimeError("cannot block the vLLM event loop")
        with self._lifecycle:
            if self._state != "open":
                return

            async def cancel():
                selected = [
                    task
                    for request_id, task in self._tasks.items()
                    if self._owners.get(request_id) == owner
                ]
                for task in selected:
                    task.cancel()
                if selected:
                    await asyncio.gather(*selected, return_exceptions=True)

            future = asyncio.run_coroutine_threadsafe(cancel(), self.loop)
        future.result(timeout=30)

    def close(self) -> None:
        if threading.current_thread() is self.thread:
            raise RuntimeError("cannot close the runner from its event loop")

        owner = False
        with self._lifecycle:
            if self._state == "open":
                self._state = "closing"
                owner = True
            elif self._state == "closed":
                if self._shutdown_error is not None:
                    raise RuntimeError("async vLLM shutdown failed") from self._shutdown_error
                return

        close_error: BaseException | None = None
        if owner:
            async def finish():
                tasks = list(self._tasks.values())
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            try:
                future = asyncio.run_coroutine_threadsafe(finish(), self.loop)
                future.result(timeout=30)
            except BaseException as error:
                close_error = error
            finally:
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.thread.join(timeout=30)
                finished = self._closed_event.wait(timeout=30)
                if not finished or self.thread.is_alive():
                    if close_error is None:
                        close_error = RuntimeError(
                            "async vLLM shutdown did not finish"
                        )
                else:
                    with self._lifecycle:
                        self._state = "closed"
        else:
            finished = self._closed_event.wait(timeout=30)
            if finished:
                self.thread.join(timeout=1)
            if not finished or self.thread.is_alive():
                close_error = RuntimeError("async vLLM shutdown did not finish")
            else:
                with self._lifecycle:
                    self._state = "closed"

        if close_error is not None:
            raise close_error
        if self._shutdown_error is not None:
            raise RuntimeError("async vLLM shutdown failed") from self._shutdown_error
