"""Lennard-Jones parallel worker pool.

Mirrors the ParallelSCINEProcessor interface so it can be used as a drop-in
replacement with run_batch_parallel(). Each worker creates its own LJ
predict_fn (cheap — no external calculator needed).
"""

from __future__ import annotations

import inspect
import os
import queue
from typing import Any, Callable, Dict, Optional

import torch
import multiprocessing as mp


def _set_thread_env(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    torch.set_num_threads(threads)


def lj_worker_fn(
    rank: int,
    sigma_scale: float,
    threads_per_worker: int,
    work_queue,
    result_queue,
    done_event,
    worker_fn: Callable[..., Any],
    worker_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Worker loop: set OMP threads, create LJ predict_fn, process samples."""
    _set_thread_env(threads_per_worker)

    from ..dependencies.calculators import make_lj_predict_fn

    predict_fn = make_lj_predict_fn(sigma_scale=sigma_scale)

    kwargs = worker_kwargs or {}
    signature = inspect.signature(worker_fn)
    use_calculator = "calculator" in signature.parameters
    while not done_event.is_set():
        try:
            item = work_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if item is None:
            break
        idx, payload = item
        try:
            if use_calculator:
                # LJ has no separate calculator object; pass None
                result = worker_fn(predict_fn, None, payload, **kwargs)
            else:
                result = worker_fn(predict_fn, payload, **kwargs)
        except Exception as exc:
            result = {"error": str(exc)}
        result_queue.put((idx, result))


class ParallelLJProcessor:
    """Parallel processing pool for Lennard-Jones workloads.

    Same interface as ParallelSCINEProcessor: submit(), result_queue, start(), close().
    """

    def __init__(
        self,
        sigma_scale: float,
        threads_per_worker: int,
        n_workers: int,
        worker_fn: Callable[..., Any],
        worker_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.sigma_scale = sigma_scale
        self.threads_per_worker = threads_per_worker
        self.n_workers = n_workers
        self.worker_fn = worker_fn
        self.worker_kwargs = worker_kwargs or {}

        self.ctx = mp.get_context("spawn")
        self.work_queue = self.ctx.Queue()
        self.result_queue = self.ctx.Queue()
        self.done_event = self.ctx.Event()
        self.processes = []

    def start(self) -> None:
        if self.processes:
            return
        for rank in range(self.n_workers):
            proc = self.ctx.Process(
                target=lj_worker_fn,
                args=(
                    rank,
                    self.sigma_scale,
                    self.threads_per_worker,
                    self.work_queue,
                    self.result_queue,
                    self.done_event,
                    self.worker_fn,
                    self.worker_kwargs,
                ),
            )
            proc.start()
            self.processes.append(proc)

    def submit(self, idx: int, payload: Any) -> None:
        self.work_queue.put((idx, payload))

    def close(self) -> None:
        if not self.processes:
            return
        self.done_event.set()
        for _ in range(self.n_workers):
            self.work_queue.put(None)
        for proc in self.processes:
            proc.join()
        self.processes = []
