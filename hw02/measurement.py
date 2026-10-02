"""Предоставленные измерения для student-written raw loop."""

from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class _ObservedBatches:
    def __init__(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
        stats: "PaddingStats",
    ) -> None:
        self._batches = batches
        self._stats = stats

    def __iter__(self) -> Iterator[Mapping[str, torch.Tensor]]:
        for batch in self._batches:
            self._stats.observe(batch)
            yield batch


class PaddingStats:
    """Count padding in consumed train batches without changing the batches."""

    def __init__(self) -> None:
        self._microbatches = 0
        self._sequences = 0
        self._input_positions = 0
        self._non_padding_positions = 0

    def observe(self, batch: Mapping[str, torch.Tensor]) -> None:
        attention_mask = batch.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("batch must contain a tensor attention_mask")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must be two-dimensional [batch, sequence]")
        if attention_mask.numel() == 0:
            raise ValueError("attention_mask cannot be empty")

        self._microbatches += 1
        self._sequences += int(attention_mask.shape[0])
        self._input_positions += attention_mask.numel()
        self._non_padding_positions += int(attention_mask.sum().item())

    def wrap(
        self,
        batches: Iterable[Mapping[str, torch.Tensor]],
    ) -> Iterable[Mapping[str, torch.Tensor]]:
        return _ObservedBatches(batches, self)

    def summary(self) -> dict[str, int | float]:
        if self._microbatches == 0:
            raise RuntimeError("no train batches were observed")
        padding_positions = self._input_positions - self._non_padding_positions
        return {
            "observed_microbatches": self._microbatches,
            "observed_sequences": self._sequences,
            "input_positions": self._input_positions,
            "non_padding_positions": self._non_padding_positions,
            "padding_fraction": padding_positions / self._input_positions,
        }


def parameter_bytes(model: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def gradient_buffer_bytes(model: nn.Module) -> int:
    """Размер полного набора gradients, если каждый trainable parameter materialized."""

    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _value_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_value_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_value_bytes(item) for item in value)
    return 0


def optimizer_state_bytes(optimizers: Sequence[torch.optim.Optimizer]) -> int:
    return sum(_value_bytes(optimizer.state) for optimizer in optimizers)


def _storage_key(tensor: torch.Tensor) -> tuple[str, int | None, int, int]:
    storage = tensor.untyped_storage()
    return (tensor.device.type, tensor.device.index, storage.data_ptr(), storage.nbytes())


@dataclass
class _PackedTensor:
    tensor: torch.Tensor
    key: tuple[str, int | None, int, int] | None
    tracker: "_SavedTensorTracker"
    released: bool = False

    def release(self) -> None:
        if not self.released and self.key is not None:
            self.tracker.release(self.key)
            self.released = True

    def __del__(self) -> None:
        self.release()


class _SavedTensorTracker:
    """Peak storage retained by autograd, excluding model parameter storage."""

    def __init__(self, model: nn.Module) -> None:
        self._parameter_storages = {_storage_key(parameter) for parameter in model.parameters()}
        self._references: dict[tuple[str, int | None, int, int], int] = {}
        self.current_bytes = 0
        self.peak_bytes = 0

    def pack(self, tensor: torch.Tensor) -> _PackedTensor:
        key = _storage_key(tensor)
        if key in self._parameter_storages:
            return _PackedTensor(tensor=tensor, key=None, tracker=self)
        references = self._references.get(key, 0)
        if references == 0:
            self.current_bytes += key[-1]
            self.peak_bytes = max(self.peak_bytes, self.current_bytes)
        self._references[key] = references + 1
        return _PackedTensor(tensor=tensor, key=key, tracker=self)

    def unpack(self, packed: _PackedTensor) -> torch.Tensor:
        packed.release()
        return packed.tensor

    def release(self, key: tuple[str, int | None, int, int]) -> None:
        references = self._references[key] - 1
        if references == 0:
            self.current_bytes -= key[-1]
            del self._references[key]
        else:
            self._references[key] = references


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _NvmlSampler:
    def __init__(self, device: torch.device, interval_seconds: float = 0.2) -> None:
        self.device = device
        self.interval_seconds = interval_seconds
        self.values: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._paused = threading.Event()
        self._pynvml: Any | None = None
        self._handle: Any | None = None

    def start(self) -> None:
        if self.device.type != "cuda":
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            index = self.device.index
            if index is None:
                index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            uuid = getattr(properties, "uuid", None)
            if uuid is not None:
                self._handle = pynvml.nvmlDeviceGetHandleByUUID("GPU-" + str(uuid))
            else:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self._pynvml = pynvml
        except Exception:
            self._pynvml = None
            self._handle = None
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._pynvml is not None and self._handle is not None
        while not self._stop.wait(self.interval_seconds):
            if self._paused.is_set():
                continue
            try:
                utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.values.append(float(utilization.gpu))
            except Exception:
                return

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def stop(self) -> float | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        return statistics.fmean(self.values) if self.values else None


class RunMeter:
    """Measure optimizer-step windows after a fixed warm-up.

    `begin_step` must be called before the first microbatch of an optimizer step;
    `end_step` must be called after update, scheduler and zero_grad. Evaluation must
    happen outside this interval.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizers: Sequence[torch.optim.Optimizer],
        device: torch.device,
        warmup_steps: int,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        self.model = model
        self.optimizers = tuple(optimizers)
        self.device = device
        self.warmup_steps = warmup_steps
        self._saved_tensors = _SavedTensorTracker(model)
        self._started = False
        self._active_step: int | None = None
        self._measurement_start: float | None = None
        self._pause_start: float | None = None
        self._paused_seconds = 0.0
        self._cpu_step_start: float | None = None
        self._cpu_durations: list[float] = []
        self._cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._current_cuda_start: torch.cuda.Event | None = None
        self._tokens = 0
        self._peak_allocated = 0
        self._peak_reserved = 0
        self._sampler = _NvmlSampler(device)
        self._summary: dict[str, float | int | None] | None = None

    @contextmanager
    def track_saved_tensors(self):
        """Measure unique non-parameter storage retained for backward."""

        with torch.autograd.graph.saved_tensors_hooks(
            self._saved_tensors.pack,
            self._saved_tensors.unpack,
        ):
            yield

    @property
    def saved_for_backward_bytes(self) -> int:
        return self._saved_tensors.peak_bytes

    def start(self) -> None:
        if self._started:
            raise RuntimeError("RunMeter has already started")
        self._started = True

    def begin_step(self, step: int) -> None:
        if not self._started:
            raise RuntimeError("call RunMeter.start() before begin_step")
        if self._active_step is not None:
            raise RuntimeError("previous optimizer step has not ended")
        self._active_step = step
        if step <= self.warmup_steps:
            return
        if self._measurement_start is None:
            _synchronize(self.device)
            if self.device.type == "cuda":
                self._capture_cuda_peak()
                torch.cuda.reset_peak_memory_stats(self.device)
            self._measurement_start = time.perf_counter()
            self._sampler.start()
        if self.device.type == "cuda":
            self._current_cuda_start = torch.cuda.Event(enable_timing=True)
            self._current_cuda_start.record()
        else:
            self._cpu_step_start = time.perf_counter()

    def end_step(self, step: int, target_tokens: int) -> None:
        if self._active_step != step:
            raise RuntimeError("end_step does not match begin_step")
        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")
        self._active_step = None
        if step <= self.warmup_steps:
            return
        if self.device.type == "cuda":
            assert self._current_cuda_start is not None
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self._cuda_events.append((self._current_cuda_start, end))
            self._current_cuda_start = None
        else:
            assert self._cpu_step_start is not None
            self._cpu_durations.append(time.perf_counter() - self._cpu_step_start)
            self._cpu_step_start = None
        self._tokens += target_tokens

    def pause(self) -> None:
        """Exclude evaluation or logging work from training wall-clock."""

        if self._active_step is not None:
            raise RuntimeError("pause only between optimizer steps")
        if self._measurement_start is None or self._pause_start is not None:
            return
        _synchronize(self.device)
        self._capture_cuda_peak()
        self._sampler.pause()
        self._pause_start = time.perf_counter()

    def resume(self) -> None:
        if self._pause_start is None:
            return
        _synchronize(self.device)
        self._paused_seconds += time.perf_counter() - self._pause_start
        self._pause_start = None
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._sampler.resume()

    def _capture_cuda_peak(self) -> None:
        if self.device.type != "cuda":
            return
        self._peak_allocated = max(
            self._peak_allocated,
            int(torch.cuda.max_memory_allocated(self.device)),
        )
        self._peak_reserved = max(
            self._peak_reserved,
            int(torch.cuda.max_memory_reserved(self.device)),
        )

    def finish(self) -> dict[str, float | int | None]:
        if self._summary is not None:
            return dict(self._summary)
        if not self._started:
            raise RuntimeError("RunMeter was not started")
        if self._active_step is not None:
            raise RuntimeError("cannot finish inside an optimizer step")
        self.resume()
        if self._measurement_start is None or self._tokens == 0:
            raise RuntimeError("no optimizer steps remained after warm-up")
        _synchronize(self.device)
        self._capture_cuda_peak()
        wall_time = time.perf_counter() - self._measurement_start - self._paused_seconds
        if self.device.type == "cuda":
            durations = [start.elapsed_time(end) / 1000.0 for start, end in self._cuda_events]
            peak_allocated = self._peak_allocated
            peak_reserved = self._peak_reserved
        else:
            durations = self._cpu_durations
            peak_allocated = 0
            peak_reserved = 0
        mean_utilization = self._sampler.stop()
        self._summary = {
            "measured_steps": len(durations),
            "target_tokens": self._tokens,
            "wall_time_seconds": wall_time,
            "mean_wall_step_seconds": wall_time / len(durations),
            "median_device_step_seconds": statistics.median(durations),
            "target_tokens_per_second": self._tokens / wall_time,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "parameter_bytes": parameter_bytes(self.model),
            "gradient_buffer_bytes": gradient_buffer_bytes(self.model),
            "optimizer_state_bytes": optimizer_state_bytes(self.optimizers),
            "saved_for_backward_bytes": self.saved_for_backward_bytes,
            "mean_gpu_utilization_percent": mean_utilization,
        }
        return dict(self._summary)
