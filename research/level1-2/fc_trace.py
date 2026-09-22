"""Read-only tensor tracing. Copy this file into each application's root.

FC_TRACE_DIR: new output directory; unset disables tracing.
FC_TRACE_LEVEL: 1 (solver only, default) or 2 (CFG/UNet packets too).
FC_TRACE_STEPS: comma-separated steps for level-2 packets; default '0'.
One process, single GPU, batch=1, ordinary txt2img only.
"""
import collections
import hashlib
import json
import os
import platform
import threading
from pathlib import Path

import numpy as np
import torch


def _json(v):
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return v if np.isfinite(v) else str(v)
    if isinstance(v, (list, tuple)):
        return [_json(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json(x) for k, x in v.items()}
    if callable(v):
        return getattr(v, '__module__', '') + '.' + getattr(v, '__qualname__', type(v).__name__)
    return str(v)


class Trace:
    def __init__(self):
        self.root = os.environ.get('FC_TRACE_DIR')
        self.level = int(os.environ.get('FC_TRACE_LEVEL', '1'))
        steps = os.environ.get('FC_TRACE_STEPS', '0')
        self.detail_steps = None if steps == 'all' else {int(s) for s in steps.split(',') if s.strip()}
        self.step = -1
        self.pass_index = -1
        self.directory = None
        self.sources = {}
        self.counts = collections.Counter()
        self.seq = 0
        self.lock = threading.RLock()

    def source(self, path):
        if self.root:
            p = Path(path)
            self.sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

    def enabled(self, level=1):
        return bool(self.root and self.directory is not None and self.level >= level
                    and (level == 1 or self.detail_steps is None or self.step in self.detail_steps))

    def begin(self, **info):
        if not self.root:
            return
        with self.lock:
            if self.pass_index < 0:
                Path(self.root).mkdir(parents=True, exist_ok=False)
            self.pass_index += 1
            self.directory = Path(self.root) / ('pass_%03d' % self.pass_index)
            self.directory.mkdir(exist_ok=False)
            self.step, self.seq = -1, 0
            self.counts.clear()
            runtime = dict(python=platform.python_version(), platform=platform.platform(),
                           torch=torch.__version__, cuda=torch.version.cuda,
                           gpu_names=([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                                      if torch.cuda.is_available() else []),
                           default_dtype=str(torch.get_default_dtype()),
                           deterministic=torch.are_deterministic_algorithms_enabled(),
                           cudnn_deterministic=torch.backends.cudnn.deterministic,
                           cudnn_benchmark=torch.backends.cudnn.benchmark,
                           cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                           matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                           float32_matmul_precision=torch.get_float32_matmul_precision(),
                           source_sha256=self.sources, **info)
            self.meta('run', **runtime)

    def set_step(self, step):
        self.step = int(step)

    def _write(self, event):
        event = dict(seq=self.seq, step=self.step, **event)
        self.seq += 1
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(_json(event), ensure_ascii=False, allow_nan=False) + '\n')

    def meta(self, name, level=1, **values):
        if self.enabled(level):
            with self.lock:
                self._write(dict(kind='meta', name=name, values=values))

    def put(self, name, tensor, level=1):
        if not self.enabled(level):
            return
        if tensor is None:
            self.meta(name, level=level, absent=True)
            return
        if not isinstance(tensor, torch.Tensor) or tensor.is_nested or tensor.is_complex():
            raise TypeError('Trace expects a real, dense tensor: ' + name)
        with self.lock:
            # Copy now: a later in-place sampler update must not change the dump.
            value = tensor.detach().to(device='cpu', copy=True).contiguous()
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            # f16/bf16/f32 are exactly representable in f64. Original dtype and
            # raw-byte hash are retained separately; no inference tensor changes.
            array = value.to(torch.float64).numpy() if value.is_floating_point() else value.numpy()
            occurrence = self.counts[(self.step, name)]
            self.counts[(self.step, name)] += 1
            filename = '%06d.npy' % self.seq
            np.save(self.directory / filename, array, allow_pickle=False)
            self._write(dict(kind='tensor', name=name, occurrence=occurrence,
                             shape=list(value.shape), dtype=str(tensor.dtype),
                             device=str(tensor.device), stride=list(tensor.stride()),
                             contiguous=tensor.is_contiguous(), file=filename,
                             raw_sha256=hashlib.sha256(raw).hexdigest()))

    def packet(self, name, values, labels):
        """Split actual model batches by cond_or_uncond (0=cond, 1=uncond).

        Chunk order may differ between apps. Label+occurrence, not call ordinal,
        is the matching key. Multiple conditions/regions require further IDs.
        """
        if not self.enabled(2):
            return
        labels = list(labels or [])
        self.meta(name + '.batch', level=2, labels=labels)
        anchor = next((v for v in values.values() if isinstance(v, torch.Tensor) and v.ndim), None)
        batch = anchor.shape[0] if anchor is not None else 0
        if not labels or not batch or batch % len(labels):
            raise ValueError('Cannot label model batch; use single-GPU ordinary CFG')
        for key, value in values.items():
            if value is None:
                self.meta(name + '.shared.' + key, level=2, absent=True)
            elif value.ndim and value.shape[0] == batch:
                for label, chunk in zip(labels, value.chunk(len(labels), dim=0)):
                    branch = {0: 'cond', 1: 'uncond'}.get(int(label), str(label))
                    self.put(name + '.' + branch + '.' + key, chunk, level=2)
            else:
                self.put(name + '.shared.' + key, value, level=2)


trace = Trace()
trace.source(__file__)
