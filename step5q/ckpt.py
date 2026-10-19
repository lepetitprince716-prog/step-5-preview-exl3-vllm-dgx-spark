# SPDX-License-Identifier: Apache-2.0
"""Safetensors checkpoint reader with large sequential reads, ported from sudoingx dsv41q (refmodel.Checkpoint,
drop_page_cache). Changes: tensor names are used as-is (Step 5 keys start with `model.`), the pinned buffer is only
pinned when CUDA is available (CPU tests), and posix_fadvise is skipped where the OS lacks it.
"""
from __future__ import annotations

import json
import logging
import os
import struct
from collections.abc import Iterable

import torch

log = logging.getLogger(__name__)

_FADVISE = getattr(os, "posix_fadvise", None)
_DONTNEED = getattr(os, "POSIX_FADV_DONTNEED", 4)


def drop_page_cache(path: str) -> None:
    """Evict one file's pages from the page cache (written pages are flushed first). On GB10 CUDA allocations do not
    reclaim page cache (sudoingx measured 45 GB "available", 8 GB free, a 67 MB cudaMalloc failing), so every big file
    is evicted once used."""
    if _FADVISE is None:
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            _FADVISE(fd, 0, 0, _DONTNEED)
        finally:
            os.close(fd)
    except OSError as e:
        log.debug("drop_page_cache %s: %s", path, e)


class Checkpoint:
    """Tensor name -> shard file + byte offsets, from model.safetensors.index.json and the shard headers."""

    _DT = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
           "I8": torch.int8, "U8": torch.uint8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64}

    def __init__(self, ckpt_dir: str):
        self.dir = ckpt_dir
        with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._headers: dict[str, tuple[int, dict]] = {}
        self._buf: torch.Tensor | None = None

    def header(self, fname: str) -> tuple[int, dict]:
        """(data start offset, {name: {dtype, shape, data_offsets}}) of one shard."""
        h = self._headers.get(fname)
        if h is None:
            with open(os.path.join(self.dir, fname), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                meta = json.loads(f.read(n))
            meta.pop("__metadata__", None)
            h = self._headers[fname] = (8 + n, meta)
        return h

    def meta(self, name: str) -> dict:
        """{dtype, shape, data_offsets} of one tensor (header only, no data read)."""
        return self.header(self.weight_map[name])[1][name]

    def names(self, prefix: str) -> list[str]:
        return sorted(k for k in self.weight_map if k.startswith(prefix))

    def get(self, name: str, device: str | torch.device = "cpu") -> torch.Tensor:
        return self.get_many([name], device)[name]

    def get_many(self, names: Iterable[str], device: str | torch.device = "cpu", gap: int = 64 << 20) -> dict:
        """Read several tensors with large sequential reads: per shard the wanted byte ranges are merged into spans
        (gaps up to `gap` read through) and each span lands in one reusable pinned buffer with readinto. safetensors'
        mmap-backed get_tensor page-faults 4 KB at a time on a cold cache (sudoingx: ~150 MB/s); this reads at NVMe
        speed. Returned tensors are on `device`; on CPU they are clones (the buffer is reused)."""
        by_file: dict[str, list[str]] = {}
        for n in names:
            by_file.setdefault(self.weight_map[n], []).append(n)
        out: dict[str, torch.Tensor] = {}
        for fname, ns in by_file.items():
            self._read_file(fname, ns, torch.device(device), gap, out)
        return out

    def _read_file(self, fname: str, ns: list[str], device: torch.device, gap: int, out: dict) -> None:
        base, meta = self.header(fname)
        ents = sorted((base + meta[n]["data_offsets"][0], base + meta[n]["data_offsets"][1], n) for n in ns)
        spans: list[list] = []
        for s0, s1, n in ents:
            if spans and s0 - spans[-1][1] <= gap:
                spans[-1][1] = max(spans[-1][1], s1)
                spans[-1][2].append((s0, s1, n))
            else:
                spans.append([s0, s1, [(s0, s1, n)]])
        with open(os.path.join(self.dir, fname), "rb", buffering=0) as f:
            for s0, s1, members in spans:
                buf = self._pinned(s1 - s0)
                f.seek(s0)
                mv, got = memoryview(buf.numpy()), 0
                while got < s1 - s0:
                    r = f.readinto(mv[got: s1 - s0])
                    if not r:
                        raise IOError(f"short read {fname} @ {s0 + got}")
                    got += r
                for t0, t1, n in members:
                    m = meta[n]
                    t = buf[t0 - s0: t1 - s0].view(self._DT[m["dtype"]]).view(m["shape"])
                    out[n] = t.clone() if device.type == "cpu" else t.to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.current_stream(device).synchronize()     # the buffer is reused by the next span
                if _FADVISE is not None:
                    _FADVISE(f.fileno(), s0, s1 - s0, _DONTNEED)

    def _pinned(self, nbytes: int) -> torch.Tensor:
        b = self._buf
        if b is None or b.numel() < nbytes:
            self._buf = None                                            # free the old buffer before the new one
            b = self._buf = torch.empty(max(nbytes, 1 << 30), dtype=torch.uint8,
                                        pin_memory=torch.cuda.is_available())
        return b
