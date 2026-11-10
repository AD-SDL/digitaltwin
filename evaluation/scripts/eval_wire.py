# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tiny length-prefixed wire protocol shared by the eval client and policy server.

Why a socket at all: the sim and the policy CANNOT share a Python
interpreter. The IsaacLab venv pins ``huggingface-hub==1.32.0``, which makes
``import transformers`` raise (transformers wants ``<1.0``), and pi05 is a
PaliGemma + Gemma model that needs transformers. So the env runs in the
IsaacLab venv and the policy runs in the lerobot venv, and they talk over a
loopback TCP socket. See ``docs/07_evaluate_policy_rollouts.md``.

Frame layout (all integers big-endian, matching ``!I``)::

    uint32  total payload length
    uint32  header length
    bytes   header   -- UTF-8 JSON {"meta": {...}, "arrays": [{...}, ...]}
    bytes   body     -- each array's raw buffer, concatenated in header order

Images go over the wire as raw uint8 rather than JPEG on purpose: loopback
moves the ~3 MB/step at >1 GB/s, and lossy re-encoding would perturb exactly
the pixels the policy was trained on. The cost is measured and reported as
``wire_ms`` in the eval output, so you can see what the transport costs.

Stdlib + numpy only -- this module is imported by BOTH venvs, so it must not
pull in torch, lerobot or isaaclab.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any

import numpy as np

_LEN = struct.Struct("!I")

# Bump if the header schema changes incompatibly.
PROTOCOL_VERSION = 1
DEFAULT_PORT = 5557


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise ``ConnectionError``."""
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"peer closed with {remaining} of {n} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_msg(sock: socket.socket, meta: dict[str, Any], arrays: dict[str, np.ndarray] | None = None) -> None:
    """Send one ``(meta, arrays)`` message."""
    arrays = arrays or {}
    specs = []
    buffers = []
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        specs.append({"key": key, "dtype": arr.dtype.str, "shape": list(arr.shape)})
        buffers.append(arr.tobytes())

    header = json.dumps({"v": PROTOCOL_VERSION, "meta": meta, "arrays": specs}).encode("utf-8")
    body = b"".join(buffers)
    payload = _LEN.pack(len(header)) + header + body
    sock.sendall(_LEN.pack(len(payload)) + payload)


def recv_msg(sock: socket.socket) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Receive one ``(meta, arrays)`` message."""
    (total,) = _LEN.unpack(recv_exactly(sock, _LEN.size))
    payload = recv_exactly(sock, total)

    (header_len,) = _LEN.unpack(payload[: _LEN.size])
    header = json.loads(payload[_LEN.size : _LEN.size + header_len].decode("utf-8"))
    if header.get("v") != PROTOCOL_VERSION:
        raise ValueError(f"protocol mismatch: peer sent v{header.get('v')}, expected v{PROTOCOL_VERSION}")

    arrays: dict[str, np.ndarray] = {}
    offset = _LEN.size + header_len
    for spec in header["arrays"]:
        dtype = np.dtype(spec["dtype"])
        shape = tuple(spec["shape"])
        nbytes = int(dtype.itemsize * np.prod(shape)) if shape else dtype.itemsize
        buf = payload[offset : offset + nbytes]
        arrays[spec["key"]] = np.frombuffer(buf, dtype=dtype).reshape(shape)
        offset += nbytes

    return header["meta"], arrays
