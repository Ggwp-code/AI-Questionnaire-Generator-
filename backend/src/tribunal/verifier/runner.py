"""Run one untrusted Python program under hard resource limits.

Standard library only: the verifier image copies this file on its own.
"""

from __future__ import annotations

import os
import resource
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

STDERR_TAIL_BYTES = 2000


@dataclass(frozen=True)
class Limits:
    cpu_seconds: int
    wall_seconds: int
    memory_mb: int
    max_processes: int
    max_file_bytes: int
    max_output_bytes: int


@dataclass(frozen=True)
class RunResult:
    status: Literal["ok", "error", "timeout", "too_large"]
    last_line: str | None
    stderr_tail: str


def _apply_limits(limits: Limits):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
        mem = limits.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (limits.max_file_bytes, limits.max_file_bytes))

    return apply


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_program(program: str, limits: Limits, python: str = sys.executable) -> RunResult:
    workdir = tempfile.mkdtemp(prefix="verifier-")
    try:
        with open(os.path.join(workdir, "program.py"), "w", encoding="utf-8") as f:
            f.write(program)
        proc = subprocess.Popen(
            [python, "-I", "program.py"],
            cwd=workdir,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": workdir},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=_apply_limits(limits),  # noqa: PLW1509 - limits must be set in the child
        )
        tail: deque[bytes] = deque()

        def drain_stderr() -> None:
            size = 0
            for chunk in iter(lambda: proc.stderr.read(4096), b""):
                tail.append(chunk)
                size += len(chunk)
                while size - len(tail[0]) >= STDERR_TAIL_BYTES:
                    size -= len(tail.popleft())

        reader = threading.Thread(target=drain_stderr, daemon=True)
        reader.start()
        out = bytearray()
        status: str | None = None
        deadline = time.monotonic() + limits.wall_seconds
        try:
            fd = proc.stdout.fileno()
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    status = "timeout"
                    break
                if not select.select([fd], [], [], left)[0]:
                    continue
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                out += chunk
                if len(out) > limits.max_output_bytes:
                    status = "too_large"
                    break
            if status is None:
                try:
                    proc.wait(timeout=max(deadline - time.monotonic(), 0.01))
                except subprocess.TimeoutExpired:
                    status = "timeout"
        finally:
            _kill_group(proc)
            proc.wait()
            proc.stdout.close()
            reader.join(timeout=1)
            proc.stderr.close()
        stderr_tail = b"".join(tail).decode("utf-8", "replace")[-STDERR_TAIL_BYTES:]
        if status is None:
            status = "ok" if proc.returncode == 0 else "error"
        lines = out.decode("utf-8", "replace").splitlines()
        last = next((ln for ln in reversed(lines) if ln.strip()), None) if status == "ok" else None
        return RunResult(status, last, stderr_tail)  # type: ignore[arg-type]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
