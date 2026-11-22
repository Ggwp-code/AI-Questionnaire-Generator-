import json
import os
import time

import httpx

from tribunal.verifier.client import HttpVerifier
from tribunal.verifier.runner import Limits, run_program

LIMITS = Limits(cpu_seconds=2, wall_seconds=4, memory_mb=256, max_processes=64, max_file_bytes=65536, max_output_bytes=4096)


def test_program_result_last_line():
    r = run_program('print("noise")\nprint(\'{"answer": 42}\')\n', LIMITS)
    assert r.status == "ok" and json.loads(r.last_line) == {"answer": 42}


def test_environment_is_empty(monkeypatch):
    monkeypatch.setenv("VERIFIER_TEST_SECRET", "hunter2")
    r = run_program("import os, json\nprint(json.dumps(sorted(os.environ)))\n", LIMITS)
    names = json.loads(r.last_line)
    # The interpreter itself may add LC_CTYPE when it coerces the C locale.
    assert r.status == "ok" and {"HOME", "PATH"} <= set(names) <= {"HOME", "PATH", "LC_CTYPE"}
    assert "VERIFIER_TEST_SECRET" not in names


def test_cpu_loop_is_stopped():
    t = time.monotonic()
    r = run_program("while True:\n    pass\n", LIMITS)
    assert r.status in ("error", "timeout") and time.monotonic() - t < LIMITS.wall_seconds + 3


def test_memory_bomb_is_stopped():
    r = run_program("x = bytearray(2 * 1024 ** 3)\nprint('{}')\n", LIMITS)
    assert r.status == "error" and "MemoryError" in r.stderr_tail


def test_output_flood_is_too_large():
    r = run_program("while True:\n    print('x' * 1000)\n", LIMITS)
    assert r.status == "too_large"


def test_runs_in_its_own_directory_and_cleans_up():
    r = run_program("import os, json\nprint(json.dumps(os.getcwd()))\n", LIMITS)
    cwd = json.loads(r.last_line)
    assert cwd != os.getcwd() and not os.path.exists(cwd)


def test_exception_is_error_with_stderr_tail():
    r = run_program("raise ValueError('boom')\n", LIMITS)
    assert r.status == "error" and "ValueError: boom" in r.stderr_tail


def test_file_size_limit():
    r = run_program("open('big', 'wb').write(b'x' * 1000000)\nprint('{}')\n", LIMITS)
    assert r.status == "error"


def test_fork_bomb_is_stopped_within_wall_limit():
    t = time.monotonic()
    r = run_program("import os\nwhile True:\n    os.fork()\n", Limits(2, 4, 256, 8, 65536, 4096))
    assert r.status != "ok" and time.monotonic() - t < 4 + 3


def test_http_verifier_maps_failures():
    def boom(request):
        raise httpx.ConnectError("down")

    for handler, expect in ((lambda r: httpx.Response(500), "500"), (boom, "ConnectError")):
        c = httpx.Client(transport=httpx.MockTransport(handler))
        r = HttpVerifier("http://v", None, LIMITS, client=c).run("print(1)")
        assert r.status == "error" and r.last_line is None
        assert r.stderr_tail == f"verifier service unavailable: {expect}"


def test_http_verifier_posts_program_and_token():
    seen = {}

    def ok(request):
        seen["auth"] = request.headers.get("authorization")
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"status": "ok", "last_line": "{}", "stderr_tail": ""})

    c = httpx.Client(transport=httpx.MockTransport(ok))
    from pydantic import SecretStr

    r = HttpVerifier("http://v/", SecretStr("tok"), LIMITS, client=c).run("p")
    assert r.status == "ok" and seen["auth"] == "Bearer tok" and seen["url"] == "http://v/run"
    assert seen["body"]["program"] == "p" and seen["body"]["limits"]["cpu_seconds"] == 2
