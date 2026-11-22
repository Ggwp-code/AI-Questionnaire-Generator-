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


def test_parent_secret_is_not_inherited(monkeypatch):
    monkeypatch.setenv("VERIFIER_TEST_SECRET", "hunter2")
    r = run_program("import os, json\nprint(json.dumps(sorted(os.environ)))\n", LIMITS)
    assert r.status == "ok" and "VERIFIER_TEST_SECRET" not in json.loads(r.last_line)


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


def test_escaped_child_does_not_hold_the_runner():
    prog = (
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    time.sleep(8)\n"
        "    os._exit(0)\n"
        "time.sleep(8)\n"
    )
    limits = Limits(cpu_seconds=2, wall_seconds=2, memory_mb=256, max_processes=4096, max_file_bytes=65536, max_output_bytes=4096)
    t = time.monotonic()
    r = run_program(prog, limits)
    assert r.status == "timeout" and time.monotonic() - t < limits.wall_seconds + 2


def _http(handler, url="http://v"):
    return HttpVerifier(url, None, LIMITS, client=httpx.Client(transport=httpx.MockTransport(handler))).run("p")


def test_http_verifier_never_raises_and_validates_bodies():
    bad = [
        httpx.Response(200, json={"status": "weird", "last_line": None, "stderr_tail": ""}),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"status": "ok", "last_line": 5, "stderr_tail": ""}),
        httpx.Response(200, json={"status": "ok", "last_line": None, "stderr_tail": 7}),
        httpx.Response(200, json=["ok"]),
    ]
    for resp in bad:
        r = _http(lambda req, resp=resp: resp)
        assert r.status == "error" and r.stderr_tail == "verifier service unavailable: malformed response", resp
    assert _http(lambda req: httpx.Response(200), url="not a url").status == "error"
    assert _http(lambda req: httpx.Response(200), url="http://[bad").status == "error"


def _example_limits():
    from pathlib import Path

    import yaml

    cfg = yaml.safe_load((Path(__file__).parents[1] / "config" / "runtime.example.yaml").read_text())
    return Limits(**cfg["verifier"])


def test_numpy_imports_under_example_limits():
    r = run_program("import numpy\nprint('{\"v\": \"%s\"}' % numpy.__version__)\n", _example_limits())
    assert r.status == "ok", r.stderr_tail


def test_sympy_imports_under_example_limits():
    import pytest

    pytest.importorskip("sympy")
    r = run_program("import sympy\nprint('{}')\n", _example_limits())
    assert r.status == "ok", r.stderr_tail


def test_child_environment_only_has_thread_pinning():
    r = run_program("import os, json\nprint(json.dumps(dict(os.environ)))\n", LIMITS)
    env = json.loads(r.last_line)
    assert {k for k in env if k not in ("PATH", "HOME", "LC_CTYPE")} == {
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}
    assert env["OMP_NUM_THREADS"] == "1"


def test_setrlimit_failure_is_an_error_result():
    hard = resource_hard_nproc()
    bad = Limits(2, 4, 256, hard + 1000, 65536, 4096) if hard is not None else None
    if bad is None:
        import pytest

        pytest.skip("no finite hard NPROC")
    r = run_program("print('{}')\n", bad)
    assert r.status == "error" and r.last_line is None and r.stderr_tail


def resource_hard_nproc():
    import resource

    h = resource.getrlimit(resource.RLIMIT_NPROC)[1]
    return None if h == resource.RLIM_INFINITY else h


def test_core_dumps_are_disabled():
    r = run_program("import resource\nprint(resource.getrlimit(resource.RLIMIT_CORE)[0])\n", LIMITS)
    assert r.last_line == "0"


def test_http_missing_key_and_deep_nesting_are_errors():
    r = _http(lambda req: httpx.Response(200, json={"status": "ok", "stderr_tail": ""}))
    assert r.status == "error" and r.stderr_tail == "verifier service unavailable: malformed response"
    deep = b"[" * 100000 + b"]" * 100000
    r = _http(lambda req: httpx.Response(200, content=deep))
    assert r.status == "error" and r.last_line is None


def test_invalid_url_raises_inside_httpx():
    r = _http(lambda req: httpx.Response(200), url="http://\x00v")
    assert r.status == "error" and r.stderr_tail.startswith("verifier service unavailable:")


def test_last_line_written_just_before_exit_is_kept():
    r = run_program("import os\nos.write(1, b'{\"a\": 1}\\n')\nos._exit(0)\n", LIMITS)
    assert r.status == "ok" and r.last_line == '{"a": 1}'


def test_lone_surrogate_program_is_an_error_result():
    r = run_program("x = '\ud800'\n", LIMITS)
    assert r.status == "error" and r.stderr_tail == "program text is not valid UTF-8"
