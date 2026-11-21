from tribunal.cli import main


def test_check_config_ok(test_env, capsys):
    assert main(["check-config"]) == 0
    assert capsys.readouterr().out.startswith("OK test-1+")


def test_check_config_reports(test_env, monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL")
    assert main(["check-config"]) == 2
    assert "- DATABASE_URL is required but not set" in capsys.readouterr().err


def test_scan_text(test_env, tmp_path, capsys):
    scan = tmp_path / "scan"
    scan.mkdir()
    f = scan / "a.txt"
    f.write_text("hello \U0001F600", encoding="utf-8")
    args = ["scan-text", "--config", str(test_env["runtime"]), str(scan)]
    assert main(args) == 1
    assert ":1:7 U+1F600" in capsys.readouterr().out
    f.write_text("hello", encoding="utf-8")
    assert main(args) == 0
