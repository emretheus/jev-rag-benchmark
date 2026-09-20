from __future__ import annotations

import os

from openjev_rag_bench.config import load_dotenv


def test_load_dotenv_sets_missing_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "TEST_DOTENV_KEY=first\n"
        'TEST_DOTENV_QUOTED="quoted value"\n'
        "TEST_DOTENV_EMPTY=\n"
        "malformed line\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TEST_DOTENV_KEY", raising=False)
    monkeypatch.delenv("TEST_DOTENV_QUOTED", raising=False)
    load_dotenv(env_file)
    assert os.environ["TEST_DOTENV_KEY"] == "first"
    assert os.environ["TEST_DOTENV_QUOTED"] == "quoted value"


def test_load_dotenv_does_not_override_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_DOTENV_EXISTING=from_file\n", encoding="utf-8")
    monkeypatch.setenv("TEST_DOTENV_EXISTING", "from_env")
    load_dotenv(env_file)
    assert os.environ["TEST_DOTENV_EXISTING"] == "from_env"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    load_dotenv(tmp_path / "does-not-exist.env")
