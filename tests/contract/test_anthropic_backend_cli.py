"""Contract tests for the CLI integration of the Anthropic backend.

Prove the CLI plumbs provider flags correctly without touching the
network or the real ``anthropic`` SDK.
"""

from __future__ import annotations

import io
import sys
import types
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from llloom.cli import main as cli_main
from llloom.llm.anthropic_backend import AnthropicModelBackend
from llloom.llm.harness import LLMInvoke
from llloom.state.journal import OperationJournal
from llloom.workspace.layout import Workspace


SOURCE = """\
# Article

## Methods

Complementarity prioritizes sites that add features not already represented in the selected set.
"""


def _seed_workspace(tmp_path: Path) -> tuple[Workspace, Path]:
    ws = Workspace.init(tmp_path)
    src = ws.raw_sources / "article.md"
    src.write_text(SOURCE, encoding="utf-8")
    return ws, src


def _text_block(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(type="text", text=text)


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch, client_obj) -> None:
    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda **kwargs: client_obj  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)


def _install_missing_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a machine without the anthropic SDK by ensuring
    ``import anthropic`` raises ``ImportError``."""

    class _Raiser:
        def find_spec(self, name, path=None, target=None):  # noqa: ARG002
            if name == "anthropic":
                raise ImportError("no anthropic SDK")
            return None

    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Raiser(), *sys.meta_path])


# --- 1. CLI with --model-provider anthropic builds the adapter ----------


def test_cli_ingest_with_anthropic_provider_builds_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_workspace(tmp_path)

    captured: dict = {}

    import llloom.cli as cli_module

    real_ingest = cli_module.ingest

    def _spy_ingest(workspace, path, **kwargs):
        captured["llm"] = kwargs.get("llm")
        return real_ingest(workspace, path, **kwargs)

    monkeypatch.setattr(cli_module, "ingest", _spy_ingest)

    # Provide a fake anthropic module so _build_harness's early probe
    # succeeds, then route generate through a fake that yields an
    # empty yaml body.
    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    captured_calls: list = []

    def _fake_generate(self, prompt: str) -> str:  # noqa: ARG001
        captured_calls.append(prompt)
        return "claims: []\n"

    monkeypatch.setattr(AnthropicModelBackend, "generate", _fake_generate)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main([
            "--root", str(ws.root),
            "ingest", str(src),
            "--model-provider", "anthropic",
            "--model", "claude-test-1",
        ])

    assert rc == 0, buf.getvalue()
    harness = captured["llm"]
    assert isinstance(harness, LLMInvoke)
    backend = harness._model  # type: ignore[attr-defined]
    assert isinstance(backend, AnthropicModelBackend)
    assert backend.model == "claude-test-1"
    assert backend.identifier == "anthropic/claude-test-1"
    assert len(captured_calls) == 1


# --- 2. Default CLI ingest does not construct the Anthropic adapter ----


def test_cli_default_ingest_does_not_construct_anthropic_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_workspace(tmp_path)

    import llloom.cli as cli_module

    captured: dict = {}
    real_ingest = cli_module.ingest

    def _spy_ingest(workspace, path, **kwargs):
        captured["llm"] = kwargs.get("llm")
        return real_ingest(workspace, path, **kwargs)

    monkeypatch.setattr(cli_module, "ingest", _spy_ingest)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main(["--root", str(ws.root), "ingest", str(src)])
    assert rc == 0, buf.getvalue()
    # Default CLI path passes llm=None so ingest uses its internal NullModel.
    assert captured["llm"] is None


# --- 3. Missing optional dependency produces a clear CLI error ---------


def test_cli_anthropic_missing_sdk_refuses_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_workspace(tmp_path)
    _install_missing_anthropic(monkeypatch)

    err_buf = io.StringIO()
    out_buf = io.StringIO()
    with redirect_stderr(err_buf), redirect_stdout(out_buf):
        rc = cli_main([
            "--root", str(ws.root),
            "ingest", str(src),
            "--model-provider", "anthropic",
            "--model", "claude-test-1",
        ])
    assert rc == 2
    assert "llloom[anthropic]" in err_buf.getvalue()
    assert out_buf.getvalue() == ""


# --- 4. index_only ingest cuts off BEFORE any backend call -------------


def _wire_index_only(ws: Workspace) -> None:
    (ws.schema / "source_classes.yaml").write_text(
        "classes:\n"
        "  markdown_prose:\n"
        "    locator: markdown_prose_v1\n"
        "  sensitive:\n"
        "    locator: markdown_prose_v1\n",
        encoding="utf-8",
    )
    (ws.schema / "ingest_policies.yaml").write_text(
        "policies:\n"
        "  markdown_prose: claim_extract_and_view_render\n"
        "  sensitive: index_only\n"
        "defaults:\n  unknown: deny\n",
        encoding="utf-8",
    )


def test_index_only_ingest_with_anthropic_provider_does_not_invoke_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws = Workspace.init(tmp_path)
    _wire_index_only(ws)
    src = ws.raw_sources / "contract.md"
    src.write_text(
        "# Contract\n\nNet-30 with a 2% early-payment discount.\n",
        encoding="utf-8",
    )

    fake = types.ModuleType("anthropic")
    fake.Anthropic = lambda **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    def _fail_generate(self, prompt: str) -> str:  # noqa: ARG001
        raise AssertionError(
            "AnthropicModelBackend.generate must not be called for index_only ingest"
        )

    monkeypatch.setattr(AnthropicModelBackend, "generate", _fail_generate)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main([
            "--root", str(ws.root),
            "ingest", str(src),
            "--source-class", "sensitive",
            "--model-provider", "anthropic",
            "--model", "claude-test-1",
        ])
    assert rc == 0, buf.getvalue()


# --- 5. Invocation-log persists once per Anthropic-backed ingest -------


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **kwargs):  # noqa: ARG002
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=self._text)]
        )


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessages(text)


def test_invocation_log_persists_once_per_anthropic_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ws, src = _seed_workspace(tmp_path)

    yaml_body = (
        "claims:\n"
        "  - entity_id: concept.complementarity\n"
        "    entity_type: concept\n"
        "    display_name: Complementarity\n"
        "    claim_id: c.cmp.1\n"
        "    claim_kind: definition\n"
        "    claim_text: Complementarity prioritizes sites that add features not already represented in the selected set.\n"
        "    locator:\n"
        "      locator_type: markdown_prose_v1\n"
        "      heading_path: [Methods]\n"
        "      paragraph_index: 1\n"
        "      sentence_start: 1\n"
        "      sentence_end: 1\n"
    )
    _install_fake_anthropic(monkeypatch, _FakeClient(yaml_body))

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli_main([
            "--root", str(ws.root),
            "ingest", str(src),
            "--model-provider", "anthropic",
            "--model", "claude-test-1",
        ])
    assert rc == 0, buf.getvalue()

    journal = OperationJournal(ws)
    entries = [e for e in journal.iter_entries() if e.op_kind == "ingest"]
    assert len(entries) == 1
    entry = entries[0]
    logs = entry.invocation_logs
    assert len(logs) == 1, f"expected exactly one invocation log, got {logs}"
    log = logs[0]
    assert log["model_identifier"] == "anthropic/claude-test-1"
    # Summary only — no raw source text in any log field.
    for read in log.get("read_inputs", []):
        for key, value in read.items():
            if isinstance(value, str):
                assert "Complementarity prioritizes" not in value, (
                    f"raw source text leaked into invocation log {log}"
                )
