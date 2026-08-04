"""Contract tests for the ``llloom page create`` CLI surface (Slice 084).

Pins eight load-bearing properties of the new ``page`` command group:

1. happy-path: ``llloom page create concept/foo`` writes
   ``pages/concepts/foo.md`` with the expected frontmatter / markers
   and the generated stub parses with :func:`parse_page`.
2. follow-up ``llloom render --dry-run page:concept/foo`` exits 0 and
   reports ``marker_health: ok`` (no contributors is acceptable).
3. explicit ``--page-class`` plus bare page id writes the stub with
   the bare ``page_id`` in frontmatter and the operator-supplied
   title.
4. existing pages refuse with exit code 1 and no overwrite.
5. path-traversal / absolute / ``.md`` / backslash page ids refuse
   with exit code 1 and create no files outside ``pages/``.
6. ambiguous page ids without ``--page-class`` refuse cleanly with
   an explanatory message.
7. the success path completes one ``op.page_create.*`` journal
   entry, records the planned page write, and leaves no workspace
   lock file behind.
8. the CLI verb-count guard is updated separately
   (``test_prepare_pdf_cli.py``).

These tests are end-to-end through ``llloom.cli.main`` so they
exercise the same path an operator hits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llloom.cli import main
from llloom.pages.regions import parse_page
from llloom.state.journal import OperationJournal
from llloom.state.lock import LOCK_FILENAME
from llloom.workspace.layout import Workspace


def _run(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    code = main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_page_create_concept_foo_writes_valid_stub(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = Workspace.init(tmp_path)
    code, out, err = _run(
        ["--root", str(tmp_path), "page", "create", "concept/foo"],
        capsys,
    )
    assert code == 0, err
    payload = json.loads(out)
    assert payload["page_id"] == "concept/foo"
    assert payload["page_class"] == "concept"
    assert payload["page_path"] == "pages/concepts/foo.md"
    assert payload["claim_block_id"] == "claim_block.concept.foo"
    assert payload["commentary_id"] == "commentary.concept.foo"
    assert payload["status"] == "draft"
    assert payload["refusal_reason"] is None
    assert payload["op_id"].startswith("op.page_create.")

    page_file = ws.root / "pages" / "concepts" / "foo.md"
    assert page_file.is_file()
    text = page_file.read_text(encoding="utf-8")

    parsed = parse_page(text)
    assert parsed.frontmatter["page_id"] == "concept/foo"
    assert parsed.frontmatter["page_class"] == "concept"
    assert parsed.frontmatter["status"] == "draft"
    assert parsed.frontmatter["write_policy"] == "mixed"
    assert parsed.claim_block_id == "claim_block.concept.foo"
    assert parsed.commentary_id == "commentary.concept.foo"
    assert parsed.claim_block_inner == ""
    assert parsed.commentary_inner == ""
    assert "# Foo" in text

    # No claims, no other pages, no fingerprint mutation.
    assert list(ws.claims_entities.glob("*.yaml")) == []
    fingerprints = (ws.root / "state" / "render_fingerprints.yaml").read_text(
        encoding="utf-8"
    )
    assert fingerprints == "fingerprints: {}\n"


def test_page_create_preserves_dotted_page_id_tail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Slice 084a regression: a dotted page id such as
    ``concept/foo.bar`` must land at ``pages/concepts/foo.bar.md``
    (not ``pages/concepts/foo.md``) and must be renderable through
    ``llloom render --dry-run page:concept/foo.bar`` with
    ``marker_health: ok``. Before the Slice 084a fix
    ``_resolve_page_path(...)`` used
    ``Path.with_suffix(".md")`` which replaced the existing
    ``.bar`` suffix on the path tail and silently dropped the rest
    of the page id.
    """
    ws = Workspace.init(tmp_path)
    create_code, out, err = _run(
        ["--root", str(tmp_path), "page", "create", "concept/foo.bar"],
        capsys,
    )
    assert create_code == 0, err
    payload = json.loads(out)
    assert payload["page_id"] == "concept/foo.bar"
    assert payload["page_class"] == "concept"
    assert payload["page_path"] == "pages/concepts/foo.bar.md"
    assert payload["claim_block_id"] == "claim_block.concept.foo.bar"
    assert payload["commentary_id"] == "commentary.concept.foo.bar"

    page_file = ws.root / "pages" / "concepts" / "foo.bar.md"
    assert page_file.is_file()
    short_file = ws.root / "pages" / "concepts" / "foo.md"
    assert not short_file.exists(), (
        "Slice 084a regression: dotted page id was truncated to "
        f"{short_file}"
    )

    parsed = parse_page(page_file.read_text(encoding="utf-8"))
    assert parsed.frontmatter["page_id"] == "concept/foo.bar"
    assert parsed.claim_block_id == "claim_block.concept.foo.bar"
    assert parsed.commentary_id == "commentary.concept.foo.bar"

    render_code, render_out, _ = _run(
        [
            "--root",
            str(tmp_path),
            "render",
            "--dry-run",
            "page:concept/foo.bar",
        ],
        capsys,
    )
    assert render_code == 0
    render_payload = json.loads(render_out)
    plan = render_payload["plan"]
    assert len(plan) == 1
    assert plan[0]["page_id"] == "concept/foo.bar"
    assert plan[0]["marker_health"] == "ok"

    # The journal entry for this op must record the dotted path, not
    # the truncated path that the Slice 084 implementation produced.
    op_id = payload["op_id"]
    journal = OperationJournal(ws)
    assert journal.exists(op_id)
    entry = journal.load(op_id)
    assert entry.op_kind == "page_create"
    assert entry.status == "completed"
    assert entry.planned_writes == ["pages/concepts/foo.bar.md"]
    assert "pages/concepts/foo.bar.md" in entry.touched_files


def test_page_create_then_render_dry_run_marker_health_ok(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Workspace.init(tmp_path)
    create_code, _, _ = _run(
        ["--root", str(tmp_path), "page", "create", "concept/foo"],
        capsys,
    )
    assert create_code == 0

    render_code, render_out, _ = _run(
        [
            "--root",
            str(tmp_path),
            "render",
            "--dry-run",
            "page:concept/foo",
        ],
        capsys,
    )
    assert render_code == 0
    payload = json.loads(render_out)
    assert payload["dry_run"] is True
    plan = payload["plan"]
    assert len(plan) == 1
    entry = plan[0]
    assert entry["target"] == "page:concept/foo"
    assert entry["page_id"] == "concept/foo"
    assert entry["marker_health"] == "ok"


def test_page_create_with_explicit_class_keeps_bare_page_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = Workspace.init(tmp_path)
    code, out, err = _run(
        [
            "--root",
            str(tmp_path),
            "page",
            "create",
            "foo",
            "--page-class",
            "concept",
            "--title",
            "Foo",
        ],
        capsys,
    )
    assert code == 0, err
    payload = json.loads(out)
    assert payload["page_id"] == "foo"
    assert payload["page_class"] == "concept"
    assert payload["page_path"] == "pages/concepts/foo.md"
    assert payload["claim_block_id"] == "claim_block.foo"
    assert payload["commentary_id"] == "commentary.foo"

    page_file = ws.root / "pages" / "concepts" / "foo.md"
    text = page_file.read_text(encoding="utf-8")
    assert "page_id: foo\n" in text
    assert "page_class: concept\n" in text
    assert "# Foo\n" in text

    parsed = parse_page(text)
    assert parsed.frontmatter["page_id"] == "foo"
    assert parsed.claim_block_id == "claim_block.foo"


def test_page_create_refuses_existing_page_with_no_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = Workspace.init(tmp_path)
    first, _, _ = _run(
        ["--root", str(tmp_path), "page", "create", "concept/foo"],
        capsys,
    )
    assert first == 0

    page_file = ws.root / "pages" / "concepts" / "foo.md"
    original = page_file.read_text(encoding="utf-8")

    second, out, err = _run(
        ["--root", str(tmp_path), "page", "create", "concept/foo"],
        capsys,
    )
    assert second == 1
    assert "llloom page create:" in err
    assert "already exists" in err
    # No traceback bleeds through.
    assert "Traceback" not in err

    # File is byte-identical: no overwrite.
    assert page_file.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "bad_page_id",
    [
        "concept/foo.md",
        "concept\\foo",
        "/absolute/foo",
        "concept/../bar",
        "concept/./bar",
        "",
    ],
)
def test_page_create_refuses_malformed_page_ids(
    bad_page_id: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ws = Workspace.init(tmp_path)
    # The empty string is rejected by argparse before our handler
    # because the positional arg is required to be non-empty by
    # convention; we still pass it through main() and assert the
    # process exits non-zero. argparse exits 2 on argv parse errors;
    # our PageCreateError refusals exit 1. Either is a clean refusal.
    argv = ["--root", str(tmp_path), "page", "create"]
    if bad_page_id:
        argv.append(bad_page_id)
    try:
        code = main(argv)
    except SystemExit as exc:  # argparse may sys.exit on missing arg
        code = int(exc.code) if exc.code is not None else 2
    captured = capsys.readouterr()
    assert code != 0, captured.out

    # Nothing should be created outside pages/<class>/<existing>/.
    existing = sorted(
        p.relative_to(ws.root).as_posix()
        for p in ws.pages.rglob("*.md")
    )
    # Only the starter overview.md may be present.
    assert existing == ["pages/overview.md"]


def test_page_create_refuses_when_class_cannot_be_inferred(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Workspace.init(tmp_path)
    code, _, err = _run(
        ["--root", str(tmp_path), "page", "create", "foo"], capsys
    )
    assert code == 1
    assert "llloom page create:" in err
    assert "cannot infer page class" in err
    assert "--page-class" in err
    assert "Traceback" not in err


def test_page_create_refuses_conflicting_inferred_and_explicit_class(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    Workspace.init(tmp_path)
    code, _, err = _run(
        [
            "--root",
            str(tmp_path),
            "page",
            "create",
            "concept/foo",
            "--page-class",
            "synthesis",
        ],
        capsys,
    )
    assert code == 1
    assert "conflicting page class" in err


def test_page_create_success_writes_one_completed_journal_entry(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ws = Workspace.init(tmp_path)
    code, out, _ = _run(
        ["--root", str(tmp_path), "page", "create", "concept/foo"],
        capsys,
    )
    assert code == 0
    payload = json.loads(out)
    op_id = payload["op_id"]

    journal = OperationJournal(ws)
    assert journal.exists(op_id)
    entry = journal.load(op_id)
    assert entry.op_kind == "page_create"
    assert entry.status == "completed"
    assert entry.planned_writes == ["pages/concepts/foo.md"]
    assert "pages/concepts/foo.md" in entry.touched_files

    # No workspace lock file left behind.
    lock_file = ws.state_locks / LOCK_FILENAME
    assert not lock_file.exists()


# ---- M003/S01: the explicit --framework-profile opt-in ----------------------
#
# Added by Coding Prompt 014. These pin the CLI request surface, the legacy
# freeze, and the deterministic profiled bytes end-to-end through
# ``llloom.cli.main``. Expected bytes are independent literals, never produced
# by calling the production renderer under test.

CANDIDATE = "0.1-rc.1"
PROFILE_BLOCK_LF = b'type: page\nframework_profile: "0.1-rc.1"\n'


def _expected_profiled_lf(page_id: str, page_class: str, title: str, tail: str) -> bytes:
    """Independent literal oracle for the profiled LF rendering."""
    return (
        "---\n"
        "type: page\n"
        'framework_profile: "0.1-rc.1"\n'
        f"page_id: {page_id}\n"
        f"page_class: {page_class}\n"
        "write_policy: mixed\n"
        "status: draft\n"
        "---\n"
        "\n"
        f"# {title}\n"
        "\n"
        f"<!-- llloom:claim-block id=claim_block.{tail} -->\n"
        "\n"
        "<!-- /llloom:claim-block -->\n"
        "\n"
        f"<!-- llloom:commentary id=commentary.{tail} owner=human -->\n"
        "\n"
        "<!-- /llloom:commentary -->\n"
    ).encode("utf-8")


def test_page_create_help_exposes_only_the_single_accepted_profile_value(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["page", "create", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--framework-profile" in out
    assert CANDIDATE in out
    # No boolean shorthand, alias, or extra vocabulary.
    for absent in ("--profile ", "--okf", "--profiled", "--framework-id"):
        assert absent not in out


def test_unknown_profile_value_is_a_deterministic_argparse_refusal(tmp_path, capsys):
    Workspace.init(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--root",
                str(tmp_path),
                "page",
                "create",
                "concept/foo",
                "--framework-profile",
                "0.1",
            ]
        )
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err
    assert "Traceback" not in err
    assert not (tmp_path / "pages" / "concepts" / "foo.md").exists()


def test_cli_page_create_verb_and_subcommand_inventory_is_unchanged(capsys):
    """The opt-in adds no command, group, or subcommand."""
    with pytest.raises(SystemExit) as excinfo:
        main(["page", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "create" in out
    for absent in ("profile", "publish", "emit", "convert", "migrate"):
        assert f"    {absent}" not in out


def test_cli_legacy_page_create_output_is_unchanged_by_the_opt_in(tmp_path, capsys):
    ws = Workspace.init(tmp_path)
    code, out, err = _run(
        ["--root", str(tmp_path), "page", "create", "concept/legacy", "--title", "Legacy"],
        capsys,
    )
    assert code == 0
    assert err == ""
    payload = json.loads(out)
    assert set(payload) == {
        "page_id",
        "page_class",
        "page_path",
        "claim_block_id",
        "commentary_id",
        "status",
        "op_id",
        "refusal_reason",
    }
    assert payload["refusal_reason"] is None
    raw = (ws.root / payload["page_path"]).read_bytes()
    assert b"framework_profile" not in raw
    assert b"type: page" not in raw


def test_cli_profiled_creation_emits_the_exact_contract_bytes(tmp_path, capsys):
    ws = Workspace.init(tmp_path)
    code, out, err = _run(
        [
            "--root",
            str(tmp_path),
            "page",
            "create",
            "concept/handoff",
            "--title",
            "Handoff",
            "--framework-profile",
            CANDIDATE,
        ],
        capsys,
    )
    assert code == 0
    assert err == ""
    payload = json.loads(out)
    # The serialized JSON shape is identical on both paths.
    assert set(payload) == {
        "page_id",
        "page_class",
        "page_path",
        "claim_block_id",
        "commentary_id",
        "status",
        "op_id",
        "refusal_reason",
    }
    assert "framework_profile" not in payload
    assert "profile_valid" not in payload

    raw = (ws.root / payload["page_path"]).read_bytes()
    assert raw == _expected_profiled_lf(
        "concept/handoff", "concept", "Handoff", "concept.handoff"
    )
    assert b"\r\n" not in raw
    parse_page(raw.decode("utf-8"))


def test_cli_profiled_creation_journals_one_completed_entry_without_temp_paths(
    tmp_path, capsys
):
    ws = Workspace.init(tmp_path)
    code, out, _ = _run(
        [
            "--root",
            str(tmp_path),
            "page",
            "create",
            "concept/journaled",
            "--framework-profile",
            CANDIDATE,
        ],
        capsys,
    )
    assert code == 0
    payload = json.loads(out)

    journal = OperationJournal(ws)
    entry = journal.load(payload["op_id"])
    assert entry.op_kind == "page_create"
    assert entry.status == "completed"
    assert entry.planned_writes == ["pages/concepts/journaled.md"]
    assert "pages/concepts/journaled.md" in entry.touched_files
    # No temp path is ever exposed as a durable planned/touched artifact.
    assert not any(".tmp" in p for p in entry.planned_writes)
    assert not any(".tmp" in p for p in entry.touched_files)

    assert not (ws.state_locks / LOCK_FILENAME).exists()
    assert list(ws.pages.rglob("*.tmp")) == []


def test_cli_profiled_creation_refuses_an_existing_target_without_mutation(
    tmp_path, capsys
):
    ws = Workspace.init(tmp_path)
    target = ws.pages / "concepts" / "taken.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"ORIGINAL CONTENT\n")

    code, _, err = _run(
        [
            "--root",
            str(tmp_path),
            "page",
            "create",
            "concept/taken",
            "--framework-profile",
            CANDIDATE,
        ],
        capsys,
    )
    assert code == 1
    assert "Traceback" not in err
    assert target.read_bytes() == b"ORIGINAL CONTENT\n"
    assert list(ws.pages.rglob("*.tmp")) == []
