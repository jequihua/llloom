"""Guard test for the synthetic fixture manifest.

The manifest at ``tests/fixtures/synthetic/manifest.yaml`` is the
documented index of synthetic fixture directories, the invariant each
proves, and the test file(s) that exercise the scenario. This guard
keeps the manifest honest: every listed fixture directory and every
listed test file must exist, and active-consumer entries must
reference the fixture by directory name or by filename so a dead-link
manifest cannot accumulate silently.

Active vs documentation-only:

- ``documentation_only: false`` — the test reads files from the
  fixture directory; the guard asserts the test file references either
  the fixture directory name or one of its listed filenames.
- ``documentation_only: true`` — the scenario is implemented inline in
  the test code; the guard only asserts the test file exists.
"""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
MANIFEST_PATH = SYNTHETIC_ROOT / "manifest.yaml"

REQUIRED_KEYS = {"directory", "invariant", "files", "tests"}
EXPECTED_DIRECTORIES = {
    "aliases",
    "canary",
    "concurrency",
    "denied",
    "precision_critical",
    "structured",
}


def test_synthetic_fixture_manifest_is_honest() -> None:
    """One guard covering: schema, directory existence, file existence,
    consumer-test existence, and the reference check for active
    consumers. Documentation-only entries skip the reference check.
    """
    data = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "manifest must be a YAML mapping"
    fixtures = data.get("fixtures")
    assert isinstance(fixtures, list), "manifest must declare a `fixtures` list"

    named = {entry["directory"] for entry in fixtures}
    assert named == EXPECTED_DIRECTORIES, (
        f"manifest must list exactly the six live synthetic fixture "
        f"directories; got {sorted(named)}, expected "
        f"{sorted(EXPECTED_DIRECTORIES)}"
    )

    active_seen = False
    for entry in fixtures:
        missing = REQUIRED_KEYS - set(entry)
        assert not missing, (
            f"manifest entry {entry.get('directory')!r} is missing "
            f"required keys: {sorted(missing)}"
        )
        directory = entry["directory"]
        invariant = entry["invariant"]
        files = entry["files"]
        tests = entry["tests"]
        assert isinstance(directory, str) and directory
        assert isinstance(invariant, str) and invariant.strip(), (
            f"manifest entry {directory!r}: `invariant` is empty"
        )
        assert isinstance(files, list), (
            f"manifest entry {directory!r}: `files` must be a list"
        )
        assert isinstance(tests, list) and tests, (
            f"manifest entry {directory!r}: `tests` must be a non-empty list"
        )

        fixture_dir = SYNTHETIC_ROOT / directory
        assert fixture_dir.is_dir(), (
            f"manifest entry {directory!r}: directory does not exist at "
            f"{fixture_dir}"
        )
        for filename in files:
            assert isinstance(filename, str), (
                f"manifest entry {directory!r}: every entry in `files` "
                f"must be a string filename"
            )
            assert (fixture_dir / filename).is_file(), (
                f"manifest entry {directory!r}: listed file {filename!r} "
                f"does not exist under {fixture_dir}"
            )

        for test_rel in tests:
            assert isinstance(test_rel, str), (
                f"manifest entry {directory!r}: every entry in `tests` "
                f"must be a workspace-relative string path"
            )
            test_path = REPO_ROOT / test_rel
            assert test_path.is_file(), (
                f"manifest entry {directory!r}: listed test {test_rel!r} "
                f"does not exist at {test_path}"
            )

            if entry.get("documentation_only", False):
                continue
            active_seen = True
            tokens = [directory, *files]
            test_text = test_path.read_text(encoding="utf-8")
            assert any(token in test_text for token in tokens), (
                f"manifest entry {directory!r}: test {test_rel!r} does "
                f"not reference the fixture directory or any listed "
                f"file (looked for any of {tokens!r}). Either restore "
                f"the reference or mark the entry "
                f"`documentation_only: true`."
            )

    assert active_seen, (
        "manifest must declare at least one active-consumer fixture so "
        "the reference check is exercised"
    )
