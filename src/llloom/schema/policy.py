"""Schema loading and policy resolution.

Loads the four required schema files and provides typed access to
ingest policies, source classes, page classes, and the spine manifest.

See ``04_specification/operations_and_cli.md`` Â§ingest for the five
policy names this module freezes, and
``04_specification/storage_and_state_model.md`` for the locator shapes
that source classes map to.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from llloom.workspace.layout import Workspace


# Five frozen ingest policies. No others are legal.
INGEST_POLICIES = frozenset(
    {
        "deny",
        "index_only",
        "structure_extract",
        "claim_extract",
        "claim_extract_and_view_render",
    }
)

IngestPolicy = str  # one of INGEST_POLICIES

# Known locator types that Phase 2 supports for span verification.
KNOWN_LOCATORS = frozenset({"markdown_prose_v1", "legal_act_v1", "code_v1"})


class SchemaError(Exception):
    """Raised for malformed or inconsistent schema files."""


@dataclass(frozen=True)
class SourceClass:
    name: str
    locator: str
    description: str = ""


@dataclass(frozen=True)
class PageClass:
    name: str
    directory: str


@dataclass
class Schema:
    """Loaded workspace schema.

    Attributes:
        source_classes: by name.
        ingest_policies: source-class-name -> policy.
        unknown_policy: fallback policy for unmapped source classes.
        page_classes: by name.
        spine_files: explicit relative paths that belong to the spine.
        spine_globs: glob patterns treated as spine when matched.
    """

    source_classes: dict[str, SourceClass]
    ingest_policies: dict[str, IngestPolicy]
    unknown_policy: IngestPolicy
    page_classes: dict[str, PageClass]
    spine_files: list[str]
    spine_globs: list[str]

    def resolve_ingest_policy(self, source_class: str) -> IngestPolicy:
        """Resolve the ingest policy for a named source class.

        Raises SchemaError if the source class is unknown AND the
        ``unknown`` default is not itself a legal policy.
        """
        if source_class in self.ingest_policies:
            return self.ingest_policies[source_class]
        if source_class in self.source_classes:
            # class exists but has no explicit policy: fall back to default
            return self.unknown_policy
        # class is unknown: fall back to default
        return self.unknown_policy

    def source_class(self, name: str) -> SourceClass:
        try:
            return self.source_classes[name]
        except KeyError as exc:
            raise SchemaError(f"unknown source class: {name}") from exc

    def is_spine(self, rel_path: str) -> bool:
        """Return True if ``rel_path`` (POSIX-style) is a spine file."""
        if rel_path in self.spine_files:
            return True
        for glob in self.spine_globs:
            if fnmatch.fnmatchcase(rel_path, glob):
                return True
        return False


def load_schema(workspace: Workspace) -> Schema:
    """Load and validate all schema files for ``workspace``."""
    src_raw = _read_yaml(workspace.schema / "source_classes.yaml")
    pol_raw = _read_yaml(workspace.schema / "ingest_policies.yaml")
    page_raw = _read_yaml(workspace.schema / "page_classes.yaml")
    spine_raw = _read_yaml(workspace.schema / "spine_manifest.yaml")

    source_classes = _parse_source_classes(src_raw)
    ingest_policies, unknown_policy = _parse_ingest_policies(pol_raw)
    page_classes = _parse_page_classes(page_raw)
    spine_files, spine_globs = _parse_spine_manifest(spine_raw)

    # Validate locator types first so malformed source classes surface
    # before cross-reference checks against ingest_policies.
    _validate_locator_types(source_classes)
    _validate_policy_coverage(source_classes, ingest_policies)

    return Schema(
        source_classes=source_classes,
        ingest_policies=ingest_policies,
        unknown_policy=unknown_policy,
        page_classes=page_classes,
        spine_files=spine_files,
        spine_globs=spine_globs,
    )


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise SchemaError(f"missing schema file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SchemaError(f"malformed schema file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SchemaError(f"schema file {path} must be a mapping")
    return data


def _parse_source_classes(raw: dict) -> dict[str, SourceClass]:
    classes_raw = raw.get("classes", {}) or {}
    if not isinstance(classes_raw, dict):
        raise SchemaError("source_classes.yaml: `classes` must be a mapping")
    out: dict[str, SourceClass] = {}
    for name, spec in classes_raw.items():
        if not isinstance(spec, dict):
            raise SchemaError(f"source class {name!r}: must be a mapping")
        locator = spec.get("locator")
        if not isinstance(locator, str) or not locator:
            raise SchemaError(f"source class {name!r}: missing `locator`")
        out[name] = SourceClass(
            name=name,
            locator=locator,
            description=str(spec.get("description", "")),
        )
    if not out:
        raise SchemaError("source_classes.yaml: at least one class is required")
    return out


def _parse_ingest_policies(raw: dict) -> tuple[dict[str, IngestPolicy], IngestPolicy]:
    mapping_raw = raw.get("policies", {}) or {}
    if not isinstance(mapping_raw, dict):
        raise SchemaError("ingest_policies.yaml: `policies` must be a mapping")
    mapping: dict[str, IngestPolicy] = {}
    for name, policy in mapping_raw.items():
        policy_str = str(policy)
        if policy_str not in INGEST_POLICIES:
            raise SchemaError(
                f"ingest_policies.yaml: policy {policy_str!r} for {name!r} "
                f"is not one of {sorted(INGEST_POLICIES)}"
            )
        mapping[name] = policy_str
    defaults = raw.get("defaults", {}) or {}
    unknown_policy = str(defaults.get("unknown", "deny"))
    if unknown_policy not in INGEST_POLICIES:
        raise SchemaError(
            f"ingest_policies.yaml: defaults.unknown {unknown_policy!r} "
            f"is not one of {sorted(INGEST_POLICIES)}"
        )
    return mapping, unknown_policy


def _parse_page_classes(raw: dict) -> dict[str, PageClass]:
    classes_raw = raw.get("classes", {}) or {}
    if not isinstance(classes_raw, dict):
        raise SchemaError("page_classes.yaml: `classes` must be a mapping")
    out: dict[str, PageClass] = {}
    for name, spec in classes_raw.items():
        if not isinstance(spec, dict):
            raise SchemaError(f"page class {name!r}: must be a mapping")
        directory = spec.get("directory")
        if not isinstance(directory, str) or not directory:
            raise SchemaError(f"page class {name!r}: missing `directory`")
        out[name] = PageClass(name=name, directory=directory)
    return out


def _parse_spine_manifest(raw: dict) -> tuple[list[str], list[str]]:
    files = raw.get("spine_files", []) or []
    globs = raw.get("spine_globs", []) or []
    if not isinstance(files, list) or not all(isinstance(s, str) for s in files):
        raise SchemaError("spine_manifest.yaml: `spine_files` must be a list of strings")
    if not isinstance(globs, list) or not all(isinstance(s, str) for s in globs):
        raise SchemaError("spine_manifest.yaml: `spine_globs` must be a list of strings")
    return list(files), list(globs)


def _validate_policy_coverage(
    source_classes: dict[str, SourceClass],
    ingest_policies: dict[str, IngestPolicy],
) -> None:
    # It is fine for some source classes to lack explicit policy (they fall
    # back to unknown), but policies referencing undefined classes are an
    # error.
    unknown_classes: Iterable[str] = [
        name for name in ingest_policies if name not in source_classes
    ]
    unknown_list = list(unknown_classes)
    if unknown_list:
        raise SchemaError(
            "ingest_policies.yaml: references undefined source classes: "
            + ", ".join(sorted(unknown_list))
        )


def _validate_locator_types(source_classes: dict[str, SourceClass]) -> None:
    unsupported = [
        sc.name
        for sc in source_classes.values()
        if sc.locator not in KNOWN_LOCATORS
    ]
    if unsupported:
        raise SchemaError(
            "source_classes.yaml: unsupported locator types on: "
            + ", ".join(sorted(unsupported))
        )

