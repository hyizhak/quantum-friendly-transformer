"""Catalogued, checksum-verified access to original experiment files."""

from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
from importlib.resources import as_file, files
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Iterator
import zipfile


@dataclass(frozen=True)
class Artifact:
    path: str
    size: int
    sha256: str
    bundled: bool
    groups: tuple[str, ...]
    related: tuple[str, ...] = ()


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.parts or path.as_posix() != value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Invalid evidence path: {value!r}")
    return path


def _destination(root: Path, name: str) -> Path:
    path = root.joinpath(*_relative_path(name).parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Evidence path escapes destination: {name!r}")
    return path


def _validate(path: Path, artifact: Artifact) -> None:
    if path.stat().st_size != artifact.size or checksum(path) != artifact.sha256:
        raise ValueError(f"Evidence checksum mismatch: {artifact.path}")


class EvidenceStore:
    """Access bundled tables and optionally a local root or companion ZIP.

    ``root`` is a directory containing ``experiments/outputs/`` (for example
    the original checkout or an extracted evidence archive). ``bundle`` is the
    large companion ZIP. Environment fallbacks are QFT_EVIDENCE_ROOT and
    QFT_EVIDENCE_BUNDLE. Files are verified before being returned.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        bundle: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        resource = files("quantum_friendly_transformer.evidence")
        catalog_bytes = resource.joinpath("catalog.json").read_bytes()
        self.catalog = json.loads(catalog_bytes)
        if self.catalog["schema_version"] != 1:
            raise ValueError("Unsupported evidence catalog version")
        self._artifacts = {}
        for record in self.catalog["artifacts"]:
            _relative_path(record["path"])
            artifact = Artifact(
                path=record["path"], size=record["size"], sha256=record["sha256"],
                bundled=record["bundled"], groups=tuple(record["groups"]),
                related=tuple(record.get("related", [])),
            )
            if artifact.path in self._artifacts:
                raise ValueError(f"Duplicate evidence path: {artifact.path}")
            self._artifacts[artifact.path] = artifact
        root = root or os.environ.get("QFT_EVIDENCE_ROOT")
        bundle = bundle or os.environ.get("QFT_EVIDENCE_BUNDLE")
        self.root = Path(root).expanduser().resolve() if root else None
        self.bundle = Path(bundle).expanduser().resolve() if bundle else None
        cache_base = cache_dir or os.environ.get("QFT_EVIDENCE_CACHE")
        if cache_base is None:
            cache_base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "qft/evidence"
        self.cache_dir = Path(cache_base).expanduser().resolve() / hashlib.sha256(catalog_bytes).hexdigest()[:16]

    def list_artifacts(
        self, group: str | None = None, bundled: bool | None = None,
    ) -> list[Artifact]:
        """List recorded files; groups are labels, not claims of run success."""
        if group is not None and group not in self.groups:
            raise ValueError(f"Unknown evidence group {group!r}; choose from {', '.join(self.groups)}")
        return [
            item for item in self._artifacts.values()
            if (group is None or group in item.groups)
            and (bundled is None or item.bundled == bundled)
        ]

    @property
    def groups(self) -> list[str]:
        return sorted({group for item in self._artifacts.values() for group in item.groups})

    def artifact(self, name: str) -> Artifact:
        """Look up a full catalog path, or a unique filename."""
        if name in self._artifacts:
            return self._artifacts[name]
        matches = [item for item in self._artifacts.values() if PurePosixPath(item.path).name == name]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ValueError(f"Ambiguous filename {name!r}; use a full catalog path")
        raise KeyError(f"Unknown evidence file: {name!r}")

    @contextmanager
    def _open_source(self, artifact: Artifact):
        if self.root is not None:
            local = _destination(self.root, artifact.path)
            if local.is_file():
                with local.open("rb") as stream:
                    yield stream
                return
        if artifact.bundled:
            resource = files("quantum_friendly_transformer.evidence").joinpath("records.zip")
            with as_file(resource) as path, zipfile.ZipFile(path) as archive:
                with archive.open(artifact.path) as stream:
                    yield stream
            return
        if self.bundle is not None:
            with zipfile.ZipFile(self.bundle) as archive:
                try:
                    stream = archive.open(artifact.path)
                except KeyError as exc:
                    raise FileNotFoundError(f"Companion bundle is missing {artifact.path}") from exc
                with stream:
                    yield stream
            return
        raise FileNotFoundError(
            f"{artifact.path} is in the companion evidence bundle, not the wheel. "
            "Set QFT_EVIDENCE_BUNDLE to that ZIP, or QFT_EVIDENCE_ROOT to a directory "
            "containing experiments/outputs/. See docs/evidence.md."
        )

    def _copy(self, artifact: Artifact, destination: Path) -> Path:
        if destination.exists():
            _validate(destination, artifact)
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with self._open_source(artifact) as source:
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                    temporary = Path(output.name)
                    digest = hashlib.sha256()
                    size = 0
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                        digest.update(block)
                        size += len(block)
            if size != artifact.size or digest.hexdigest() != artifact.sha256:
                raise ValueError(f"Evidence checksum mismatch: {artifact.path}")
            # Respect a file another process may have created while copying.
            if destination.exists():
                _validate(destination, artifact)
            else:
                temporary.replace(destination)
            return destination
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def path(self, name: str) -> Path:
        """Return a verified local path, extracting an archive entry if needed."""
        artifact = self.artifact(name)
        if self.root is not None:
            local = _destination(self.root, artifact.path)
            if local.is_file():
                _validate(local, artifact)
                return local
        return self._copy(artifact, _destination(self.cache_dir, artifact.path))

    def read_json(self, name: str):
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def load_checkpoint(self, name: str, *, map_location="cpu") -> dict:
        """Load a verified saved state; base models/adapters are reconstructed separately."""
        import torch

        artifact = self.artifact(name)
        if "checkpoints" not in artifact.groups:
            raise ValueError(f"Not a catalogued checkpoint: {name}")
        return torch.load(self.path(artifact.path), map_location=map_location, weights_only=True)

    def read_csv(self, name: str) -> list[dict[str, str]]:
        with self.path(name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def extract(
        self, destination: str | Path, *, group: str | None = None, include_bulk: bool = False,
    ) -> Iterator[Path]:
        """Restore original relative paths; refuse to overwrite different data."""
        root = Path(destination).expanduser()
        for artifact in self.list_artifacts(group, bundled=None if include_bulk else True):
            yield self._copy(artifact, _destination(root, artifact.path))

    def verify(self, *, group: str | None = None, include_bulk: bool = False) -> dict[str, str]:
        """Stream checksums without extracting files; return only failures."""
        failures = {}
        for artifact in self.list_artifacts(group, bundled=None if include_bulk else True):
            try:
                digest = hashlib.sha256()
                size = 0
                with self._open_source(artifact) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        size += len(block)
                        digest.update(block)
                if size != artifact.size or digest.hexdigest() != artifact.sha256:
                    raise ValueError("checksum mismatch")
            except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
                failures[artifact.path] = str(exc)
        return failures
