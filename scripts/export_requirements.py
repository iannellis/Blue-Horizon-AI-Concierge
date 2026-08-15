"""Regenerate the pinned ``requirements.txt`` files used for deployment and CI.

``deploy/requirements.txt`` and ``eval/requirements.txt`` are plain
pip-installable dependency lists exported from ``uv.lock`` -- they are what
``Dockerfile`` and the CI eval job actually install from, not
``pyproject.toml`` directly. Historically these were produced with
``poetry export --with ui`` / ``--with ui,eval``, which never emitted the
root project package itself, regardless of Poetry's package-mode setting.

``uv export`` behaves differently: it *does* emit the current project by
default whenever ``pyproject.toml`` is in package mode (the default; see
``[tool.uv] package`` if that key is present). ``Dockerfile`` only copies
``blue_horizon/`` and ``ui/`` into the image, not ``pyproject.toml`` or the
build backend, so an export that includes the project itself would give pip
a package it has no source tree to build -- breaking the image build. This
script always passes ``--no-emit-project`` to rule that out, so it -- not a
bare ``uv export`` invocation -- is the supported way to regenerate these
files.

Run with:
    uv run python scripts/export_requirements.py
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RequirementsTarget:
    """One ``requirements.txt`` file to regenerate from ``uv.lock``.

    Attributes:
        groups: Dependency groups (from ``pyproject.toml``'s
            ``[dependency-groups]``) to include, in addition to the project's
            base dependencies.
        output_path: File to write the exported requirements to, relative to
            the repository root.

    """

    groups: tuple[str, ...]
    output_path: Path


REQUIREMENTS_TARGETS = (
    # Mirrors the old `poetry export --with ui`.
    RequirementsTarget(groups=("ui",), output_path=Path("deploy/requirements.txt")),
    # Mirrors the old `poetry export --with ui,eval`.
    RequirementsTarget(
        groups=("ui", "eval"), output_path=Path("eval/requirements.txt"),
    ),
)


def main() -> None:
    """Regenerate every file in `REQUIREMENTS_TARGETS` from `uv.lock`.

    Raises:
        subprocess.CalledProcessError: If any `uv export` invocation fails,
            e.g. because the lockfile is out of sync with `pyproject.toml`.

    """
    for target in REQUIREMENTS_TARGETS:
        _export_requirements_file(target)


def _export_requirements_file(target: RequirementsTarget) -> None:
    """Export one dependency-group combination to a pinned requirements file.

    Args:
        target: Which dependency groups to export and where to write them.

    Raises:
        subprocess.CalledProcessError: If the `uv export` invocation fails.

    """
    command = ["uv", "export", "--format", "requirements.txt", "--no-hashes",
        "--no-header", "--no-annotate", "--no-emit-project"]
    for group in target.groups:
        command += ["--group", group]
    output_path = REPO_ROOT / target.output_path
    command += ["-o", str(output_path)]

    subprocess.run(command, cwd=REPO_ROOT, check=True)  # noqa: S603
    print(f"Wrote {target.output_path}")  # noqa: T201


if __name__ == "__main__":
    main()
