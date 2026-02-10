import os
import pathlib

import nox

os.environ.update({"PDM_IGNORE_SAVED_PYTHON": "1"})

nox.options.sessions = (
    "lint",
    "test",
)


@nox.session
def format(session):
    session.run_install("pdm", "sync", external=True)
    session.run("ruff", "check", "--fix")
    session.run("ruff", "format")


@nox.session
def lint(session):
    session.run_install("pdm", "sync", "-G", "all", external=True)
    session.run("ruff", "check")
    session.run(
        "ruff",
        "format",
        "--diff",
    )
    session.run("mypy", pathlib.Path(__file__).parent / "sarkit_assurance")


@nox.session(requires=[])
def test(session):
    """Run the required tests"""
