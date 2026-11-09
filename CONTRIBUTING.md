# Contributing to pyRadBio

The binding rules -- test-driven development, golden tests, licence hygiene,
data handling and numerics -- are in
[`AGENTS.md`](https://github.com/e0404/pyRadBio/blob/develop/AGENTS.md). They apply to
human and AI contributors alike. This page covers the workflow.

## Setup

```bash
git clone https://github.com/e0404/pyRadBio.git
cd pyRadBio
pip install -e ".[all]"
pre-commit install
```

## Before you open a pull request

```bash
ruff format . && ruff check --fix .
mypy pyradbio
pytest -m "not slow"
```

CI runs the same checks on Python 3.10-3.14; the `@slow` Monte-Carlo tests run
locally only. The pre-commit hooks also run codespell, validate-pyproject and a
CHANGELOG check.

## Branches and pull requests

- Branch from `develop` (`feat/...`, `fix/...`, `docs/...`) and open the pull
  request against `develop`.
- One logical change per commit, with a conventional-commit message
  (`feat(lem): ...`, `fix(mixing): ...`, `test(core): ...`).
- Add a line under `[Unreleased]` in `CHANGELOG.md`.
- `main` only receives releases (see below).

## Releases

1. Branch `rc/vX.Y.Z` from `develop`, without a version bump, and open a pull
   request against `main`.
2. Fix whatever CI or the review turns up on the `rc/` branch until the pull
   request is green and approved.
3. Only then bump the version: `version` in `pyproject.toml`, `version` and
   `date-released` in `CITATION.cff`, and a dated `[X.Y.Z]` section in
   `CHANGELOG.md` (keep an empty `[Unreleased]` above it and update the compare
   links), as a `chore(release): X.Y.Z` commit.
4. Merge the pull request.
5. Tag the merge commit on `main` as `vX.Y.Z` and push the tag. The `Publish`
   workflow checks that the tag matches `pyproject.toml` and uploads the release
   to PyPI; wait for it to succeed.
6. Merge `main` back into `develop`.

## Data and references

- Never commit PIDE or other registration-gated data. Point the loaders at
  your local copy instead (`data/pide/` or `PYRADBIO_PIDE_DIR`).
- Every published value in code or tests cites its DOI and the table or
  equation it comes from; a value you could not verify is marked
  `# UNVERIFIED` and noted in the model's summary in `docs/references/`.

## Reporting issues

Use GitHub issues. For a deviation from a paper, include the DOI, the figure or
table, and the code that reproduces it.
