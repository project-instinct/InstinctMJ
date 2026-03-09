# Contributing to InstinctMJ

Thank you for helping improve `InstinctMJ`.

This repository is the `mjlab`-native port of InstinctLab tasks, so contributions should prioritize behavior fidelity, clear validation, and minimal, focused changes.

## Before You Start

- For larger features or refactors, open an issue first so scope and compatibility can be discussed before implementation.
- Keep changes local to `InstinctMJ` unless the change clearly requires coordination with `mjlab` or `instinct_rl`.
- Preserve existing task semantics unless the pull request explicitly proposes a behavior change.

## Development Setup

Clone the sibling repositories and install them in editable mode:

```bash
git clone https://github.com/mujocolab/mjlab.git
git clone https://github.com/project-instinct/instinct_rl.git
git clone https://github.com/project-instinct/InstinctMJ.git

cd InstinctMJ
uv sync
```

If you use `pip`, install `../mjlab`, `../instinct_rl`, and `.` in editable mode.

## Contribution Guidelines

- Keep pull requests scoped and avoid mixing unrelated refactors with task logic changes.
- Match the existing code style and repository structure.
- Prefer `mjlab`-native implementations over compatibility layers or adapters.
- Preserve task IDs, training workflow, and command-line entry points unless the change intentionally updates public interfaces.
- Update documentation when user-facing behavior, setup, or commands change.

## Validation

At minimum, include the checks you ran in the pull request description. Typical validation includes:

- `python -m py_compile ...` for changed Python modules
- task import / registration checks such as `instinct-list-envs`
- focused train / play / script commands relevant to the changed task family

If a change is not easily testable in the current environment, explain what was verified and what still needs manual validation.

## Pull Request Checklist

Before submitting a PR, please confirm:

- the change is scoped to the intended behavior or repository maintenance task
- documentation and config references are updated when needed
- validation commands and outcomes are included in the PR description
- you have read and agree to `CONTRIBUTOR_AGREEMENT.md`

## License

By contributing to this repository, you agree that your contributions are provided under the repository license in `LICENSE` and the terms described in `CONTRIBUTOR_AGREEMENT.md`.
