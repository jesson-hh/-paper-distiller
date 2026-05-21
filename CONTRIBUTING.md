# Contributing to paper-distiller

Thanks for your interest. paper-distiller is a small-team open source project — contributions of any size welcome, from typo fixes to feature work.

## Quick start

```bash
git clone https://github.com/jesson-hh/paper-distiller
cd paper-distiller
pip install -e ".[dev]"

# Set up env so tests can find an LLM (mocked in tests, but env vars must parse)
cp examples/example.env .env
# Edit .env — only PD_API_KEY/PD_BASE_URL/PD_MODEL are required for imports

pytest -v       # should be 436+ tests, ~70s on a recent laptop
```

## Workflow

1. **Open an issue first** for anything bigger than a typo. Discuss approach before writing code — saves rework.
2. **Branch from `main`**. Branch name format: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`.
3. **TDD if practical**. Write a failing test first, then make it pass. See `tests/` for patterns.
4. **Run the full suite** before pushing: `pytest -v`. CI runs the same matrix.
5. **Lint with ruff**: `ruff check src tests`. We don't enforce a formatter — keep code readable.
6. **Commit messages**: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `release:` prefixes (Conventional Commits-ish, but not strictly enforced).
7. **PR**: target `main`, link the issue. CI must pass. A reviewer will look within a few days.

## Code conventions

- **Python 3.10+**. We use type hints, `from __future__ import annotations`, and union syntax `int | None`.
- **Modules are small and focused**. Files over 600 LOC usually indicate a missing abstraction — split.
- **Public API**: only what's exported from `paper_distiller.__init__`. Everything else is private and may change.
- **Tests**: one test file per module, named `tests/<path mirroring src>/test_<module>.py`. Use `pytest-mock` (`mocker` fixture) for stubs.
- **No emoji in code** unless the user requested it. Status icons in `chat/ui.py` are the exception.
- **Chinese-primary output** for vault content, but **English-primary code + comments**.

## Architecture orientation

```
src/paper_distiller/
├── chat/            # REPL agent (agent_loop, slash_commands, permissions, ui)
├── agents/          # Async DAG sub-agents (searchers, processor, writer, ...)
├── arxiv_local/     # Local arXiv mirror (SQLite + FTS5 + OAI-PMH sync)
├── proofs/          # Per-vault theorem / technique knowledge base
├── llm/             # OpenAI-compatible client + streaming + pricing
├── sources/         # arXiv + Semantic Scholar API wrappers
├── vault/           # Markdown CRUD + crosslink index
├── distill/         # LLM prompts + extraction logic
├── pipeline.py      # Top-level fetch+distill orchestration
├── config.py        # Env-based Config dataclass
└── prompts/         # Editable .md prompts for distill / filter / survey
```

`docs/ARCHITECTURE.md` has the deeper module map.

## Testing

- **Mock external services**. Never hit live arxiv/SS/OpenAlex in tests — they're rate-limited and flaky.
- **Isolate side effects**. `tests/conftest.py` provides:
  - `_isolate_arxiv_local` (autouse): per-test `PD_ARXIV_LOCAL_DIR` tmp dir
  - `_reset_rate_limiters` (autouse): wipes SourceLimiter cooldowns between tests
- **Integration tests** in `tests/integration/` go end-to-end through the CLI with everything mocked.
- **TDD red-green discipline**: see `superpowers:test-driven-development` patterns if you've worked with that skill set.

## Releasing (maintainers only)

1. Update `__version__` in 3 places: `pyproject.toml`, `src/paper_distiller/__init__.py`, `tests/test_smoke.py`.
2. Add a `## [vX.Y.Z]` section to `CHANGELOG.md` with the date and grouped (Added / Changed / Fixed / Internal).
3. `git tag -a vX.Y.Z -m "..."` and push the tag.
4. `.github/workflows/release.yml` builds + uploads to PyPI via OIDC trusted publishing (no token needed).

## Questions

- Open a [Discussions](https://github.com/jesson-hh/paper-distiller/discussions) thread for design-level questions.
- Use [Issues](https://github.com/jesson-hh/paper-distiller/issues) for concrete bugs / features.

## License

By contributing, you agree your work is released under the [MIT License](LICENSE).
