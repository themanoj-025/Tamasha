# Tamasha — File Move Ledger

## This pass (2026-08-11)

| Old path | New path | Category | Reason | Risk | Verified |
|---|---|---|---|---|---|
| `docs/migration_summary.md` | `docs/migration/migration_summary.md` | Meta/docs | Consolidate migration records under `docs/migration/` per enterprise standard | Low (docs only) | ✅ no inbound refs; `git mv` preserved history |

## Prior pass (v5.0, already committed)

The v5.0 restructuring (commits `14fd658`, `b4dc3d4`) moved application code
into the current skeleton. Its own migration record is preserved at
`docs/migration/migration_summary.md` and references the removals/cleanups
performed then (e.g. removal of `AGENTS_FIX.md`, PROJECT_OVERVIEW cleanup,
v5.0 reporting artifacts).

## Non-moves (documented decisions)

| Path | Decision | Reason |
|---|---|---|
| `src/tamasha/**` | keep | Canonical src-layout core package; installed via `setup.py`/pyproject |
| `api/**`, `app/**` | keep | Framework-canonical interface layers (FastAPI include_router, Streamlit multipage); Docker/Render entry contract |
| `tests/**` | keep | Already under `tests/` with `conftest.py` |
| `data/**`, `models/**`, `reports/**`, `ops/**` | keep | Already canonical artifact directories |
| `.env`, `.streamlit/secrets.toml`, `*.egg-info/`, `cache.db*` | leave untracked | Correctly gitignored; never committed |
| `src/tamasha.egg-info/` (on disk) | leave untracked | Build artifact, gitignored by `*.egg-info/` — flagged, not deleted (per protocol) |
