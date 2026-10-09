# Tamasha — Migration Summary (v5.0)
- Removed AGENTS_FIX.md
- Cleaned PROJECT_OVERVIEW.md
- Added v5.0 reporting artifacts

# Tamasha — Migration Summary (2026-08-11 restructure pass)
- Moved `docs/migration_summary.md` → `docs/migration/migration_summary.md` (history preserved via `git mv`).
- Rewrote `docs/architecture.md` + `docs/folder_structure.md` (were 3–10 line stubs).
- Added `docs/module_dependency.md`, `docs/startup_flow.md`, `docs/package_overview.md`, `docs/migration/old_tree_to_new_tree.md`, `docs/migration/file_move_ledger.md`.
- **No application code moved** — Tamasha already conformed to the enterprise skeleton (src-layout core package, `api/` + `app/` interface layers, `tests/`, canonical artifact dirs).

## Verification
- `py_compile` sweep: all OK; `import tamasha` (0.1.0), `tamasha.predict`, `tamasha.config`: OK.
- pytest: **113 passed, 28 failed — all pre-existing environment issues, none caused by this docs-only pass**:
  - 27 failures: `prometheus_fastapi_instrumentator` 7.0.1 ↔ `fastapi` 0.139.2 incompatibility (`_IncludedRouter` has no `path` attribute) — affects every TestClient-based API/auth/contract test. Loose pins in `requirements.txt` (`fastapi>=0.104,<1.0`, `prometheus_fastapi_instrumentator>=6.1`) let pip resolve the incompatible pair. **Backlog:** pin a compatible pair (e.g. fastapi<0.118 or instrumentator<7) and re-run.
  - 1 failure: `test_10k_rows_completes_under_threshold` — time-threshold scale test, machine-speed dependent. **Backlog:** review threshold.
- No stale references to the old `docs/migration_summary.md` path anywhere in the repo.

---

## Phase 3 Re-run — Full Protocol Verification (2026-08-12)

**Mandate:** Full re-execution of the Principal Architect restructuring protocol; zero-regression; evidence-backed Phase 7.

**Discovery (P1) / Classification (P2) / Target conformance (P3):** Structure conforms (api/, app/, src/, models/, scripts/, tests/, ops/).

**Moves (P4) & Naming (P5):** No moves required this pass. Banned-token scan: clean.

**Verification (P7) — evidence:**
| Check | Command | Result |
|---|---|---|
| Import resolution | python -c 'import api.main' | OK (CORS configured) |
| Lint (criticals) | python -m ruff check . --select=E9,F63,F7,F82 | 0 errors |
| Syntax compile | py_compile on all .py | OK |
| Tests | python -m pytest -q | 114 passed, 27 failed |

**Risk & Rollback (P8):** No moves — no new risk.

**Follow-up backlog (P9):**
- 27 TestClient failures are the known pre-existing fastapi/prometheus_fastapi_instrumentator loose-pin pairing issue (backlog item from Phase 2, unchanged). Pin compatible versions to resolve.

---

## Phase 3 Addendum — prometheus-fastapi-instrumentator pin fix (2026-08-12)

**Bug (pre-existing):** 27 TestClient tests failed with `AttributeError: '_IncludedRouter' object has no attribute 'path'` raised in `prometheus_fastapi_instrumentator/routing.py:_get_route_name`. Instrumentator 7.0.1 predates FastAPI 0.116+'s internal `include_router` change (routes are now `_IncludedRouter` objects without `.path`); every instrumented request crashed.

**Fix:** pinned `prometheus-fastapi-instrumentator>=8.1,<9.0` (8.1.0) in requirements.txt and setup.py (was `>=6.1` / `>=6.0,<7.0`) — 8.1.0 specifically fixed the `_IncludedRouter` crash. Installed 8.1.0 (+ starlette 0.52.1 → 1.6.0; fastapi 0.139.2 unchanged).

**Verification (evidence):**
| Check | Command | Result |
|---|---|---|
| Full suite | `python -m pytest -q` | 140 passed, 1 failed (was 114 passed, 27 failed) |
| Remaining failure | `tests/test_clash_detection_scale.py` | pre-existing machine-speed threshold issue (10k rows exceeds 8s threshold on this host even in isolation; flagged in Phase 2 backlog) — unrelated to this fix |

**Cross-repo regression (env-wide starlette 0.52→1.6 upgrade):** IMS 342 passed; Smart-Spam-Detector 108+1 pre-existing; Dabba CI-equivalent 221 passed; CCFD 451 passed; Veridoc 179 passed; import checks OK for AegisAI/NextGen/Price-My-Car/finsight/Emotion-Lens.

**CI status:** CI installs from requirements.txt → now resolves instrumentator 8.1.0 → TestClient failures resolved.

_Amendment: also raised the fastapi floor to `>=0.116,<1.0` in requirements.txt (was `>=0.104`) so fresh CI installs always resolve a fastapi compatible with instrumentator 8.x; synced `src/tamasha.egg-info/requires.txt` generated metadata._
