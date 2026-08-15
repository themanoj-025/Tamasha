# Tamasha — Old Tree → New Tree

Tamasha reached the enterprise skeleton in the earlier v5.0 restructuring pass
(commit `14fd658` / `b4dc3d4`). This migration pass only consolidates the
migration record and completes the Phase-6 documentation suite.

## Tree changes in this pass

```
Before                                After
──────                                ─────
docs/migration_summary.md      →      docs/migration/migration_summary.md
docs/architecture.md (3-line stub)    docs/architecture.md (full architecture doc)
docs/folder_structure.md (stub)       docs/folder_structure.md (annotated tree)
—                                     docs/module_dependency.md        (new)
—                                     docs/startup_flow.md             (new)
—                                     docs/package_overview.md         (new)
—                                     docs/migration/old_tree_to_new_tree.md (new)
—                                     docs/migration/file_move_ledger.md     (new)
```

## No-code-move rationale

The application code already conforms to the target skeleton:

- `src/tamasha/` — src-layout core package, domain-organized (`data`, `models`,
  `features`, `network`, `nlp`, `timing`, `cv`, `evaluation`)
- `api/` — FastAPI interface layer with lifespan DI
- `app/` — Streamlit interface layer
- `tests/` — 18 pytest modules incl. contract/regression suites
- `data/`, `models/`, `reports/`, `ops/`, `docs/` — canonical artifact dirs
- Root holds only canonical metadata (README, LICENSE, Dockerfile, compose,
  Makefile, pyproject, requirements, packages.txt, render.yaml, .gitignore)

Moving any of it would break framework conventions (Streamlit multipage,
FastAPI include_router, `python -m tamasha.*` entry points) with zero benefit —
same precedent as AegisAI and Emotion-Lens.
