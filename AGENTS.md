# AGENTS.md — Tamasha

> Canonical project instructions. Pointers like `CLAUDE.md` or
> `.github/copilot-instructions.md` should say "See AGENTS.md".

---

## Project overview

**Tamasha** — a natural-language-to-code generation sandbox (text-to-SQL
and text-to-code). Core components:

- **API** — FastAPI service exposing text-to-SQL/code generation.
- **ML** — LLM + fine-tuned code/SQL generation models.
- **Web / App** — frontend consuming the API.
- **Pipeline** — inference + evaluation job worker.

Stack: Python 3.11+ · FastAPI · Hugging Face · Streamlit.

---

## Exact commands

```bash
# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Lint / typecheck / test
make lint
pre-commit run --all-files
python -m mypy . --ignore-missing-imports
python -m pytest tests/ -v --cov=. --cov-fail-under=70

# Run
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

---

## Folder map

| Path | Purpose |
|------|---------|
| `api/` | FastAPI application (routes, services) |
| `ml/` | Model + generation logic |
| `pipeline/` | Inference + evaluation jobs |
| `dashboard/` | Streamlit app |
| `tests/` | pytest suite |
| `.github/workflows/` | CI (ruff, mypy, pytest, gitleaks, trivy) |

## Do / don't

- **Do** keep the generation interface stable so the model can swap.
- **Do not** commit `.env` files.
- **Do not** commit generated code that contains secrets or real data.

## Security rules

- No secrets in the repository; `gitleaks` CI gate gates on hits.
- Generated code must be scanned for secrets before any file leaves the
  sandbox.

## AI-assistance convention

Commits authored by AI must carry the trailer:

```text
AI-Assisted: yes | no | partial
```

See `.gitmessage` for the template. Do not rewrite historic commits
retroactively.
