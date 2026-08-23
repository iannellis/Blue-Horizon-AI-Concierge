# Deployment and CI

## Docker

The included `Dockerfile` builds a single image that runs both the FastAPI backend and
the Streamlit UI under `supervisord`. This is the configuration used for deployment on
[HuggingFace Spaces](https://huggingface.co/spaces).

```bash
docker build -t blue-horizon .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=... \
  -e REDIS_URL=... \
  -e PGSQL_RW_DB_URL=... \
  -e PGSQL_RO_DB_URL=... \
  blue-horizon
```

The UI is served on port `7860`. The FastAPI backend runs internally on
`127.0.0.1:8000` and is not exposed outside the container.

Optional Google OAuth can be enabled by setting `GOOGLE_CLIENT_ID`,
`GOOGLE_CLIENT_SECRET`, and `COOKIE_SECRET` as HuggingFace Space secrets.
`deploy/generate_secrets.py` writes the project-level `.streamlit/secrets.toml` that
Streamlit's auth support expects from those environment variables at container start.

Both processes run as user `user` (uid 1000) rather than root. `data/` is excluded from
the image entirely, since resets are performed via Neon branches rather than by
reloading from source files.

## Continuous integration

[`.github/workflows/ci.yml`](https://github.com/iannellis/Blue-Horizon-AI-Concierge/blob/main/.github/workflows/ci.yml)
runs on every push to `main` that touches relevant paths. It is path-filtered, so an
eval-only change does not rebuild the container and a docs-only change runs nothing.

```
changes (detect changed paths)
   |
   +-- unit-tests ................ blue_horizon/, eval/, ui/, tests/ changed
   |
   +-- db-integration-tests ...... eval/ or blue_horizon/ changed
   |      resets the Development Neon branch, then runs -m db_integration
   |
   +-- eval ...................... needs: db-integration-tests
   |      resets the branch again, runs the 23-case smoke eval,
   |      then ci_check against the pinned baseline
   |
   +-- deploy .................... needs: all three above passed or skipped
          pushes an orphan commit to the HuggingFace Space
```

The `needs:` edge from `eval` to `db-integration-tests` exists for sequencing, not just
gating: both jobs reset and write to the same shared Development branch, and running
them concurrently would have each pull the rug out from under the other.

Every job tolerates its upstream being *skipped* as well as *succeeded*, so a change
that touches only `deploy/` still deploys without waiting on tests that had no reason to
run.

### A failure mode worth knowing about

The `eval` job requires `db-integration-tests` to pass first. When
`db-integration-tests` is broken, `eval` **skips** rather than fails, and the deploy
gate treats a skip as acceptable. That combination once hid a real eval crash for
several commits: the eval had not been passing, it had not been running at all.

If eval results stop appearing in CI, check whether `db-integration-tests` is red before
concluding the eval is fine.

### Other workflows

| Workflow | Trigger | Config | Baseline |
|---|---|---|---|
| `ci.yml` (smoke eval) | Push to `main` | `eval_config_23.toml` | `hotel_agent_eval_23_baseline.json` |
| `eval_206.yml` (full eval) | Manual | `eval_config_206.toml` | `hotel_agent_eval_206_baseline.json` |
| `stress.yml` | Manual | `stress_config.toml` | None |
| `docs.yml` | Push to `main` touching `docs/` or `mkdocs.yml` | None | None |

All evaluation workflows upload `eval/logs/` and `eval/outputs/` as downloadable
artifacts. The smoke eval blocks the deploy job on failure.

### Required secrets and variables

Secrets, set on the `CI` environment:

| Secret | Purpose |
|---|---|
| `LANGSMITH_API_KEY` | Dataset loading and tracing |
| `GEMINI_API_KEY` | Judge and Ragas metrics |
| `OPENAI_API_KEY` | Agents and embeddings |
| `NEON_API_KEY` | Branch reset |
| `PGSQL_RW_EVAL_DB_URL` | Development branch, read-write role |
| `PGSQL_RO_EVAL_DB_URL` | Development branch, read-only role |
| `PGSQL_ROOT_EVAL_DB_URL` | Development branch, schema owner. Optional; the one test needing it skips gracefully |
| `REDIS_URL` | Agent cache |
| `HF_TOKEN` | HuggingFace write token |

Variables: `LANGSMITH_TRACING`, `LANGCHAIN_PROJECT`, `HF_SPACE`.

## The documentation site

This site is built with [MkDocs](https://www.mkdocs.org/) and the
[Material](https://squidfunk.github.io/mkdocs-material/) theme, with
[mkdocstrings](https://mkdocstrings.github.io/) generating the
[Code Reference](../reference/index.md) from the project's Google-style docstrings.

Build and preview locally:

```bash
uv sync --group docs
uv run mkdocs serve
```

Then open `http://127.0.0.1:8000`.

`.github/workflows/docs.yml` builds the site and publishes it to GitHub Pages on every
push to `main` that touches `docs/`, `mkdocs.yml`, or the Python package. The Pages
source must be set to **GitHub Actions** in the repository settings.

mkdocstrings reads the source statically via [griffe](https://mkdocstrings.github.io/griffe/)
rather than importing it, so the docs build needs only MkDocs and its plugins installed,
not the project's runtime dependencies or any credentials.
