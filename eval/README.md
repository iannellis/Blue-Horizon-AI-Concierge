# Blue Horizon evaluation harness

End-to-end evaluation suite for the Blue Horizon AI Concierge agent. Covers multi-turn
conversation quality, routing accuracy, injection resistance, database integrity, RAG
retrieval quality, and concurrency stress testing.

**The documentation for this directory now lives in the project documentation site:**

- [Evaluation results](https://iannellis.github.io/Blue-Horizon-AI-Concierge/evaluation/)
  - metric tables and what grades what
- [Harness](https://iannellis.github.io/Blue-Horizon-AI-Concierge/evaluation/harness/)
  - directory layout, prerequisites, CLI options, and how a run executes
- [Datasets and baselines](https://iannellis.github.io/Blue-Horizon-AI-Concierge/evaluation/datasets/)
  - dataset schema, the evaluator list, and baseline files
- [Stress test](https://iannellis.github.io/Blue-Horizon-AI-Concierge/evaluation/stress-test/)
  - concurrency results and the database invariant checks

The source for those pages is in [`docs/evaluation/`](../docs/evaluation/).

## Quick reference

```bash
# Smoke eval (23 cases) -- the CI gate
python -m eval.run_experiment --config eval/eval_config_23.toml

# Full eval (206 cases)
python -m eval.run_experiment --config eval/eval_config_206.toml

# Fast local routing check, no LangSmith traces
python -m eval.run_experiment --config eval/eval_config_206.toml --router-only --no-upload

# Compare a completed run against its baseline
python -m eval.ci_check eval/outputs/<experiment_name>/results.jsonl

# Concurrency stress test
python -m eval.stress --config eval/stress_config.toml
```
