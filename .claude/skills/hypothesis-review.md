---
name: hypothesis-review
description: Review generated hypotheses before they enter the research loop
---

# Hypothesis Review

## When to use
After the hypothesis agent generates a batch, or when editing
`src/alphaforge/agents/hypothesis_templates.yaml`.

## Checklist
- [ ] Every hypothesis is falsifiable: concrete event definition, window, scope
- [ ] `event_def` contains `window_days` in [1, 60] (schema enforces; reviewer confirms intent)
- [ ] `instrument_scope` tickers all exist in `config/universe.yaml`
- [ ] `expected_effect` (sign) is stated BEFORE testing — no post-hoc flips
- [ ] Template params sampled from the YAML grid, not invented ad hoc
- [ ] No near-duplicate hypotheses (same family + scope + params)
- [ ] If LLM backend used: output passed Pydantic validation; reject-and-regenerate otherwise

## Commands
```bash
python -c "
from alphaforge.agents.hypothesis import HypothesisAgent
for h in HypothesisAgent().generate('post-earnings drift', seed=42): print(h.id, h.family, h.event_def, h.instrument_scope)"
pytest tests/test_agents_graph.py::TestHypothesisAgent -q
```
