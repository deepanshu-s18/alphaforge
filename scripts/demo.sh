# Demo script — records a full pipeline run for the README GIF

# 1. Hypotheses generated from seeded templates
python -c "
from alphaforge.agents.hypothesis import HypothesisAgent
for h in HypothesisAgent().generate('post-earnings drift in megacap tech', seed=42):
    print(f'{h.id} [{h.family}] {h.statement[:90]}')
print()

# 2. Full orchestrated run (synthetic mode, auto-approved HITL)
" && alphaforge "post-earnings drift in megacap tech" --metrics-json /tmp/demo_metrics.json

# 3. Show the report
open "$(python -c "import json; print(json.load(open('/tmp/demo_metrics.json'))['report_path'])")"
