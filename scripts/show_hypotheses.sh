#!/usr/bin/env bash
# Demo helper: print generated hypotheses (used by demo.tape)
python3 - <<'EOF'
from alphaforge.agents.hypothesis import HypothesisAgent

for h in HypothesisAgent().generate("post-earnings drift in megacap tech"):
    print(f"{h.id} [{h.family}] {h.statement[:72]}")
EOF
