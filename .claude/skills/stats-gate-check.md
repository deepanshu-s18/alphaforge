---
name: stats-gate-check
description: Verify the multiple-testing correction gate is intact and honest
---

# Stats Gate Check

## When to use
Any change touching `mcp_servers/statistics`, the validation agent, or
thresholds in `config/thresholds.yaml`.

## Checklist
- [ ] `apply_bonferroni`: rejects iff `p <= alpha/m` — compare against manual calc
- [ ] `apply_bh_fdr`: standard BH step-up — must match
      `statsmodels.stats.multipletests(method='fdr_bh')` (reference test exists)
- [ ] Corrections applied ONCE per run across ALL tested hypotheses — never
      per-hypothesis, never twice
- [ ] Survival = (Bonferroni OR BH) AND sign match. A significant effect with
      the wrong sign is a REJECTION, not a discovery
- [ ] `min_events` gate marks hypotheses `UNTESTABLE` (they are excluded from
      the correction family and reported, not dropped)
- [ ] Bootstrap CIs seeded; same seed → same CI
- [ ] p-values in [0,1]; effect sizes in bps; CI brackets the point estimate

## Commands
```bash
pytest tests/test_statistics.py -q
python -c "
from alphaforge.mcp_servers.statistics import server as st
import numpy as np
ps = list(np.random.default_rng(0).uniform(0,1,20))
print('bonf', sum(st.apply_bonferroni(ps)), 'bh', sum(st.apply_bh_fdr(ps)))"
```
