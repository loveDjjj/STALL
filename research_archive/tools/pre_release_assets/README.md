# Pre-release asset tools

This directory preserves commands for the superseded score-CSV release assembled before
`u0_locked_v1`. The assets use dataset-specific patch configurations and an alpha sweep over
`results/paper_scores`; they are not authorities for the locked U0 protocol.

| Compatibility command | Archived implementation |
|---|---|
| `tools/verify_alpha_stalled_release.py` | `research_archive/tools/pre_release_assets/verify_alpha_stalled_release.py` |

The compatibility command remains available because
`scripts/reproduce/rebuild_paper_assets.sh` reconstructs this historical asset family. Current
release validation must use `tools/verify_u0_locked_release.py`.
