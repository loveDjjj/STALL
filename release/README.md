# Release 索引

`release/` 只保存已经冻结、可验证且有明确实验身份的轻量发布资产。运行中的
checkpoint、临时分片和未完成实验必须写入 `results/runs/<experiment_id>/` 或
`results/<experiment_id>/`，不能直接写入本目录。

| 目录 | 身份 | 当前角色 | 验证入口 |
|---|---|---|---|
| `u0/` | `u0_locked_v1` | 当前 strict-20 主发布及其明确登记的补充资产 | `tools/verify_u0_locked_release.py` |
| `u0_external_genvidbench/` | `u0_external_genvidbench_v1` | 冻结 U0 在外部 GenVidBench 上的确认性评测 | `reports/u0_locked_external_validation.md` |

当前方法权威仍是 `configs/alpha_stalled_u0_locked.yaml`，不是目录中最新修改的文件。
每个子目录的 README 必须索引其全部文件，并区分核心发布、补充证据和外部确认。
新增 release 必须具有新 protocol/experiment ID、独立目录、实验注册表记录、运行
manifest 和验证命令；禁止覆盖 `u0/` 中的既有核心资产。
