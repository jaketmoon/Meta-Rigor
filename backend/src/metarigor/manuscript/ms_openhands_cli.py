from __future__ import annotations

"""MS generic 10-case 的 OpenHands CLI natural-output 外部对照实验。

复用 Data Extraction 已封存的 OpenHands 1.16.0 隔离运行时；每个 case 只看到与
Codex CLI 对照 byte-identical 的 Manuscript 输入 package。原始失败保留，用户授权的
失败补跑写入独立 attempt，后续分析通过带 ``rerun`` 标记的 selection 选择补跑结果。
"""


SURFACE_ROLES = {
    "MANUSCRIPT",
    "TABLES",
    "SUPPLEMENT",
    "REFERENCES",
    "PRISMA_FLOW",
    "FOREST_PLOT",
}


