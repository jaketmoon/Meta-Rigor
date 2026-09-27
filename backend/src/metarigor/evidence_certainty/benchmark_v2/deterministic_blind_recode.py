from __future__ import annotations

"""对 EC V2 OpenHands 自由 Markdown 做确定性盲态转录。

OpenHands 输出契约要求候选显式写出五个 domain judgment、overall downgrade 和
final certainty。这里仅解析这些显式字段，并把候选中的 evidence ID 与逐字片段回绑
到 blind package；不读取 gold、MR 输出或 external comparator，也不推断研究语义。
"""


