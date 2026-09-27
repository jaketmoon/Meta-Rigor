# RoB 消融实现说明

`no-task-contract` 对应 `minus_decomposition`：将 Full 的逐领域独立调用合并为同一 Study 内的一次联合调用，共享文档并按 Item key 返回各领域回答。实现见 [`make_jobs`](../rob_reproduction.py)。

该条件保留来源限制、各输出槽位的任务范围、回答 schema，以及程序负责的条件适用性和领域判定。这些共同约束维持任务对象与交付要求的一致性；比较关注领域级独立调用与 Study 级联合调用的差异，不涉及全部契约性约束的移除。
