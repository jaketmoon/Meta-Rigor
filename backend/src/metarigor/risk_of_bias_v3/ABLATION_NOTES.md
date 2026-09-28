# RoB Ablation Implementation Notes

`no-task-contract` corresponds to `minus_decomposition`: Full's independent per-domain calls are combined into one joint call per Study, sharing documents and returning domain answers by Item key. See [`make_jobs`](../rob_reproduction.py) for the implementation.

This condition retains source restrictions, the task scope of each output slot, the answer schema, and program-controlled conditional applicability and domain decisions. These shared constraints keep task objects and delivery requirements consistent. The comparison concerns independent domain-level calls versus joint Study-level calls, not the removal of all contractual constraints.
