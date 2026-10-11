# Energy budgets: a receipt, never a refusal

JouleDB measures and reports energy; it never refuses, delays, or stops work
because of it. Opt-in budgets work the same way.

## Tenant budgets (`quotas.energy_budget_uj`)

- `TenantManager::record_energy` always records the energy and returns the new
  total. Nothing is rejected.
- Once spending passes the budget, the tenant info (`GET /tenants/{id}`) shows
  `budget_exceeded: true`, `overrun_joules` (joules past the budget), and the
  configured `quotas.energy_budget_uj`.
- `check_energy_budget` is advisory only (true/false). It is never used to
  refuse work.

## Workflow budgets (`energy_budget_uj` on a definition)

- Every step runs, whatever the budget.
- The run record (`WorkflowInstance`) carries `budget_exceeded`,
  `overrun_joules`, and `energy_budget_uj`.

## Alerting

Each overrun:

- logs a `WARN` event on the tracing target `jouledb::energy_budget`, with
  tenant or workflow, `budget_uj`, `spent_uj` and the overrun; and
- increments a process-wide counter, `tenant::budget_exceeded_events()`.

Alert on either.

## Commit durability (JOULEDB_SYNC_COMMIT)

Default `on`: every commit is fsynced (F_FULLFSYNC on macOS) before it is
acknowledged, using group commit — writers that commit while an fsync is in
flight share the next one. `JOULEDB_SYNC_COMMIT=off` acknowledges once the WAL
reaches the OS page cache (survives a process crash, not power loss).
After an fsync failure, further writes are refused.
