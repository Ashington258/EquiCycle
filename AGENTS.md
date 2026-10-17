# Repository Instructions

- At the start of a task, read [`memory.md`](./memory.md) for concise project context and unresolved work. Treat time-sensitive device status in it as historical until freshly verified.
- Add conversation details to `memory.md` only when the user explicitly asks.
- For work under `control_unit/`, modify files only inside `control_unit/windows_runtime/` unless the user explicitly requests another path. The user's active runtime and tests are executed from this Windows runtime copy; do not modify `control_unit/src/` by default.
