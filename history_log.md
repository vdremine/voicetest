# History Log

## 2026-05-22

- Do not configure `ufw` automatically from project scripts.
- Firewall rules are managed manually outside the repo.
- `scripts/bootstrap_server.sh` must not install, enable, or modify `ufw`.
- Keep transport focus on:
  - browser joins room
  - agent joins room
  - frontend sees `agent-001`
  - frontend receives `agent_ready`
  - agent receives user audio track
  - agent can send data-message back
