# Rayvan Principles

1. **Local-first** — Core workflows run on the developer machine. No hosted control plane is required to get value.
2. **Secrets remain protected** — Credentials stay in OS-backed secure storage. UI and MCP clients do not receive raw secrets.
3. **Inspect before changing** — Rayvan prioritizes discovery, explanation, and comparison before proposing mutations.
4. **Plan before executing** — Infrastructure changes are expressed as explicit, reviewable action plans.
5. **Humans approve mutations** — No plugin, UI surface, MCP tool, or AI agent may execute infrastructure changes without approval.
6. **Plugins are capability-limited** — Provider plugins receive only the permissions and credentials they need for their declared capabilities.
7. **Provider details do not leak into the core domain** — The core model speaks in projects, environments, integrations, and findings—not provider-specific nouns.
8. **AI uses the same controlled interfaces as humans** — Agents operate through Rayvan domain tools and approval workflows.
9. **Useful for one developer before useful for one thousand** — Rayvan should help a solo developer on day one.
10. **Clear explanations over infrastructure jargon** — Configuration and findings should be understandable without deep platform expertise.
