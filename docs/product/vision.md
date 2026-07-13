# Rayvan Vision

Rayvan is a local-first DevOps control plane designed to make infrastructure understandable and safely operable without requiring an enterprise internal developer platform.

Small teams, solo developers, and AI coding agents need a practical way to answer questions such as:

- What infrastructure is connected to this project?
- How do local, staging, and production environments differ?
- Which configuration values are missing, inconsistent, or drifted?
- What changed in the latest deployment?
- What safe action can be taken next?

Rayvan connects to providers like GitHub, Vercel, Supabase, Railway, RunPod, Sentry, and cloud storage through local plugins. It keeps credentials on the developer machine, explains configuration in plain language, and requires explicit approval before executing infrastructure mutations.

Rayvan is not a hosted SaaS control plane. It runs where development happens: on the desktop, beside the codebase, with optional MCP access for coding agents.
