# ADR 0001: Local-First Architecture

## Status

Accepted

## Context

Rayvan aims to help small teams, solo developers, and AI coding agents understand and safely operate infrastructure connected to a software project. The product needs to inspect configuration, detect drift, show deployment health, and propose approved infrastructure changes across providers such as GitHub, Vercel, and Supabase.

A hosted multi-tenant control plane would introduce additional concerns early: account management, remote credential storage, billing, analytics, and continuous uptime requirements. Rayvan's primary users already work locally with code, environment files, and provider dashboards.

## Decision

Rayvan will initially be a **local-first desktop application** with:

- A Tauri desktop shell and React UI
- Rust native services for filesystem, credentials, and plugin hosting
- Local Node.js processes for provider plugins and the MCP server
- Shared TypeScript domain packages used by the UI, MCP server, and plugins

There will be no hosted backend, remote API, authentication service, billing, or analytics in the initial architecture.

## Consequences

### Positive

- Credentials and secrets can remain on the developer machine using OS-backed storage.
- The product can deliver value to a single developer without operational overhead.
- AI agents can access infrastructure context through a local MCP server with the same approval boundaries as humans.
- Provider integrations can evolve independently as local plugins.

### Negative

- Collaboration features across machines require future design work.
- Users must run Rayvan locally to get full functionality.
- Plugin and MCP process supervision becomes an explicit desktop responsibility.

## Rejected alternatives

### Hosted SaaS control plane

Rejected for the initial release because it increases security scope, operating cost, and time-to-value for solo developers.

### Browser-only web application

Rejected because local filesystem access, secure credential storage, and plugin process hosting are central to Rayvan's workflows.

### Single-process Node monolith

Rejected because Rust is better suited for native desktop integration and controlled process lifecycle, while TypeScript remains appropriate for plugins and MCP tooling.
