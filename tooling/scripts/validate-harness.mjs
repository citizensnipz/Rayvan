#!/usr/bin/env node
/**
 * Non-mutating Rayvan harness validator.
 * Confirms shared, Codex, and Cursor agent definitions remain aligned.
 */
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs";
import { join, dirname, resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "../..");

const errors = [];
const warnings = [];

function fail(message) {
  errors.push(message);
}

function warn(message) {
  warnings.push(message);
}

function read(path) {
  const full = join(ROOT, path);
  if (!existsSync(full)) {
    fail(`Missing required file: ${path}`);
    return "";
  }
  const content = readFileSync(full, "utf8");
  if (content.includes("\uFFFD")) {
    fail(`Encoding corruption detected in ${path}`);
  }
  return content;
}

function readJson(path) {
  try {
    return JSON.parse(read(path));
  } catch (error) {
    fail(`Invalid JSON in ${path}: ${error.message}`);
    return null;
  }
}

function parseFlatToml(content) {
  const normalized = content.replace(/\r\n/g, "\n");
  const result = {};
  const multilineRegex = /^([A-Za-z0-9_]+)\s*=\s*"""\n([\s\S]*?)"""/gm;
  let multilineMatch;
  while ((multilineMatch = multilineRegex.exec(normalized)) !== null) {
    result[multilineMatch[1]] = multilineMatch[2].trim();
  }
  for (const line of normalized.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#") || trimmed.startsWith("[")) {
      continue;
    }
    const match = trimmed.match(/^([A-Za-z0-9_]+)\s*=\s*"([^"]*)"\s*$/);
    if (match) {
      result[match[1]] = match[2];
    }
  }
  return result;
}

function parseCodexConfigAgents(content) {
  const agents = {};
  const blockRegex =
    /\[agents\.(?:"([^"]+)"|([A-Za-z0-9_-]+))\]\s*\n([\s\S]*?)(?=\n\[|\s*$)/g;
  let match;
  while ((match = blockRegex.exec(content)) !== null) {
    const name = match[1] ?? match[2];
    const body = match[3];
    const description = body.match(/description\s*=\s*"([^"]*)"/)?.[1];
    const configFile = body.match(/config_file\s*=\s*"([^"]*)"/)?.[1];
    agents[name] = { description, configFile };
  }
  return agents;
}

function parseYamlFrontmatter(content) {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!match) {
    return null;
  }
  const frontmatter = {};
  for (const line of match[1].split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const kv = trimmed.match(/^([A-Za-z0-9_]+):\s*(.+)$/);
    if (!kv) continue;
    let value = kv[2].trim();
    if (value === "true") value = true;
    else if (value === "false") value = false;
    frontmatter[kv[1]] = value;
  }
  return frontmatter;
}

function checkSize(path, maxBytes) {
  const full = join(ROOT, path);
  if (!existsSync(full)) return;
  const size = statSync(full).size;
  if (size > maxBytes) {
    warn(`${path} is ${size} bytes (limit ${maxBytes})`);
  }
}

function checkLocalMarkdownLinks(content, sourcePath) {
  const linkRegex = /\[[^\]]+\]\(([^)]+)\)/g;
  let match;
  while ((match = linkRegex.exec(content)) !== null) {
    const target = match[1];
    if (/^(https?:|mailto:|#)/.test(target)) continue;
    const normalized = target.split("#")[0];
    if (!normalized) continue;
    const resolved = join(ROOT, dirname(sourcePath), normalized);
    if (!existsSync(resolved)) {
      fail(`Broken local link in ${sourcePath}: ${target}`);
    }
  }
}

function main() {
  const manifest = readJson("ai/harness-manifest.json");
  if (!manifest) {
    report();
    process.exit(1);
  }

  const bootstrap = read(manifest.paths.bootstrap);
  const agentsMd = read(manifest.paths.agents_md);
  const cursorRule = read(manifest.paths.cursor_rule);
  const codexConfig = read(manifest.paths.codex_config);

  if (!bootstrap.includes(manifest.paths.agents_md)) {
    fail("BOOTSTRAP.md must reference AGENTS.md");
  }
  if (!cursorRule.includes(manifest.paths.agents_md)) {
    fail("Cursor harness rule must reference AGENTS.md");
  }
  if (!cursorRule.includes(manifest.paths.bootstrap)) {
    fail("Cursor harness rule must reference ai/BOOTSTRAP.md");
  }
  if (!agentsMd.includes(manifest.paths.bootstrap)) {
    fail("AGENTS.md must reference ai/BOOTSTRAP.md");
  }

  for (const doc of manifest.topical_docs) {
    if (!existsSync(join(ROOT, doc))) {
      fail(`Topical document missing: ${doc}`);
      continue;
    }
    if (!bootstrap.includes(doc)) {
      fail(`BOOTSTRAP.md does not route topical document: ${doc}`);
    }
  }

  const codexAgents = parseCodexConfigAgents(codexConfig);
  const roleIds = manifest.roles.map((role) => role.id);

  for (const role of manifest.roles) {
    const { id, contract, codex, cursor } = role;

    if (!existsSync(join(ROOT, contract))) {
      fail(`Missing shared contract for ${id}: ${contract}`);
    }

    const codexPath = join(".codex", codex.config_file);
    const codexContent = read(codexPath);
    const codexParsed = parseFlatToml(codexContent);

    for (const key of Object.keys(codexParsed)) {
      if (!manifest.allowed_codex_agent_keys.includes(key)) {
        fail(`Unknown Codex key "${key}" in ${codexPath}`);
      }
    }

    for (const key of manifest.allowed_codex_agent_keys) {
      if (key === "developer_instructions") continue;
      if (codexParsed[key] !== codex[key]) {
        fail(
          `${codexPath}: expected ${key}=${JSON.stringify(codex[key])}, got ${JSON.stringify(codexParsed[key])}`,
        );
      }
    }

    if (!codexParsed.developer_instructions?.includes(contract)) {
      fail(`${codexPath} must reference contract ${contract}`);
    }

    if (!codexAgents[id]) {
      fail(`Codex config.toml missing [agents."${id}"] registration`);
    } else if (codexAgents[id].configFile !== codex.config_file) {
      fail(
        `Codex config.toml config_file mismatch for ${id}: expected ${codex.config_file}`,
      );
    }

    const cursorPath = `.cursor/agents/${id}.md`;
    const cursorContent = read(cursorPath);
    const frontmatter = parseYamlFrontmatter(cursorContent);
    if (!frontmatter) {
      fail(`${cursorPath} missing YAML frontmatter`);
      continue;
    }

    for (const key of Object.keys(frontmatter)) {
      if (!manifest.allowed_cursor_frontmatter_keys.includes(key)) {
        fail(`Unknown Cursor frontmatter key "${key}" in ${cursorPath}`);
      }
    }

    if (frontmatter.name !== id) {
      fail(`${cursorPath}: expected name ${id}, got ${frontmatter.name}`);
    }

    for (const [key, expected] of Object.entries(cursor)) {
      const actual = frontmatter[key];
      const normalized =
        key === "is_background" && actual === undefined ? false : actual;
      if (normalized !== expected) {
        fail(
          `${cursorPath}: expected ${key}=${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`,
        );
      }
    }

    if (!cursorContent.includes(contract)) {
      fail(`${cursorPath} must reference contract ${contract}`);
    }

    if (cursor.readonly === true && frontmatter.readonly !== true) {
      fail(`${cursorPath} must enforce readonly: true`);
    }

    if (codex.sandbox_mode === "read-only" && cursor.readonly !== true) {
      fail(`Read-only sandbox role ${id} must have Cursor readonly: true`);
    }

    checkSize(contract, manifest.max_active_harness_file_bytes);
    checkSize(codexPath, manifest.max_active_harness_file_bytes);
    checkSize(cursorPath, manifest.max_active_harness_file_bytes);
    checkLocalMarkdownLinks(read(contract), contract);
    checkLocalMarkdownLinks(cursorContent, cursorPath);
  }

  const codexDir = join(ROOT, ".codex/agents");
  if (existsSync(codexDir)) {
    for (const file of readdirSync(codexDir)) {
      if (!file.endsWith(".toml")) continue;
      const id = file.replace(/\.toml$/, "");
      if (!roleIds.includes(id)) {
        fail(`Unexpected Codex agent file without manifest role: ${file}`);
      }
    }
  }

  const cursorDir = join(ROOT, ".cursor/agents");
  if (existsSync(cursorDir)) {
    for (const file of readdirSync(cursorDir)) {
      if (!file.endsWith(".md")) continue;
      const id = file.replace(/\.md$/, "");
      if (!roleIds.includes(id)) {
        fail(`Unexpected Cursor agent file without manifest role: ${file}`);
      }
    }
  }

  const rolesDir = join(ROOT, ".agents/roles");
  if (existsSync(rolesDir)) {
    for (const file of readdirSync(rolesDir)) {
      if (!file.endsWith(".md")) continue;
      const id = file.replace(/\.md$/, "");
      if (!roleIds.includes(id)) {
        fail(`Unexpected shared role contract without manifest role: ${file}`);
      }
    }
  }

  checkSize(manifest.paths.bootstrap, manifest.max_active_harness_file_bytes);
  checkSize(manifest.paths.agents_md, manifest.max_active_harness_file_bytes);
  checkLocalMarkdownLinks(bootstrap, manifest.paths.bootstrap);
  checkLocalMarkdownLinks(agentsMd, manifest.paths.agents_md);

  report();
  process.exit(errors.length > 0 ? 1 : 0);
}

function report() {
  if (warnings.length > 0) {
    console.warn("Harness validator warnings:");
    for (const message of warnings) {
      console.warn(`  - ${message}`);
    }
  }
  if (errors.length > 0) {
    console.error("Harness validator failed:");
    for (const message of errors) {
      console.error(`  - ${message}`);
    }
    return;
  }
  console.log("Harness validator passed.");
}

main();
