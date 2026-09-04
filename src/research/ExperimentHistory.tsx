import { useMemo } from "react";
import type { RunSummary } from "./types";

const metric = (value: unknown, digits = 3) => typeof value === "number" ? value.toFixed(digits) : "—";
const shortHash = (value?: string) => value ? value.slice(0, 8) : "—";

export function ExperimentHistory({ runs, selected, setSelected, onOpen, onCompare, refresh }: {
  runs: RunSummary[];
  selected: Set<string>;
  setSelected: (runs: Set<string>) => void;
  onOpen: (runId: string) => void;
  onCompare: () => void;
  refresh: () => void;
}) {
  const filters = useMemo(() => ({ suites: [...new Set(runs.map((run) => run.suite))], architectures: [...new Set(runs.map((run) => run.architecture))] }), [runs]);
  return <section className="history-view">
    <header className="view-title"><div><p className="eyebrow">Persistent archive</p><h1>Experiment history</h1><p>Every UI and compatible CLI run appears here from the same on-disk schema.</p></div><div className="actions"><button onClick={refresh}>Refresh</button><button className="primary" disabled={selected.size < 2} onClick={onCompare}>Compare {selected.size || ""}</button></div></header>
    <div className="filter-row"><label><span>Suite</span><select id="history-suite" onChange={filterTable}><option value="">All suites</option>{filters.suites.map((value) => <option key={value}>{value}</option>)}</select></label><label><span>Architecture</span><select id="history-architecture" onChange={filterTable}><option value="">All architectures</option>{filters.architectures.map((value) => <option key={value}>{value}</option>)}</select></label><label><span>Status</span><select id="history-status" onChange={filterTable}><option value="">All states</option>{["completed", "failed", "cancelled", "interrupted"].map((value) => <option key={value}>{value}</option>)}</select></label><label><span>Minimum tokens</span><input id="history-tokens" type="number" min="0" placeholder="Any" onInput={filterTable} /></label><label><span>From date</span><input id="history-date" type="date" onInput={filterTable} /></label><label className="search"><span>Search name / tag</span><input id="history-search" onInput={filterTable} placeholder="Filter runs…" /></label></div>
    <div className="table-wrap"><table id="history-table"><thead><tr><th /><th>Date / run</th><th>Suite</th><th>Architecture</th><th>Tokens</th><th>Val loss</th><th>Perplexity</th><th>tok/s</th><th>Runtime</th><th>State</th><th>Commit</th></tr></thead><tbody>{runs.map((run) => {
      const headline = run.headline ?? {};
      return <tr key={run.run_id} data-suite={run.suite} data-architecture={run.architecture} data-status={run.status} data-tokens={typeof headline.tokens_processed === "number" ? headline.tokens_processed : 0} data-date={run.started_at ? new Date(run.started_at).getTime() : 0} data-search={`${run.name} ${(run.tags ?? []).join(" ")}`.toLowerCase()} onDoubleClick={() => onOpen(run.run_id)}>
        <td><input aria-label={`Select ${run.name}`} type="checkbox" checked={selected.has(run.run_id)} onChange={(event) => { const next = new Set(selected); event.target.checked ? next.add(run.run_id) : next.delete(run.run_id); setSelected(next); }} /></td>
        <td><button className="run-link" onClick={() => onOpen(run.run_id)}><b>{run.name}</b><small>{run.started_at ? new Date(run.started_at).toLocaleString() : run.run_id}</small></button></td>
        <td>{run.suite}</td><td>{run.architecture.replaceAll("_", " ")}</td><td>{typeof headline.tokens_processed === "number" ? headline.tokens_processed.toLocaleString() : "—"}</td><td>{metric(headline.validation_loss)}</td><td>{metric(headline.perplexity, 2)}</td><td>{metric(headline.tokens_per_second, 0)}</td><td>{formatDuration(headline.runtime_seconds)}</td><td><span className={`status ${run.status}`}>{run.status}</span></td><td className="mono">{shortHash(run.git?.commit)}{run.git?.dirty ? "*" : ""}</td>
      </tr>;
    })}</tbody></table></div>
    {!runs.length && <div className="empty-state"><b>No experiments yet</b><p>Configure and launch a run to begin the archive.</p></div>}
  </section>;
}

function filterTable() {
  const suite = (document.querySelector("#history-suite") as HTMLSelectElement)?.value;
  const architecture = (document.querySelector("#history-architecture") as HTMLSelectElement)?.value;
  const status = (document.querySelector("#history-status") as HTMLSelectElement)?.value;
  const tokens = Number((document.querySelector("#history-tokens") as HTMLInputElement)?.value || 0);
  const dateValue = (document.querySelector("#history-date") as HTMLInputElement)?.value;
  const date = dateValue ? new Date(dateValue).getTime() : 0;
  const search = (document.querySelector("#history-search") as HTMLInputElement)?.value.toLowerCase();
  document.querySelectorAll<HTMLTableRowElement>("#history-table tbody tr").forEach((row) => {
    row.hidden = Boolean((suite && row.dataset.suite !== suite) || (architecture && row.dataset.architecture !== architecture) || (status && row.dataset.status !== status) || Number(row.dataset.tokens) < tokens || Number(row.dataset.date) < date || (search && !row.dataset.search?.includes(search)));
  });
}
function formatDuration(value: unknown) { if (typeof value !== "number") return "—"; return value < 60 ? `${value.toFixed(1)}s` : `${Math.floor(value / 60)}m`; }
