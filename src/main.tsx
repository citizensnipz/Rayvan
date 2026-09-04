import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import "./styles.css";
import { ExperimentBuilder } from "./research/ExperimentBuilder";
import { ExperimentHistory } from "./research/ExperimentHistory";
import { LiveExperiment } from "./research/LiveExperiment";
import { RunComparison } from "./research/RunComparison";
import { cancelExperiment, estimateExperiment, getActiveExperiment, getExperiment, getSchema, listExperiments, startExperiment } from "./research/api";
import type { Estimate, ExperimentConfig, ResearchEvent, ResearchSchema, RunDetail, RunState, RunSummary } from "./research/types";
import mark from "./assets/rayvan-mark.svg";

type View = "build" | "live" | "history" | "report" | "compare";

function App() {
  const [view, setView] = useState<View>("build");
  const [schema, setSchema] = useState<ResearchSchema>();
  const [config, setConfig] = useState<ExperimentConfig>();
  const [estimate, setEstimate] = useState<Estimate>();
  const [estimating, setEstimating] = useState(false);
  const [activeRun, setActiveRun] = useState<string>();
  const [runState, setRunState] = useState<RunState>("idle");
  const [events, setEvents] = useState<ResearchEvent[]>([]);
  const [logs, setLogs] = useState<string[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [detail, setDetail] = useState<RunDetail>();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [comparison, setComparison] = useState<RunDetail[]>([]);
  const [error, setError] = useState<string>();
  const [network, setNetwork] = useState("Checking node…");
  const eventQueue = useRef<ResearchEvent[]>([]);
  const eventFlush = useRef<number | undefined>(undefined);

  const refreshRuns = async () => {
    try { setRuns(await listExperiments()); } catch (reason) { setError(message(reason)); }
  };

  useEffect(() => {
    void Promise.all([getSchema(), listExperiments(), getActiveExperiment()]).then(([loadedSchema, history, active]) => {
      setSchema(loadedSchema); setConfig(loadedSchema.defaults); setRuns(history);
      if (active) { setActiveRun(active.runId); setRunState("running"); setView("live"); void getExperiment(active.runId).then((loaded) => setEvents(loaded.events)); }
    }).catch((reason) => setError(message(reason)));
    let stopped = false;
    const pollNetwork = async () => {
      try {
        const status = await invoke<{ networkStatus: string }>("get_application_status");
        if (!stopped) setNetwork(status.networkStatus === "connected" ? "Node connected" : status.networkStatus === "connecting" ? "Node connecting" : "Node offline");
      } catch { if (!stopped) setNetwork("Node unavailable"); }
      if (!stopped) window.setTimeout(pollNetwork, 3000);
    };
    void pollNetwork();
    const unlistenEvent = listen<ResearchEvent>("research-event", ({ payload }) => {
      eventQueue.current.push(payload);
      if (eventFlush.current == null) {
        eventFlush.current = window.setTimeout(() => {
          const batch = eventQueue.current.splice(0);
          setEvents((current) => [...current, ...batch]);
          eventFlush.current = undefined;
        }, 120);
      }
      if (payload.type === "state_changed") setRunState(String(payload.state) as RunState);
      if (payload.type === "validation") setRunState("validation");
      if (payload.type === "training_step") setRunState("running");
      if (payload.type === "run_completed") { setRunState("completed"); void refreshRuns(); }
      if (payload.type === "run_failed") { setRunState("failed"); setError(String(payload.error ?? "Experiment failed")); void refreshRuns(); }
      if (payload.type === "run_cancelled") { setRunState("cancelled"); void refreshRuns(); }
    });
    const unlistenLog = listen<{ runId: string; line: string }>("research-log", ({ payload }) => setLogs((current) => [...current.slice(-999), payload.line]));
    return () => { stopped = true; if (eventFlush.current != null) window.clearTimeout(eventFlush.current); void unlistenEvent.then((fn) => fn()); void unlistenLog.then((fn) => fn()); };
  }, []);

  useEffect(() => {
    if (!config) return;
    setEstimating(true);
    const timer = window.setTimeout(() => {
      void estimateExperiment(config).then((value) => { setEstimate(value); setError(undefined); }).catch((reason) => { setEstimate(undefined); setError(message(reason)); }).finally(() => setEstimating(false));
    }, 450);
    return () => window.clearTimeout(timer);
  }, [config]);

  const active = Boolean(activeRun && ["initializing", "running", "validation", "diagnostics"].includes(runState));
  const navItems: Array<[View, string, string]> = [["build", "New experiment", "＋"], ["live", "Live run", "◉"], ["history", "History", "≡"]];
  const title = useMemo(() => view === "build" ? "Experiment Builder" : view === "live" ? "Live Telemetry" : view === "history" ? "Run Archive" : view === "compare" ? "Comparison" : "Run Report", [view]);

  const launch = async () => {
    if (!config) return;
    setError(undefined); setEvents([]); setLogs([]); setDetail(undefined); setRunState("initializing"); setView("live");
    try { const started = await startExperiment(config); setActiveRun(started.runId); }
    catch (reason) { setRunState("failed"); setError(message(reason)); setView("build"); }
  };
  const stop = async () => { try { await cancelExperiment(); setLogs((current) => [...current, "Cancellation requested; waiting for a safe optimizer-step boundary…"]); } catch (reason) { setError(message(reason)); } };
  const openRun = async (runId: string) => { try { const loaded = await getExperiment(runId); setDetail(loaded); setRunState((loaded.summary?.status ?? "interrupted") as RunState); setView("report"); } catch (reason) { setError(message(reason)); } };
  const compare = async () => { try { const loaded = await Promise.all([...selected].map(getExperiment)); setComparison(loaded); setView("compare"); } catch (reason) { setError(message(reason)); } };

  return <div className="app">
    <aside className="sidebar">
      <div className="brand"><img src={mark} /><div><b>Rayvan</b><span>EMC Research</span></div></div>
      <nav>{navItems.map(([id, label, icon]) => <button key={id} className={view === id ? "active" : ""} onClick={() => setView(id)}><i>{icon}</i><span>{label}</span>{id === "live" && active && <em />}</button>)}</nav>
      <div className="sidebar-foot"><span className={`connection ${network === "Node connected" ? "online" : ""}`} />{network}<small>Research Console · Schema v{schema?.schema_version ?? "—"}</small></div>
    </aside>
    <main className="workspace">
      <header className="topbar"><div><span>RESEARCH /</span><b>{title}</b></div><div className="top-actions">{active && <button className="active-run" onClick={() => setView("live")}><i /> {activeRun?.slice(-8)} running</button>}<button className="icon-button" title="Refresh history" onClick={refreshRuns}>↻</button></div></header>
      {error && <div className="error-banner"><b>Action needed</b><span>{error}</span><button onClick={() => setError(undefined)}>×</button></div>}
      <div className="content">
        {!schema || !config ? <div className="loading"><i /><p>Loading the Python experiment schema…</p></div> : <>
          {view === "build" && <ExperimentBuilder schema={schema} config={config} setConfig={setConfig} estimate={estimate} estimating={estimating} active={active} onRun={launch} />}
          {view === "live" && <LiveExperiment events={events} state={runState} runId={activeRun} logs={logs} onCancel={active ? stop : undefined} />}
          {view === "history" && <ExperimentHistory runs={runs} selected={selected} setSelected={setSelected} onOpen={openRun} onCompare={compare} refresh={refreshRuns} />}
          {view === "report" && detail && <LiveExperiment events={detail.events} state={(detail.summary?.status ?? runState) as RunState} runId={detail.runId} logs={[]} detail={detail} />}
          {view === "compare" && <RunComparison runs={comparison} />}
        </>}
      </div>
    </main>
  </div>;
}

function message(reason: unknown) { return reason instanceof Error ? reason.message : typeof reason === "string" ? reason : JSON.stringify(reason); }

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
