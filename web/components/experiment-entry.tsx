"use client";

import { useRef, useState } from "react";
import { ratio } from "../lib/format";
import type { HistorySummary } from "../lib/history";

export default function ExperimentEntry({ experiment, current }: { experiment: HistorySummary; current: boolean }) {
  const [content, setContent] = useState<string>();
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(false);
  const pending = useRef(false);
  const url = `/rounds/${experiment.round}/`;

  async function load(open: boolean) {
    if (!open || content || pending.current) return;
    pending.current = true;
    setLoading(true);
    setError(false);
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error("Report unavailable");
      const document = new DOMParser().parseFromString(await response.text(), "text/html");
      const report = document.querySelector(".experiment-details");
      if (!report) throw new Error("Report content missing");
      // Same-origin static HTML rendered by our server components. Markdown
      // does not permit raw HTML, and framework scripts are outside this node.
      setContent(report.innerHTML);
    } catch {
      setError(true);
    } finally {
      pending.current = false;
      setLoading(false);
    }
  }

  return <details className="experiment" id={`round-${experiment.round}`} onToggle={event => { void load(event.currentTarget.open); }}>
    <summary>
      <span className="round-index">{experiment.round}</span>
      <span className="experiment-overview">
        <span className="experiment-heading"><strong>{experiment.name}</strong><span className={`status ${experiment.status}`}><i aria-hidden="true" />{experiment.status}</span></span>
        <span className="mechanism">{experiment.mechanism}</span>
        <span className="experiment-metadata">
          <span><span className="metadata-label">Exactness</span>{experiment.exactness}</span>
          <span><span className="metadata-label">Decode</span>{experiment.gain === undefined ? "Not benchmarked" : `${ratio(experiment.gain)} vs. baseline`}</span>
          <span>{experiment.changesBaseline ? "Baseline advanced" : "Baseline unchanged"}{current ? " · Current baseline" : ""}</span>
        </span>
        <span className="lesson"><span className="metadata-label">Evidence</span>{experiment.lesson}</span>
      </span>
      <span className="disclosure-symbol" aria-hidden="true" />
      <span className="sr-only">Experiment details</span>
    </summary>
    <div className="experiment-details">
      {loading && <p className="detail-loading" role="status">Loading report…</p>}
      {error && <p className="detail-loading" role="alert">The report could not be loaded. <button type="button" onClick={() => void load(true)}>Retry</button> or use the full report link below.</p>}
      {content && <div dangerouslySetInnerHTML={{ __html: content }} />}
      <a className="full-report-link" href={url}>Read the full report on its own page ↗</a>
    </div>
  </details>;
}
