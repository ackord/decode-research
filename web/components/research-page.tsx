import type { ReactNode } from "react";
import { loadResearch } from "../lib/research";
import { ratio, percent } from "../lib/format";
import { historyHref, paginateHistory, roundHistoryHref } from "../lib/history";
import ExperimentEntry from "./experiment-entry";
import { REPOSITORY_URL } from "../lib/site";
import StickyHeader from "./sticky-header";
import CodeBlock from "./code-block";

function Section({ id, number, label, children, className = "" }: {
  id: string; number: string; label: string; children: ReactNode; className?: string;
}) {
  return <section id={id} className={`research-section ${className}`} aria-label={label}>
    <div className="section-label"><span>{number}</span><span>{label}</span></div>
    <div className="section-content">{children}</div>
  </section>;
}

export default async function ResearchPage({ historyPage = 1 }: { historyPage?: number }) {
  const research = await loadResearch();
  const { experiments, current, cumulative, steps, testbed } = research;
  const accepted = experiments.filter(e => e.changesBaseline).length;
  const history = paginateHistory(experiments, historyPage);
  const modelName = testbed.model.split("/").at(-1);
  const helperSources = research.candidateFiles.filter(file => file.path !== "candidate/inference.py");
  const currentReportUrl = current ? `/rounds/${current.report.round}/` : undefined;
  const stageDescriptions = [
    ["Research", "One strong experiment, grounded in code and prior evidence."],
    ["Review", "A fresh reviewer tries to falsify and strengthen the proposal."],
    ["Implement", "Build the reviewed mechanism inside candidate/."],
    ["Evaluate", "Require exact token equality with the reference."],
    ["Benchmark", "Measure decode against the current best, twice."],
    ["Decide", "Accept a clear gain; otherwise retain the baseline."],
  ];
  const baselineRows = (items: typeof steps) => <ol className="baseline-progression">{items.map((step) => <li key={step.round ?? "original"} className={step.round ? "accepted-step" : ""}><span className="progress-dot" aria-hidden="true" /><div><span className="eyebrow">{step.round ? `Round ${step.round} · accepted` : "Inception · reference"}</span><h3>{step.round ? <a href={roundHistoryHref(experiments, step.round)}>{step.name}</a> : step.name}</h3><p>{step.round ? `${ratio(step.gain)} incremental gain against the preceding baseline` : "Synchronous scalar greedy decoding"}</p></div><div className="step-result"><strong>{ratio(step.cumulative)}</strong><span>{step.cumulative === cumulative ? "Current baseline" : step.round ? "Prior baseline" : "Starting baseline"}</span></div></li>)}</ol>;
  return <div className="site-shell">
    <a className="skip-link" href="#progress">Skip to research</a>
    {/* eslint-disable-next-line @next/next/no-html-link-for-pages -- Native home navigation intentionally resets pagination and open disclosures. */}
    <StickyHeader><a className="header-context" href="/">Decode Research</a><nav aria-label="Main navigation"><a href="#progress">Progress</a><a href="#current-best">Implementation</a><a href="#objective">Objective</a><a href="#method">Method</a><a href="#experiments">History</a><a className="repository-link" href={REPOSITORY_URL} target="_blank" rel="noopener noreferrer">GitHub ↗</a></nav></StickyHeader>
    <main id="top">
      <div className="hero">
        <div className="hero-main"><h1>Exact autoregressive<br />decode optimization</h1><p className="hero-description">This project uses an autonomous research loop to improve decode throughput for a fixed language model. Generated tokens must exactly match reference greedy decoding. Correctness and performance are evaluated by the orchestrator.</p><a className="text-link" href="#experiments">Experiment history <span aria-hidden="true">↘</span></a></div>
        <aside className="hero-aside" aria-label="Research at a glance"><div className="aside-heading"><span className="tiny-square" aria-hidden="true" />The testbed</div><dl><div><dt>Target</dt><dd>{modelName}</dd></div><div><dt>Model revision</dt><dd><code title={testbed.revision}>{testbed.revision.slice(0, 12)}</code></dd></div><div><dt>Objective</dt><dd>Decode throughput</dd></div><div><dt>Correctness</dt><dd>Exact greedy tokens</dd></div><div><dt>Research record</dt><dd>{experiments.length} rounds · {accepted} accepted</dd></div></dl><p>Demonstrated on this testbed.<br />Designed to investigate mechanisms<br className="desktop-break" /> that could matter at larger scales.</p></aside>
      </div>

      <Section id="progress" number="01" label="Measured progress" className="progress-section">
        <div className="progress-heading"><h2>Cumulative decode improvement</h2><span className="eyebrow">Accepted sequence only</span></div>
        <div className="progress-result"><div><span className="big-result">{ratio(cumulative)}</span><p className="result-label">current / original decode throughput</p></div><div className="result-context"><p><strong>{percent(cumulative)} higher throughput.</strong><br />Exact output on the recorded evaluation workload.</p><p>Derived from {accepted === 1 ? "one accepted experiment" : `${accepted} accepted experiments`}. Rejected rounds contribute evidence, never gains.</p></div></div>
        {steps.length > 5 && <p className="small-copy">Original baseline and the three most recent accepted changes. The complete accepted sequence is available below.</p>}
        {baselineRows(steps.length > 5 ? [steps[0], ...steps.slice(-3)] : steps)}
        {steps.length > 5 && <details className="secondary-disclosure"><summary>All {accepted} accepted baseline changes<span aria-hidden="true">+</span></summary>{baselineRows(steps)}</details>}
        <details className="methodology"><summary>How this number is calculated<span aria-hidden="true">+</span></summary><div><p className="formula">Cumulative speedup = ∏ accepted confirmed gains</p><p>Each confirmation divides candidate throughput by the current baseline’s throughput. The confirmed gain is the lower of the two confirmation ratios. Accepted gains compound; rejected and abandoned rounds leave the baseline unchanged. This compounded record is not a separate end-to-end remeasurement.</p><p>The harness compares implementations back-to-back with alternating order, using {testbed.warmups} warmups and {testbed.runs} measured pairs. It uses a {testbed.contextTokens}-token context and requests {testbed.newTokens} output tokens. The first token comes from prefill; the remaining {testbed.newTokens - 1} steps are timed, including synchronization at completion.</p><p>A {ratio(cumulative)} throughput ratio means {percent(cumulative)} higher throughput. The equivalent decode-time reduction is {((1 - 1 / cumulative) * 100).toFixed(1)}%. No absolute tokens-per-second figure is recorded in these reports.</p></div></details>
      </Section>

      <Section id="current-best" number="02" label="The current best">
        <div className="current-heading"><h2>Current implementation</h2><span className="current-marker">{current ? `Baseline · round ${current.report.round}` : "Baseline · inception"}</span></div>
        <p className="section-lead"><code>candidate/inference.py</code> contains the current decoder used as the baseline for subsequent experiments. <code>reference/inference.py</code> contains the original greedy decoder. The source views below are read directly from the repository.</p>
        <dl className="implementation-metadata">
          <div><dt>Latest accepted round</dt><dd>{current ? <a href={currentReportUrl}>Round {current.report.round}</a> : "Original baseline"}</dd></div>
          <div><dt>Cumulative throughput</dt><dd>{ratio(cumulative)} vs. original</dd></div>
          <div><dt>Exact-token evaluation</dt><dd>{current ? <a href={`${currentReportUrl}#round-${current.report.round}-evaluation`}>{current.exactness}</a> : "No accepted experiment yet"}</dd></div>
        </dl>
        {current && <p className="implementation-evidence"><a href={`${currentReportUrl}#round-${current.report.round}-confirmations`}>Benchmark confirmations ↗</a></p>}
        <details className="secondary-disclosure code-disclosure"><summary>Current decoder<span className="code-path">candidate/inference.py</span></summary><CodeBlock code={research.candidate} language="python" /></details>
        <details className="secondary-disclosure code-disclosure"><summary>Original greedy decoder<span className="code-path">reference/inference.py</span></summary><CodeBlock code={research.reference} language="python" /></details>
        <details className="secondary-disclosure code-disclosure"><summary>Changes from the original reference<span className="code-path">reference/inference.py → candidate/inference.py</span></summary>{research.sourceDiff ? <CodeBlock code={research.sourceDiff} language="diff" /> : <p className="small-copy">The inference.py files are identical.</p>}</details>
        {helperSources.length > 0 && <details className="secondary-disclosure"><summary>Additional candidate Python files<span className="code-path">{helperSources.length} {helperSources.length === 1 ? "file" : "files"}</span></summary>{helperSources.map(file => <details className="secondary-disclosure code-disclosure" key={file.path}><summary>{file.path}</summary><CodeBlock code={file.content} language="python" /></details>)}</details>}
      </Section>

      <Section id="objective" number="03" label="The objective">
        <h2>Research objective and constraints</h2>
        <div className="intro-columns"><p>Autoregressive generation advances one token at a time. Each decode step runs the model over the newest token with a cached history, then selects the largest logit. This project searches for inference procedures that make that sequence faster.</p><p>“Exact” means the final token sequence matches reference greedy decoding. Internal approximation and auxiliary components are allowed; changed outputs are not. Prefill processes the prompt, but its time is not the target metric.</p></div>
        <div className="constraints"><div><h3>Fixed target model</h3><p>Weights, tokenizer, architecture, and training stay unchanged.</p></div><div><h3>Decode timing boundary</h3><p>No later-token generation or verification in prefill. All generated-token work counts as decode.</p></div><div><h3>Independent requests</h3><p>No reused outputs, KV caches, hidden states, or other request-specific computation.</p></div><div><h3>Hardware-independent optimization</h3><p>No hardware, kernel, compiler, or benchmark tricks. Auxiliary training cannot use benchmark prompts or results.</p></div></div>
        <p className="section-footnote">Frontier-scale relevance is a research criterion, not a demonstrated result. A gain here does not establish a gain on larger models or different workloads.</p>
      </Section>

      <Section id="method" number="04" label="The research loop">
        <h2>Research, review, and evaluation</h2><p className="section-lead">The researcher proposes one worthwhile experiment. The reviewer checks its assumptions and repairs weaknesses. The implementer builds the reviewed mechanism. The orchestrator owns correctness and timing.</p>
        <ol className="workflow">{stageDescriptions.map(([title, description], index) => <li key={title}><span className="stage-number">{String(index + 1).padStart(2, "0")}</span><h3>{title}<span aria-hidden="true">→</span></h3><p>{description}</p></li>)}</ol>
        <div className="history-feedback"><span aria-hidden="true">↳</span><p><strong>History informs subsequent rounds.</strong> Accepted experiments become the baseline. Rejected experiments remain part of the record: narrow negative lessons, not blanket blacklists. An implementation limitation can leave a mechanism unresolved.</p></div>
        <p className="section-footnote">Exactness is checked on fixed prompts and, for passing candidates, the full benchmark workload before timing. Acceptance requires at least 3% higher decode throughput in both confirmation runs.</p>
      </Section>

      <Section id="experiments" number="05" label="The experiment record" className="experiments-section">
        <div className="record-heading"><h2>Experiment history</h2><span className="eyebrow">{experiments.length} rounds / chronological order</span></div><p className="section-lead">A record of hypotheses, implementations, and measured outcomes. Expand a round to load its report. Performance ratios are incremental against the baseline used in that round.</p>
        <div className="history-range"><span>{history.total ? `Rounds ${history.items[0].round}–${history.items.at(-1)!.round}` : "No rounds recorded"}</span><span>Page {history.page} of {history.pageCount}</span></div>
        <div className="experiment-list">{history.items.map(experiment => <ExperimentEntry key={experiment.round} experiment={experiment} current={experiment.round === current?.report.round} />)}</div>
        {history.pageCount > 1 && <nav className="history-pagination" aria-label="Experiment history pages">
          {history.page > 1 ? <a href={historyHref(history.page - 1)}>← Previous</a> : <span>← Previous</span>}
          <span>{history.page} / {history.pageCount}</span>
          {history.page < history.pageCount ? <a href={historyHref(history.page + 1)}>Next →</a> : <span>Next →</span>}
        </nav>}
      </Section>
    </main>
    <footer className="site-footer"><a href={REPOSITORY_URL} target="_blank" rel="noopener noreferrer">Source on GitHub ↗</a><a href="#top">Top ↑</a></footer>
  </div>;
}
