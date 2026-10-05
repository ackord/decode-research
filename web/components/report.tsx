import type { ReactNode } from "react";
import Markdown from "react-markdown";
import { ratio } from "../lib/format";
import type { Experiment, Proposal } from "../lib/research";

export function Prose({ text }: { text: string }) {
  return <div className="report-prose"><Markdown>{text}</Markdown></div>;
}

export function Note({ title, id, children }: { title: string; id?: string; children: ReactNode }) {
  return <div className="report-note" id={id}><h4>{title}</h4>{children}</div>;
}

function ProposalDetails({ proposal }: { proposal: Proposal }) {
  return <>
    <Note title="Mechanism"><Prose text={proposal.idea} /></Note>
    <Note title="Work intended to be removed"><Prose text={proposal.cost_removed} /></Note>
    <Note title="Evidence & source reasoning"><Prose text={proposal.source_evidence} /></Note>
    <div className="report-pair">
      <Note title="Plausible generalization"><Prose text={proposal.generalization} /></Note>
      <Note title="Main risk"><Prose text={proposal.risk} /></Note>
    </div>
    <Note title="Relation to prior evidence"><Prose text={proposal.history_relation} /></Note>
  </>;
}

export function ExperimentReport({ experiment }: { experiment: Experiment }) {
  const { report, proposal } = experiment;
  return (
    <div className="experiment-details">
      <div className="detail-intro"><span className="eyebrow">Round {report.round} / research record</span><p>Proposals explain the intended mechanism. Evaluation and confirmations establish the recorded outcome.</p></div>
      {experiment.limitation && <div className="limitation"><span className="eyebrow">Implementation limitation · mechanism unresolved</span><p>{experiment.limitation} These timings do not establish the performance of the active proposed mechanism.</p></div>}
      {proposal && <ProposalDetails proposal={proposal} />}
      <Note title="Independent review"><Prose text={report.review.reason} /></Note>
      {report.review.plan && <details className="secondary-disclosure"><summary>Reviewed implementation plan<span aria-hidden="true">↗</span></summary><Prose text={report.review.plan} /></details>}
      {report.implementation && <Note title="Implementation outcome"><p className="eyebrow">{report.implementation.outcome}</p><Prose text={report.implementation.summary} /></Note>}
      <Note title="Exact-token evaluation" id={`round-${report.round}-evaluation`}>
        {report.eval ? <><p className="evaluation-verdict">{experiment.exactness}. {report.eval.correct ? "Output matched reference greedy decoding on the recorded workload." : "Correctness gate failed; the candidate was not benchmarked."}</p>
          <ul className="evaluation-cases">{report.eval.cases.map((item, index) => <li key={index}><span>{item.prompt === "<benchmark>" ? "Full benchmark workload" : item.prompt}</span><span className={item.correct ? "case-pass" : "case-fail"}>{item.correct ? "Match" : "Mismatch"}</span></li>)}</ul>
        </> : <p>No exact-token evaluation was recorded.</p>}
      </Note>
      <Note title="Benchmark confirmations" id={`round-${report.round}-confirmations`}>
        {report.confirmations?.length ? <>
          <div className="table-scroll"><table><caption>Each gain is candidate throughput divided by the baseline throughput for that confirmation.</caption><thead><tr><th>Run</th><th>Baseline / reference</th><th>Candidate / reference</th><th>Incremental gain</th></tr></thead><tbody>{report.confirmations.map((confirmation, index) => <tr key={index}><th>{String(index + 1).padStart(2, "0")}</th><td>{ratio(confirmation.baseline_relative_decode_speed)}</td><td>{ratio(confirmation.candidate_relative_decode_speed)}</td><td>{ratio(confirmation.gain)}</td></tr>)}</tbody></table></div>
          <p className="small-copy">Confirmed gain: {experiment.gain === undefined ? "not recorded" : ratio(experiment.gain)}. The orchestrator uses the lower of the two gains and requires at least 1.030× in both runs.</p>
        </> : <p>No throughput result was recorded. Absence of timing is not evidence of a slowdown.</p>}
      </Note>
      <div className="final-decision"><span className="eyebrow">Final decision / {report.status}</span><Prose text={report.reason} /><p className="small-copy">{experiment.changesBaseline ? "Accepted into the baseline used for subsequent experiments." : "The accepted baseline was retained. This record remains evidence for future research."}</p></div>
      <details className="secondary-disclosure"><summary>Original researcher proposal<span aria-hidden="true">↗</span></summary>{report.research.proposal && <ProposalDetails proposal={report.research.proposal} />}<Note title="Researcher reasoning"><Prose text={report.research.reason} /></Note>{report.research.plan && <Note title="Original plan"><Prose text={report.research.plan} /></Note>}</details>
      {experiment.patch && <details className="secondary-disclosure code-disclosure"><summary>Tested patch<span className="code-path">experiments/{report.round}/patch.diff</span></summary><pre tabIndex={0}><code>{experiment.patch}</code></pre></details>}
      <p className="source-note">Source: experiments/{report.round}/report.json. Report prose is preserved; path redactions originate in the research record.</p>
    </div>
  );
}
