import { notFound } from "next/navigation";
import { ExperimentReport } from "../../../components/report";
import { roundHistoryHref } from "../../../lib/history";
import { loadResearch } from "../../../lib/research";
import { REPOSITORY_URL } from "../../../lib/site";
import StickyHeader from "../../../components/sticky-header";

export const dynamicParams = false;

export async function generateStaticParams() {
  const { experiments } = await loadResearch();
  return experiments.map(e => ({ round: e.report.round }));
}

export default async function RoundPage({ params }: { params: Promise<{ round: string }> }) {
  const { round } = await params;
  const { experiments } = await loadResearch();
  const experiment = experiments.find(e => e.report.round === round);
  if (!experiment) notFound();
  return <div className="site-shell round-page">
    <StickyHeader><a href={roundHistoryHref(experiments, round)}>← Experiment history</a><span className="header-context">Round {round} · {experiment.report.status}</span><a href={REPOSITORY_URL} target="_blank" rel="noopener noreferrer">GitHub ↗</a></StickyHeader>
    <main><h1>{experiment.name}</h1><ExperimentReport experiment={experiment} /></main>
    <footer className="site-footer"><a href={REPOSITORY_URL} target="_blank" rel="noopener noreferrer">Source on GitHub ↗</a><a href={roundHistoryHref(experiments, round)}>Return to history</a></footer>
  </div>;
}
