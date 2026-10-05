import type { Experiment } from "./research";

export const HISTORY_PAGE_SIZE = 10;

// Only these fields cross the client boundary. Report prose and patches are
// confined to the individual static round documents until requested.
export type HistorySummary = {
  round: string;
  status: Experiment["report"]["status"];
  name: string;
  mechanism: string;
  exactness: string;
  lesson: string;
  gain?: number;
  changesBaseline: boolean;
};

export function paginateHistory(experiments: Experiment[], page: number) {
  const pageCount = Math.max(1, Math.ceil(experiments.length / HISTORY_PAGE_SIZE));
  if (!Number.isInteger(page) || page < 1 || page > pageCount) throw new Error("Invalid history page");
  const items: HistorySummary[] = experiments.slice((page - 1) * HISTORY_PAGE_SIZE, page * HISTORY_PAGE_SIZE).map(e => ({
    round: e.report.round, status: e.report.status, name: e.name,
    mechanism: e.mechanism, exactness: e.exactness, lesson: e.lesson,
    gain: e.gain, changesBaseline: e.changesBaseline,
  }));
  return { items, page, pageCount, total: experiments.length };
}

export const historyHref = (page: number) => page === 1 ? "/#experiments" : `/history/${page}/#experiments`;

export function roundHistoryHref(experiments: Experiment[], round: string) {
  const index = experiments.findIndex(e => e.report.round === round);
  if (index < 0) return "/#experiments";
  return historyHref(Math.floor(index / HISTORY_PAGE_SIZE) + 1).replace("#experiments", `#round-${round}`);
}
