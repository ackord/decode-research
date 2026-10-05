import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { createTwoFilesPatch } from "diff";

export type SourceFile = { path: string; content: string };

async function candidateSources(root: string): Promise<SourceFile[]> {
  async function walk(relative: string): Promise<SourceFile[]> {
    const entries = await readdir(path.join(root, relative), { withFileTypes: true });
    const groups = await Promise.all(entries.map(async entry => {
      const file = `${relative}/${entry.name}`;
      if (entry.isDirectory() && !entry.name.startsWith(".") && entry.name !== "__pycache__") return walk(file);
      if (!entry.isFile() || !entry.name.endsWith(".py")) return [];
      const content = await readFile(path.join(root, file), "utf8");
      return content.trim() || file === "candidate/inference.py" ? [{ path: file, content }] : [];
    }));
    return groups.flat();
  }
  return (await walk("candidate")).sort((a, b) => {
    if (a.path === "candidate/inference.py") return -1;
    if (b.path === "candidate/inference.py") return 1;
    return a.path.localeCompare(b.path);
  });
}

export type Proposal = {
  name: string;
  idea: string;
  cost_removed: string;
  source_evidence: string;
  generalization: string;
  risk: string;
  history_relation: string;
};

export type Decision = {
  decision: "experiment" | "abandon";
  proposal: Proposal | null;
  plan: string;
  reason: string;
};

export type Confirmation = {
  baseline_relative_decode_speed: number;
  candidate_relative_decode_speed: number;
  gain: number;
};

export type Report = {
  round: string;
  status: "accepted" | "rejected" | "abandoned";
  research: Decision;
  review: Decision;
  implementation?: { outcome: "implemented" | "abandon"; summary: string };
  eval?: { correct: boolean; cases: { prompt: string; correct: boolean }[] };
  confirmations?: Confirmation[];
  confirmed_gain?: number;
  reason: string;
};

export type Experiment = {
  report: Report;
  proposal: Proposal | null;
  name: string;
  mechanism: string;
  exactness: string;
  lesson: string;
  limitation?: string;
  gain?: number;
  patch?: string;
  changesBaseline: boolean;
};

export type BaselineStep = { round: string | null; name: string; gain: number; cumulative: number };

function object(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function positive(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

export function parseReport(value: unknown, source: string): Report {
  const invalid = (message: string): never => { throw new Error(`${source}: ${message}`); };
  if (!object(value)) return invalid("report must be an object");
  if (typeof value.round !== "string" || !/^\d+$/.test(value.round)) invalid("invalid round");
  if (!["accepted", "rejected", "abandoned"].includes(String(value.status))) invalid("invalid final status");
  if (typeof value.reason !== "string") invalid("missing decision reason");
  for (const role of ["research", "review"]) {
    const result = value[role];
    if (!object(result) || !["experiment", "abandon"].includes(String(result.decision)) ||
        typeof result.reason !== "string" || typeof result.plan !== "string") {
      invalid(`invalid ${role} result`);
    }
    const decision = result as Record<string, unknown>;
    if (decision.proposal !== null) {
      if (!object(decision.proposal)) invalid(`invalid ${role} proposal`);
      for (const field of ["name", "idea", "cost_removed", "source_evidence", "generalization", "risk", "history_relation"]) {
        if (typeof (decision.proposal as Record<string, unknown>)[field] !== "string") invalid(`missing ${role} proposal ${field}`);
      }
    } else if (decision.decision === "experiment") invalid(`${role} experiment has no proposal`);
  }
  if (value.implementation !== undefined && (!object(value.implementation) ||
      !["implemented", "abandon"].includes(String(value.implementation.outcome)) || typeof value.implementation.summary !== "string")) {
    invalid("invalid implementation result");
  }
  if (value.eval !== undefined) {
    if (!object(value.eval) || typeof value.eval.correct !== "boolean" || !Array.isArray(value.eval.cases)) invalid("invalid evaluation");
    const evaluation = value.eval as Record<string, unknown>;
    if (!(evaluation.cases as unknown[]).every(c => object(c) && typeof c.prompt === "string" && typeof c.correct === "boolean")) invalid("invalid evaluation cases");
  }
  if (value.confirmations !== undefined && (!Array.isArray(value.confirmations) || !value.confirmations.every(c =>
      object(c) && positive(c.baseline_relative_decode_speed) && positive(c.candidate_relative_decode_speed) && positive(c.gain)))) {
    invalid("invalid benchmark confirmations");
  }
  if (value.confirmed_gain !== undefined && !positive(value.confirmed_gain)) invalid("invalid confirmed gain");
  if (value.status === "accepted") {
    if (!positive(value.confirmed_gain)) invalid("accepted round requires a confirmed gain");
    if (!object(value.eval) || value.eval.correct !== true) invalid("accepted round requires passed exact evaluation");
  }
  return value as Report;
}

export function firstSentence(text: string): string {
  return text.match(/^.*?[.!?](?=\s|$)/s)?.[0] ?? text;
}

function mechanismSentence(text: string): string {
  const first = firstSentence(text);
  // A short setup sentence can describe preserved work rather than the change.
  // Prefer the following operation sentence in that case, without rewriting it.
  if (/^Keep\b/.test(first)) {
    const next = firstSentence(text.slice(first.length).trim());
    if (next) return next;
  }
  return first;
}

export function summarize(report: Report, patch?: string): Experiment {
  const proposal = report.review.proposal ?? report.research.proposal;
  const cases = report.eval?.cases ?? [];
  const passed = cases.filter(c => c.correct).length;
  const sentences = report.implementation?.summary.match(/[^.!?]+[.!?](?:\s|$)|[^.!?]+$/g) ?? [];
  const limitation = sentences.find(sentence => /\bdisabled\b/i.test(sentence))?.trim();
  let lesson = report.reason;
  if (limitation) lesson = limitation;
  else if (report.eval?.correct === false) lesson = `Diverged on ${cases.length - passed} of ${cases.length} evaluation cases; no timing result recorded.`;
  else if (report.status === "accepted") lesson = "Passed exact evaluation and exceeded the 3% throughput threshold in both confirmations.";
  else if (report.confirmations?.length) lesson = "The tested implementation did not meet the 3% gain threshold in both confirmations.";
  return {
    report,
    proposal,
    name: proposal?.name ?? "Research round abandoned",
    mechanism: mechanismSentence(proposal?.idea ?? report.review.reason),
    exactness: report.eval ? `${report.eval.correct ? "Passed" : "Failed"} · ${passed}/${cases.length} cases` : "Not evaluated",
    lesson,
    limitation,
    gain: report.confirmed_gain,
    patch,
    changesBaseline: report.status === "accepted",
  };
}

export function progression(experiments: Experiment[]): BaselineStep[] {
  let cumulative = 1;
  const steps: BaselineStep[] = [{ round: null, name: "Original scalar decoder", gain: 1, cumulative }];
  for (const experiment of experiments) {
    if (!experiment.changesBaseline) continue;
    if (!positive(experiment.gain)) throw new Error(`Round ${experiment.report.round}: accepted round requires a confirmed gain`);
    cumulative *= experiment.gain;
    steps.push({ round: experiment.report.round, name: experiment.name, gain: experiment.gain, cumulative });
  }
  return steps;
}

async function optionalText(file: string): Promise<string | undefined> {
  try { return await readFile(file, "utf8"); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw error;
  }
}

export async function loadResearch(root = path.resolve(process.cwd(), "..")) {
  const directory = path.join(root, "experiments");
  const entries = (await readdir(directory, { withFileTypes: true }))
    .filter(entry => entry.isDirectory() && /^\d+$/.test(entry.name))
    .sort((a, b) => Number(a.name) - Number(b.name));
  const experiments = await Promise.all(entries.map(async entry => {
    const file = path.join(directory, entry.name, "report.json");
    const report = parseReport(JSON.parse(await readFile(file, "utf8")), `experiments/${entry.name}/report.json`);
    if (report.round !== entry.name) throw new Error(`${file}: round does not match directory`);
    return summarize(report, await optionalText(path.join(directory, entry.name, "patch.diff")));
  }));
  const [candidateFiles, reference, config] = await Promise.all([
    candidateSources(root),
    readFile(path.join(root, "reference/inference.py"), "utf8"),
    readFile(path.join(root, "harness/config.py"), "utf8"),
  ]);
  const mainCandidate = candidateFiles.find(file => file.path === "candidate/inference.py");
  if (!mainCandidate) throw new Error("Missing candidate/inference.py");
  const candidate = mainCandidate.content;
  const sourceDiff = candidate === reference ? "" : createTwoFilesPatch(
    "reference/inference.py", "candidate/inference.py", reference, candidate,
  );
  const stringConstant = (name: string) => {
    const match = config.match(new RegExp(`^${name} = "([^"]+)"`, "m"));
    if (!match) throw new Error(`harness/config.py: missing ${name}`);
    return match[1];
  };
  const numberConstant = (name: string) => {
    const match = config.match(new RegExp(`^${name} = (\\d+)$`, "m"));
    if (!match) throw new Error(`harness/config.py: missing ${name}`);
    return Number(match[1]);
  };
  const steps = progression(experiments);
  return {
    experiments, steps, candidate, candidateFiles, reference, sourceDiff,
    current: experiments.filter(e => e.changesBaseline).at(-1),
    cumulative: steps.at(-1)!.cumulative,
    testbed: {
      model: stringConstant("MODEL_ID"), revision: stringConstant("MODEL_REVISION"),
      contextTokens: numberConstant("BENCH_CONTEXT_TOKENS"), newTokens: numberConstant("BENCH_NEW_TOKENS"),
      warmups: numberConstant("WARMUPS"), runs: numberConstant("RUNS"),
    },
  };
}

