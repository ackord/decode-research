import assert from "node:assert/strict";
import { test } from "node:test";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { loadResearch, parseReport, progression, summarize, type Report } from "../lib/research";
import { paginateHistory, roundHistoryHref } from "../lib/history";
import { applyPatch } from "diff";

const root = path.resolve(import.meta.dirname, "../..");

function report(round: string, status: Report["status"], gain?: number): Report {
  const decision = { decision: "abandon" as const, proposal: null, reason: "No sound experiment", plan: "" };
  return { round, status, research: decision, review: decision, reason: "Recorded decision", ...(gain === undefined ? {} : { confirmed_gain: gain }), ...(status === "accepted" ? { eval: { correct: true, cases: [] } } : {}) };
}

test("actual history preserves outcomes and computes the accepted baseline", async () => {
  const data = await loadResearch(root);
  const rounds = (await readdir(path.join(root, "experiments"))).filter(name => /^\d+$/.test(name)).sort((a, b) => Number(a) - Number(b));
  const reports: Report[] = await Promise.all(rounds.map(async round => JSON.parse(await readFile(path.join(root, "experiments", round, "report.json"), "utf8"))));
  const accepted = reports.filter(item => item.status === "accepted");
  assert.deepEqual(data.experiments.map(e => e.report.round), rounds);
  assert.deepEqual(data.experiments.map(e => e.report.status), reports.map(item => item.status));
  assert.equal(data.current?.report.round, accepted.at(-1)?.round);
  assert.equal(data.steps.length, accepted.length + 1);
  assert.equal(data.cumulative, accepted.reduce((gain, item) => gain * item.confirmed_gain!, 1));
  assert.equal(data.experiments[0].exactness, "Failed · 2/5 cases");
  assert.equal(data.experiments[0].gain, undefined);
  assert.match(data.experiments[1].mechanism, /packed 4-bit/);
  assert.match(data.experiments[2].limitation!, /Batching remains disabled/);
  assert.equal(data.experiments[4].changesBaseline, false);
  assert.equal(data.experiments.find(e => e.report.round === "0004")?.report.confirmations?.[1].gain, 1.3360200885062277);
  assert.equal(data.candidateFiles[0].path, "candidate/inference.py");
  assert.equal(data.sourceDiff ? applyPatch(data.reference, data.sourceDiff) : data.reference, data.candidate);
  assert.equal(data.testbed.contextTokens, 512);
  assert.equal(data.testbed.newTokens, 128);
});

test("accepted gains compound while rejected and abandoned gains have no effect", () => {
  const records = [report("0001", "accepted", 1.2), report("0002", "rejected", 3), report("0003", "abandoned"), report("0004", "accepted", 1.1)];
  const steps = progression(records.map(r => summarize(parseReport(r, "fixture"))));
  assert.deepEqual(steps.map(s => s.round), [null, "0001", "0004"]);
  assert.equal(steps.at(-1)?.cumulative, 1.2 * 1.1);
  assert.equal(progression([])[0].cumulative, 1);
});

test("100 rounds use bounded chronological pages without serializing reports or patches", () => {
  const experiments = Array.from({ length: 100 }, (_, index) => summarize(
    report(String(index + 1).padStart(4, "0"), index % 10 === 0 ? "accepted" : "rejected", index % 10 === 0 ? 1.04 : undefined),
    "LARGE_PATCH_MUST_NOT_APPEAR_IN_SUMMARY",
  ));
  const first = paginateHistory(experiments, 1);
  const last = paginateHistory(experiments, 10);
  assert.equal(first.pageCount, 10);
  assert.equal(first.items.length, 10);
  assert.deepEqual(first.items.map(e => e.round), experiments.slice(0, 10).map(e => e.report.round));
  assert.equal(last.items[0].round, "0091");
  assert.equal(last.items.at(-1)?.round, "0100");
  assert.equal(JSON.stringify(first).includes("LARGE_PATCH"), false);
  assert.equal("report" in first.items[0], false);
  assert.equal("proposal" in first.items[0], false);
  assert.equal(roundHistoryHref(experiments, "0042"), "/history/5/#round-0042");
  assert.equal(roundHistoryHref(experiments, "0004"), "/#round-0004");
  assert.equal(progression(experiments).length, 11);
  assert.throws(() => paginateHistory(experiments, 11), /Invalid history page/);
});

test("missing evaluation, timing, proposal, and patch remain explicit", () => {
  const result = summarize(parseReport(report("0001", "abandoned"), "fixture"));
  assert.equal(result.exactness, "Not evaluated");
  assert.equal(result.gain, undefined);
  assert.equal(result.patch, undefined);
  assert.equal(result.proposal, null);
  assert.equal(result.name, "Research round abandoned");
});

test("malformed evidence and invalid accepted gains fail with a source path", () => {
  for (const value of [null, {}, { ...report("0001", "accepted", 1.1), confirmed_gain: undefined }, report("0001", "accepted", -1), report("0001", "accepted", Infinity), { ...report("0001", "accepted", 1.1), eval: { correct: false, cases: [] } }, { ...report("0001", "rejected"), confirmations: [{ gain: "fast" }] }]) {
    assert.throws(() => parseReport(value, "experiments/fixture/report.json"), /experiments\/fixture\/report.json:/);
  }
});

test("a new report is discovered on ingestion without editing presentation data", async () => {
  // All generated test data stays inside web/.
  const temporary = await mkdtemp(path.resolve(import.meta.dirname, "../test-results-fixture-"));
  try {
    for (const directory of ["experiments/0010", "experiments/0002", "candidate/helpers", "reference", "harness"]) await mkdir(path.join(temporary, directory), { recursive: true });
    await Promise.all([
      writeFile(path.join(temporary, "experiments/0010/report.json"), JSON.stringify(report("0010", "accepted", 1.1))),
      writeFile(path.join(temporary, "experiments/0002/report.json"), JSON.stringify(report("0002", "accepted", 1.2))),
      writeFile(path.join(temporary, "candidate/inference.py"), "candidate source"),
      writeFile(path.join(temporary, "candidate/helpers/scoring.py"), "def score():\n    return 1\n"),
      writeFile(path.join(temporary, "candidate/__init__.py"), ""),
      writeFile(path.join(temporary, "candidate/weights.bin"), "not a Python source file"),
      writeFile(path.join(temporary, "reference/inference.py"), "reference source"),
      writeFile(path.join(temporary, "harness/config.py"), 'MODEL_ID = "fixture/model"\nMODEL_REVISION = "revision"\nBENCH_CONTEXT_TOKENS = 512\nBENCH_NEW_TOKENS = 128\nWARMUPS = 3\nRUNS = 20\n'),
    ]);
    const data = await loadResearch(temporary);
    assert.deepEqual(data.experiments.map(e => e.report.round), ["0002", "0010"]);
    assert.equal(data.cumulative, 1.2 * 1.1);
    assert.equal(data.current?.report.round, "0010");
    assert.deepEqual(data.candidateFiles.map(file => file.path), ["candidate/inference.py", "candidate/helpers/scoring.py"]);
    assert.equal(applyPatch(data.reference, data.sourceDiff), data.candidate);
    // A later baseline can remove a helper and restore the reference path.
    // Source views must follow the actual files rather than accepted proposals.
    await rm(path.join(temporary, "candidate/helpers/scoring.py"));
    await writeFile(path.join(temporary, "candidate/inference.py"), data.reference);
    const rebuilt = await loadResearch(temporary);
    assert.deepEqual(rebuilt.candidateFiles.map(file => file.path), ["candidate/inference.py"]);
    assert.equal(rebuilt.sourceDiff, "");
  } finally { await rm(temporary, { recursive: true, force: true }); }
});
