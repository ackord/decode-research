"""Run continuous decode research with Codex."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "orchestration" / "prompts"
RUNTIME = ROOT / ".run"
WORKTREE = ROOT.parent / f".{ROOT.name}-run"
BRANCH = "research/current"
MIN_GAIN = 1.03
LIMIT = re.compile(
    r"you(?:'|’)ve hit your usage limit|usage limit (?:has been )?reached|insufficient_quota|quota exceeded",
    re.I,
)

PATH_PART = r"""[^\s/\\\]\[(){}<>"'`]+"""
URL_TOKEN = r"""https?://[^\s\]\[(){}<>"'`]+"""
PATH_TOKEN = re.compile(
    URL_TOKEN
    + rf"""|(?<![\w./\\-])(?:/(?:{PATH_PART}/)*{PATH_PART}|"""
    + rf"""[A-Za-z]:[\\/](?:{PATH_PART}[\\/])*{PATH_PART})"""
)

MODEL_OVERRIDE = os.environ.get("CODEX_MODEL")
RESEARCH_MODEL = os.environ.get("CODEX_RESEARCH_MODEL", MODEL_OVERRIDE or "gpt-6-astra")
REVIEW_MODEL = os.environ.get("CODEX_REVIEW_MODEL", MODEL_OVERRIDE or "gpt-6-astra")
IMPLEMENT_MODEL = os.environ.get("CODEX_IMPLEMENT_MODEL", MODEL_OVERRIDE or "gpt-6.1-sol")
RESEARCH_EFFORT = os.environ.get("CODEX_RESEARCH_EFFORT", "max")
REVIEW_EFFORT = os.environ.get("CODEX_REVIEW_EFFORT", "max")
IMPLEMENT_EFFORT = os.environ.get("CODEX_IMPLEMENT_EFFORT", "medium")

EXPERIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "idea": {"type": "string"},
        "cost_removed": {"type": "string"},
        "source_evidence": {"type": "string"},
        "generalization": {"type": "string"},
        "risk": {"type": "string"},
        "history_relation": {"type": "string"},
    },
    "required": [
        "name",
        "idea",
        "cost_removed",
        "source_evidence",
        "generalization",
        "risk",
        "history_relation",
    ],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["experiment", "abandon"]},
        "proposal": {"anyOf": [EXPERIMENT_SCHEMA, {"type": "null"}]},
        "plan": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["decision", "proposal", "plan", "reason"],
    "additionalProperties": False,
}

IMPLEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["implemented", "abandon"]},
        "summary": {"type": "string"},
    },
    "required": ["outcome", "summary"],
    "additionalProperties": False,
}

_processes = set()
_process_lock = threading.Lock()


class UsageLimit(RuntimeError):
    pass


def run(args, cwd=ROOT, check=True):
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


def git(*args, cwd=ROOT, check=True):
    return run(["git", *args], cwd=cwd, check=check)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sanitize_text(text, cwd):
    roots = sorted(
        {Path(cwd).resolve(), ROOT.resolve()},
        key=lambda path: len(str(path)),
        reverse=True,
    )

    def replace(match):
        value = match.group(0)
        if value.startswith(("http://", "https://")):
            return value
        if value.startswith("/"):
            path = Path(value)
            for root in roots:
                try:
                    return path.relative_to(root).as_posix() or "."
                except ValueError:
                    pass
        return "<redacted-path>"

    return PATH_TOKEN.sub(replace, text)


def sanitize_agent_output(value, cwd):
    if isinstance(value, str):
        return sanitize_text(value, cwd)
    if isinstance(value, list):
        return [sanitize_agent_output(item, cwd) for item in value]
    if isinstance(value, dict):
        return {key: sanitize_agent_output(item, cwd) for key, item in value.items()}
    return value


def stop_agents():
    with _process_lock:
        processes = list(_processes)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def cleanup():
    registered = git("worktree", "list", "--porcelain", check=False).stdout
    if f"worktree {WORKTREE}\n" in registered:
        git("worktree", "remove", "--force", str(WORKTREE), check=False)
    elif WORKTREE.exists():
        marker = WORKTREE / ".git"
        if marker.is_file() and str(ROOT / ".git") in marker.read_text():
            shutil.rmtree(WORKTREE)
        else:
            raise RuntimeError(f"Refusing to delete unexpected path: {WORKTREE}")
    git("branch", "-D", BRANCH, check=False)
    shutil.rmtree(RUNTIME, ignore_errors=True)
    git("worktree", "prune", check=False)


def ensure_ready():
    if git("status", "--porcelain").stdout.strip():
        raise RuntimeError("Commit or stash local changes before starting")
    for tool in ("uv", "codex"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} is not installed")


def next_round_id():
    numbers = []
    for path in (ROOT / "experiments").glob("[0-9][0-9][0-9][0-9]"):
        if path.is_dir():
            numbers.append(int(path.name))
    return f"{max(numbers, default=0) + 1:04d}"


def result_text(report):
    gain = report.get("confirmed_gain")
    eval_result = report.get("eval", {})
    if isinstance(gain, (int, float)):
        return f"{gain:.3f}x"
    if isinstance(eval_result, dict) and eval_result.get("correct") is False:
        return "exact fail"
    if report.get("status") == "abandoned":
        return "not tested"
    return "not benchmarked"


def round_proposal(report):
    reviewed = report["review"]["proposal"]
    if isinstance(reviewed, dict):
        return reviewed
    proposed = report["research"]["proposal"]
    return proposed if isinstance(proposed, dict) else None


def history_text():
    def compact(value, limit=260):
        return " ".join(str(value).split())[:limit]

    lines = []
    for path in sorted((ROOT / "experiments").glob("[0-9][0-9][0-9][0-9]/report.json")):
        report = json.loads(path.read_text())
        proposal = round_proposal(report)
        name = proposal.get("name", "") if proposal else "none"
        idea = proposal.get("idea", "") if proposal else ""
        outcome = report.get("reason", "")
        review_note = report["review"]["reason"]
        lines.append(
            f"{report['round']}: {report['status']} | experiment={name} | result={result_text(report)} "
            f"| outcome={compact(outcome)} | review={compact(review_note)} | idea={compact(idea)}"
        )

    return "\n".join(lines) or "No previous rounds."


def agent(name, prompt, schema, cwd, model, effort, writable=False, network=False):
    RUNTIME.mkdir(exist_ok=True)
    output = RUNTIME / f"{name}.json"
    schema_path = RUNTIME / f"{name}.schema.json"
    save_json(schema_path, schema)

    args = [
        "codex",
        "--no-daemon",
        "--ask-for-approval", "never",
        "exec",
        "--ephemeral",
        "--sandbox", "workspace-write" if writable else "read-only",
        "--model", model,
        "-c", f'model_reasoning_effort="{effort}"',
        "-c", f"sandbox_workspace_write.network_access={'true' if network else 'false'}",
        "--output-schema", str(schema_path),
        "--output-last-message", str(output),
        "-",
    ]

    process = subprocess.Popen(
        args,
        cwd=cwd,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    stdout_chunks = []
    stderr_chunks = []

    def read_stream(stream, chunks):
        for line in iter(stream.readline, ""):
            chunks.append(line)

    with _process_lock:
        _processes.add(process)

    stdout_thread = threading.Thread(
        target=read_stream,
        args=(process.stdout, stdout_chunks),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=read_stream,
        args=(process.stderr, stderr_chunks),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        try:
            process.stdin.write(prompt)
            process.stdin.close()
        except BrokenPipeError:
            pass
        process.wait()
        stdout_thread.join()
        stderr_thread.join()
    finally:
        with _process_lock:
            _processes.discard(process)

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    combined = stdout + "\n" + stderr

    if process.returncode:
        if LIMIT.search(combined):
            raise UsageLimit("Codex usage limit reached")
        raise RuntimeError(f"Codex {name} failed: {stderr.strip()[-1000:]}")
    return sanitize_agent_output(json.loads(output.read_text()), cwd)


def research(history, cwd):
    prompt = (PROMPTS / "research.md").read_text().format(history=history)
    return agent(
        "research",
        prompt,
        DECISION_SCHEMA,
        cwd,
        RESEARCH_MODEL,
        RESEARCH_EFFORT,
    )


def review(history, research_result, cwd):
    prompt = (PROMPTS / "review.md").read_text().format(
        history=history,
        research=json.dumps(research_result, indent=2),
    )
    return agent(
        "review",
        prompt,
        DECISION_SCHEMA,
        cwd,
        REVIEW_MODEL,
        REVIEW_EFFORT,
    )


def proposal_from(result, role):
    proposal = result["proposal"]
    if result["decision"] == "experiment":
        if not isinstance(proposal, dict):
            raise RuntimeError(f"{role} chose an experiment without returning a proposal")
        return proposal
    if proposal is not None:
        raise RuntimeError(f"{role} abandoned the round but returned a proposal")
    return None


def command_json(args, cwd, allow_incorrect=False):
    result = run(args, cwd=cwd, check=False)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{result.stderr}") from error
    if result.returncode and not (allow_incorrect and data.get("correct") is False):
        raise RuntimeError(f"Command failed: {' '.join(args)}\n{result.stderr}")
    return data


def measure(cwd):
    result = command_json(["uv", "run", "--frozen", "python", "-m", "harness.bench"], cwd)
    return result["relative_decode_speed"]


def changed_files(cwd):
    tracked = git("diff", "--name-only", "HEAD", cwd=cwd).stdout.splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard", cwd=cwd).stdout.splitlines()
    return sorted(set(tracked + untracked))


def candidate_patch(cwd):
    untracked = git(
        "ls-files", "--others", "--exclude-standard", "--", "candidate", cwd=cwd
    ).stdout.splitlines()
    if untracked:
        git("add", "--intent-to-add", "--", *untracked, cwd=cwd)
    try:
        return git("diff", "HEAD", "--", "candidate", cwd=cwd).stdout
    finally:
        if untracked:
            git("reset", "--quiet", "--", *untracked, cwd=cwd)


def report_experiment_name(report):
    proposal = round_proposal(report)
    return proposal.get("name", "")[:60] if proposal else ""


def finish_round(cwd, report, patch=None, keep_candidate=False):
    round_dir = cwd / "experiments" / report["round"]
    round_dir.mkdir(parents=True, exist_ok=True)
    save_json(round_dir / "report.json", report)
    if patch:
        (round_dir / "patch.diff").write_text(patch)

    if not keep_candidate:
        git("restore", "--source=HEAD", "--staged", "--worktree", "--", "candidate", cwd=cwd)
        git("clean", "-fd", "--", "candidate", cwd=cwd)

    git("add", "--", "experiments", cwd=cwd)
    if keep_candidate:
        git("add", "--", "candidate", cwd=cwd)

    idea = report_experiment_name(report) or "round"
    git("commit", "-m", f"round {report['round']}: {report['status']} {idea}", cwd=cwd)
    git("merge", "--ff-only", BRANCH, cwd=ROOT)


def run_round():
    round_id = next_round_id()
    parent = git("rev-parse", "HEAD").stdout.strip()
    git("worktree", "add", "-b", BRANCH, str(WORKTREE), parent)

    history = history_text()
    print(f"round {round_id}: researching", flush=True)
    research_result = research(history, WORKTREE)
    proposal_from(research_result, "Research")

    print(f"round {round_id}: reviewing", flush=True)
    review_result = review(history, research_result, WORKTREE)
    proposal = proposal_from(review_result, "Review")

    report = {
        "round": round_id,
        "parent_commit": parent,
        "models": {
            "research": {"model": RESEARCH_MODEL, "effort": RESEARCH_EFFORT},
            "review": {"model": REVIEW_MODEL, "effort": REVIEW_EFFORT},
            "implementation": {"model": IMPLEMENT_MODEL, "effort": IMPLEMENT_EFFORT},
        },
        "status": "abandoned" if review_result["decision"] == "abandon" else "started",
        "research": research_result,
        "review": review_result,
    }

    if review_result["decision"] == "abandon":
        report["reason"] = review_result["reason"]
        finish_round(WORKTREE, report)
        print(f"round {round_id}: abandoned", flush=True)
        return

    print(f"round {round_id}: implementing {proposal['name']}", flush=True)
    implement_prompt = (PROMPTS / "implement.md").read_text().format(
        proposal=json.dumps(proposal, indent=2),
        plan=review_result["plan"],
    )
    implementation = agent(
        "implementation",
        implement_prompt,
        IMPLEMENT_SCHEMA,
        WORKTREE,
        IMPLEMENT_MODEL,
        IMPLEMENT_EFFORT,
        writable=True,
        network=True,
    )
    report["implementation"] = implementation

    if git("rev-parse", "HEAD", cwd=WORKTREE).stdout.strip() != parent:
        raise RuntimeError("Implementation agent changed git history")

    files = changed_files(WORKTREE)
    patch = candidate_patch(WORKTREE)

    if implementation["outcome"] == "abandon":
        report.update(status="abandoned", reason=implementation["summary"], changed_files=files)
        finish_round(WORKTREE, report, patch)
        print(f"round {round_id}: abandoned", flush=True)
        return

    forbidden = [path for path in files if not path.startswith("candidate/")]
    if forbidden or "candidate/inference.py" not in files:
        report.update(
            status="rejected",
            reason="implemented experiment did not change candidate/inference.py cleanly",
            changed_files=files,
        )
        finish_round(WORKTREE, report, patch)
        print(f"round {round_id}: rejected", flush=True)
        return

    eval_result = command_json(
        ["uv", "run", "--frozen", "python", "-m", "harness.eval"],
        WORKTREE,
        allow_incorrect=True,
    )
    report["eval"] = eval_result
    if not eval_result.get("correct"):
        report.update(status="rejected", reason="exact-token eval failed")
        finish_round(WORKTREE, report, patch)
        print(f"round {round_id}: rejected", flush=True)
        return

    confirmations = []
    for index in range(2):
        if index == 0:
            baseline = measure(ROOT)
            candidate = measure(WORKTREE)
        else:
            candidate = measure(WORKTREE)
            baseline = measure(ROOT)
        confirmations.append({
            "baseline_relative_decode_speed": baseline,
            "candidate_relative_decode_speed": candidate,
            "gain": candidate / baseline,
        })

    report["confirmations"] = confirmations
    report["confirmed_gain"] = min(item["gain"] for item in confirmations)
    accepted = report["confirmed_gain"] >= MIN_GAIN
    report["status"] = "accepted" if accepted else "rejected"
    report["reason"] = (
        f"decode gain confirmed twice at >= {MIN_GAIN:.2f}x"
        if accepted
        else f"decode gain was not >= {MIN_GAIN:.2f}x twice"
    )
    finish_round(WORKTREE, report, patch, keep_candidate=accepted)
    print(f"round {round_id}: {report['status']} ({report['confirmed_gain']:.3f}x)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    cleanup()
    ensure_ready()

    try:
        while True:
            run_round()
            cleanup()
            if args.once:
                return
    except UsageLimit as error:
        stop_agents()
        print(error, flush=True)
        raise SystemExit(2)
    except KeyboardInterrupt:
        stop_agents()
        print("Interrupted. Next run will clean unfinished work.", flush=True)
        raise SystemExit(130)
    except Exception as error:
        stop_agents()
        print(error, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
