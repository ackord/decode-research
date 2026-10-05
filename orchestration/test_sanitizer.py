import tempfile
import unittest
from pathlib import Path

from orchestration.orchestrator import sanitize_agent_output, sanitize_text


class SanitizerTests(unittest.TestCase):
    def test_relative_paths_and_urls_are_preserved(self):
        text = (
            "candidate/inference.py embedding/output zero/one "
            "https://github.com/openai/codex/blob/main/codex-rs/exec/src/cli.rs"
        )
        self.assertEqual(sanitize_text(text, Path.cwd()), text)

    def test_repo_paths_become_relative_and_external_paths_are_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inside = root / "candidate" / "inference.py"
            text = f"inside={inside}:24 outside=/Users/alice/private/model.py:7"
            self.assertEqual(
                sanitize_text(text, root),
                "inside=candidate/inference.py:24 outside=<redacted-path>",
            )

    def test_windows_absolute_paths_are_redacted(self):
        text = r"path=C:\Users\Alice\private\model.py:9"
        self.assertEqual(sanitize_text(text, Path.cwd()), "path=<redacted-path>")

    def test_nested_agent_output_is_sanitized(self):
        value = {
            "proposal": {
                "source_evidence": "candidate/inference.py and /Users/alice/private.txt"
            },
            "notes": ["embedding/output", "https://example.com/a/b"],
        }
        self.assertEqual(
            sanitize_agent_output(value, Path.cwd()),
            {
                "proposal": {
                    "source_evidence": "candidate/inference.py and <redacted-path>"
                },
                "notes": ["embedding/output", "https://example.com/a/b"],
            },
        )


if __name__ == "__main__":
    unittest.main()
