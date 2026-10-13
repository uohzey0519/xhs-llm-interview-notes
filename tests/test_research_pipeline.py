"""Protect corpus integrity and reproducibility in a public, raw-free checkout."""

import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PublicPipeline(unittest.TestCase):
    def test_public_rerun_and_raw_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            for directory in ("scripts", "data/processed", "data/research", "notes"):
                shutil.copytree(ROOT / directory, repo / directory,
                                ignore=shutil.ignore_patterns("__pycache__"))
            def run(*args):
                return subprocess.run([sys.executable, *args], cwd=repo,
                                      text=True, capture_output=True)
            def digest(path):
                return hashlib.sha256((repo / path).read_bytes()).hexdigest()
            protected = ["data/processed/sources.csv", "data/processed/run_summary_2026-05-24.json"]
            baseline = {p: digest(p) for p in protected}
            outputs = protected + ["data/processed/detailed_answers.md",
                                   "data/processed/question_bank.md", "notes/llm-interview-master.md"]
            for script in ("enrich_detailed_answers.py", "generate_notebook.py"):
                args = ["scripts/" + script]
                if script == "generate_notebook.py":
                    args.append("--from-sources")
                result = run(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
            once = {p: digest(p) for p in outputs}
            self.assertEqual(baseline, {p: digest(p) for p in protected})
            for args in (["scripts/enrich_detailed_answers.py"],
                         ["scripts/generate_notebook.py", "--from-sources"]):
                result = run(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(once, {p: digest(p) for p in outputs})
            result = run("scripts/generate_notebook.py")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No raw extracts", result.stderr)
            self.assertEqual(once, {p: digest(p) for p in outputs})
            # A partial raw export must also fail before touching any output.
            raw = repo / "data/raw/xhs_note_extracts_2026-05-24.jsonl"
            raw.parent.mkdir(parents=True)
            raw.write_text('{"note_id": "incomplete"}\n')
            result = run("scripts/generate_notebook.py")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("omit existing notes", result.stderr)
            self.assertEqual(once, {p: digest(p) for p in outputs})


if __name__ == "__main__":
    unittest.main()
