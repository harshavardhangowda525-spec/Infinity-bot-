"""Synthetic, verifiable bug-fix tasks for tool-use fine-tuning and benchmarking.

Every task is a tiny Python project (module + unittest file) with exactly one
injected bug, plus the ground-truth fix. Families:

- ``typo``:    a misspelled variable. The fix is visible in the real tool
               output (the parameter name in ``read_file``; Python also prints
               ``Did you mean: '...'?``).
- ``operator``: a wrong arithmetic operator. The intended operation is
               signalled by the function-name prefix (add_/sub_/mul_).
- ``compare``: a flipped comparison. Signalled by is_above_/is_below_.

Train and eval tasks use identifiers generated from DISJOINT syllable sets and different value
ranges, so eval measures generalisation to unseen names within these
templates. It does not measure general coding ability.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

TEST_COMMAND = "python3 -m unittest -q"

# Identifiers are generated from DISJOINT syllable sets, so train and eval never share
# a name and the model must copy names from observations instead of memorising them.
SYLLABLES = {
    "train": ["ka", "ro", "mi", "tu", "ne", "sa", "lo", "vi", "pe", "da", "fu", "zo", "ri", "ba", "go", "hu", "te", "ni"],
    "eval": ["el", "an", "or", "is", "ut", "om", "ex", "ay", "ib", "uk"],
}
SUFFIX_POOLS = SYLLABLES  # kept for backwards compatibility in tests


def _word(rng: random.Random, split: str, lo: int = 2, hi: int = 3) -> str:
    return "".join(rng.choice(SYLLABLES[split]) for _ in range(rng.randint(lo, hi)))


def _typo(rng: random.Random, word: str) -> str:
    for _ in range(20):
        i = rng.randint(0, len(word) - 2)
        t = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        if t != word:
            return t
    return word + word[-1]


OPS = {"add": ("+", "-"), "sub": ("-", "+"), "mul": ("*", "+")}
DESCRIPTIONS = [
    "The test for {fn} fails. Fix it.",
    "Fix the failing test in test_{mod}.py.",
    "{fn} is broken; make the tests pass.",
    "Tests are failing. Find and fix the bug in {mod}.py.",
]


@dataclass
class BugTask:
    id: str
    family: str
    split: str
    description: str
    files: dict[str, str]
    module_file: str
    function: str
    buggy_line: str  # stripped source line containing the bug
    fixed_line: str
    test_command: str = TEST_COMMAND
    meta: dict = field(default_factory=dict)


def _test_file(mod: str, fn: str, cases: list[tuple[tuple, object]]) -> str:
    lines = ["import unittest", f"from {mod} import {fn}", "", "", f"class Test{fn.title().replace('_', '')}(unittest.TestCase):",
             f"    def test_{fn}(self):"]
    for args, expected in cases:
        lines.append(f"        self.assertEqual({fn}({', '.join(map(repr, args))}), {expected!r})")
    lines += ["", "", 'if __name__ == "__main__":', "    unittest.main()", ""]
    return "\n".join(lines)


def make_task(rng: random.Random, split: str, idx: int, family: str | None = None) -> BugTask:
    family = family or rng.choice(["typo", "operator", "compare"])
    mod = _word(rng, split)
    suffix = _word(rng, split)
    lo, hi = (1, 9) if split == "train" else (11, 29)
    if family == "operator":
        kind = rng.choice(list(OPS))
        good, bad = OPS[kind]
        fn = f"{kind}_{suffix}"
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        fixed, buggy = f"return a {good} b", f"return a {bad} b"
        exp = eval(f"{a} {good} {b}")
        src = f"def {fn}(a, b):\n    {buggy}\n"
        cases = [((a, b), exp)]
    elif family == "compare":
        kind = rng.choice(["above", "below"])
        fn = f"is_{kind}_{suffix}"
        good, bad = (">", "<") if kind == "above" else ("<", ">")
        x, limit = rng.randint(lo, hi), rng.randint(lo, hi)
        while x == limit:
            limit = rng.randint(lo, hi)
        fixed, buggy = f"return x {good} limit", f"return x {bad} limit"
        src = f"def {fn}(x, limit):\n    {buggy}\n"
        cases = [((x, limit), eval(f"{x} {good} {limit}"))]
    else:
        var = _word(rng, split, 2, 3)
        typo = _typo(rng, var)
        fn = f"scale_{suffix}"
        k = rng.randint(lo, hi)
        n = rng.randint(lo, hi)
        fixed, buggy = f"return {var} * {k}", f"return {typo} * {k}"
        src = f"def {fn}({var}):\n    {buggy}\n"
        cases = [((n,), n * k)]
    files = {f"{mod}.py": src, f"test_{mod}.py": _test_file(mod, fn, cases)}
    desc = rng.choice(DESCRIPTIONS).format(fn=fn, mod=mod)
    return BugTask(f"{split}-{idx:05d}", family, split, desc, files, f"{mod}.py", fn, buggy, fixed)


def make_tasks(split: str, n: int, seed: int) -> list[BugTask]:
    if split not in SYLLABLES:
        raise ValueError("split must be 'train' or 'eval'")
    rng = random.Random(f"{seed}:{split}")
    return [make_task(rng, split, i) for i in range(n)]


def expert_actions(task: BugTask) -> list[dict]:
    """Ground-truth action sequence. Every step's observation comes from real tools when replayed."""
    return [
        {"thought": "Look at the project files.", "plan": ["list files", "run tests", "read code", "fix bug", "rerun tests"],
         "tool": "list_dir", "args": {}},
        {"thought": "Run the tests to see the failure.", "tool": "run_command", "args": {"command": task.test_command}},
        {"thought": f"Read {task.module_file}.", "tool": "read_file", "args": {"path": task.module_file}},
        {"thought": "Fix the bug.", "tool": "edit_file", "args": {"path": task.module_file, "old": task.buggy_line, "new": task.fixed_line}},
        {"thought": "Run the tests again.", "tool": "run_command", "args": {"command": task.test_command}},
        {"thought": "Tests pass.", "final": {"status": "completed", "summary": f"Fixed {task.function}: {task.buggy_line} -> {task.fixed_line}."}},
    ]
