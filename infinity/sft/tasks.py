"""Synthetic, verifiable bug-fix tasks for tool-use fine-tuning and benchmarking.

Every task is a tiny Python project (module + unittest file) with exactly one
injected bug, plus the ground-truth fix. Families:

- ``typo``:    a misspelled variable. The fix is visible in the real tool
               output (Python prints ``Did you mean: 'total'?``).
- ``operator``: a wrong arithmetic operator. The intended operation is
               signalled by the function-name prefix (add_/sub_/mul_).
- ``compare``: a flipped comparison. Signalled by is_above_/is_below_.

Train and eval tasks use DISJOINT identifier pools and different value
ranges, so eval measures generalisation to unseen names within these
templates. It does not measure general coding ability.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

TEST_COMMAND = "python3 -m unittest -q"

SUFFIX_POOLS = {
    "train": ["rx", "kt", "lumen", "orbit", "delta", "vex", "nova", "pyl", "quark", "zen", "flux", "core", "mint",
              "arc", "bolt", "echo", "fern", "glyph", "halo", "ion", "jade", "kite", "loop", "moss", "nim", "opal",
              "pike", "reef", "sage", "tide", "umbra", "volt", "wisp", "yarn", "zinc", "amber", "brook", "cedar"],
    "eval": ["prism", "comet", "dune", "ember", "frost", "grove", "harbor", "iris", "juniper", "kelp", "lotus",
             "maple", "nectar", "onyx", "pebble", "quill", "raven", "spruce", "thorn", "unity"],
}
MODULE_POOLS = {
    "train": ["calc", "mathx", "ops", "numbers", "arith", "logic", "metrics", "units", "scores", "ledger", "stats", "tally"],
    "eval": ["engine", "kernel", "compute", "toolkit", "helpers", "finance"],
}
VAR_PAIRS = {
    "train": [("total", "totl"), ("count", "cuont"), ("price", "prcie"), ("value", "vlaue"), ("amount", "amuont"),
              ("width", "widht"), ("height", "heigth"), ("speed", "sped"), ("weight", "wieght"), ("score", "socre")],
    "eval": [("length", "lenght"), ("budget", "budgte"), ("factor", "facotr"), ("radius", "raduis"), ("volume", "volmue")],
}
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
    mod = rng.choice(MODULE_POOLS[split])
    suffix = rng.choice(SUFFIX_POOLS[split])
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
        var, typo = rng.choice(VAR_PAIRS[split])
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
    if split not in SUFFIX_POOLS:
        raise ValueError("split must be 'train' or 'eval'")
    rng = random.Random(f"{seed}:{split}")
    return [make_task(rng, split, i) for i in range(n)]


def expert_actions(task: BugTask) -> list[dict]:
    """Ground-truth action sequence. Every step's observation comes from real tools when replayed."""
    return [
        {"thought": "Run the tests to see the failure.", "plan": ["run tests", "read code", "fix bug", "rerun tests"],
         "tool": "run_command", "args": {"command": task.test_command}},
        {"thought": f"Read {task.module_file}.", "tool": "read_file", "args": {"path": task.module_file}},
        {"thought": "Fix the bug.", "tool": "edit_file", "args": {"path": task.module_file, "old": task.buggy_line, "new": task.fixed_line}},
        {"thought": "Run the tests again.", "tool": "run_command", "args": {"command": task.test_command}},
        {"thought": "Tests pass.", "final": {"status": "completed", "summary": f"Fixed {task.function}: {task.buggy_line} -> {task.fixed_line}."}},
    ]
