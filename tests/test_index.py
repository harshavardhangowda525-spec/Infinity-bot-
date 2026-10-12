import json
import time

import pytest

from infinity.agent import Agent, Limits, Mode, PermissionMode, PermissionPolicy
from infinity.agent.index import CodeIndex
from infinity.agent.models import ScriptedModel
from infinity.agent.permissions import Workspace
from infinity.agent.tools import ToolContext, default_registry


@pytest.fixture()
def proj(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "billing.py").write_text(
        "import json\nfrom .util import fmt\n\nRATE = 0.18\n\nclass Invoice:\n    def total(self, items):\n        return sum(items) * (1 + RATE)\n\n"
        "def make_invoice(items):\n    return Invoice().total(items)\n")
    (tmp_path / "pkg" / "util.py").write_text("def fmt(x):\n    return f'{x:.2f}'\n")
    (tmp_path / "app.py").write_text("from pkg.billing import make_invoice\nprint(make_invoice([1, 2]))\n")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "cart.ts").write_text(
        "import { api } from './api';\nexport interface Item { price: number }\nexport function cartTotal(items: Item[]) {\n  return items.reduce((a, b) => a + b.price, 0);\n}\n"
        "export const addItem = (items: Item[], item: Item) => [...items, item];\nclass Cart {}\n")
    (tmp_path / ".env").write_text("SECRET_KEY=1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("function hidden() {}\n")
    (tmp_path / "broken.py").write_text("def oops(:\n")
    return tmp_path


def test_python_and_ts_symbols(proj):
    idx = CodeIndex(proj)
    r = idx.update()
    assert r["changed"] >= 5
    syms = {(s["name"], s["kind"]) for s in idx.find_symbol("", 200)}
    for expected in [("Invoice", "class"), ("total", "method"), ("make_invoice", "function"), ("RATE", "variable"),
                     ("cartTotal", "function"), ("addItem", "function"), ("Item", "interface"), ("Cart", "class")]:
        assert expected in syms, expected
    assert not idx.find_symbol("hidden")  # node_modules ignored
    assert "broken.py" in idx.stats()["parse_errors"]
    mk = idx.find_symbol("make_invoice")[0]
    assert (mk["path"], mk["line"]) == ("pkg/billing.py", 10)
    refs = {r["path"] for r in idx.references("make_invoice")}
    assert refs == {"app.py"}
    assert {r["path"] for r in idx.importers("util")} == {"pkg/billing.py"}


def test_incremental_update_and_deletion(proj):
    idx = CodeIndex(proj)
    idx.update()
    assert idx.update()["changed"] == 0
    time.sleep(0.01)
    (proj / "pkg" / "util.py").write_text("def fmt(x):\n    return str(x)\n\ndef shout(s):\n    return s.upper()\n")
    r = idx.update()
    assert r["changed"] == 1 and idx.find_symbol("shout")
    (proj / "app.py").unlink()
    assert idx.update()["removed"] == 1 and not idx.references("make_invoice")


def test_relevant_ranks_mentioned_symbols(proj):
    idx = CodeIndex(proj)
    idx.update()
    top = idx.relevant("make_invoice applies the wrong RATE", 3)
    assert top[0]["name"] in ("make_invoice", "RATE")


def test_index_tools_through_registry(proj):
    ctx = ToolContext(workspace=Workspace(proj), policy=PermissionPolicy(PermissionMode.READ_ONLY))
    reg = default_registry()
    out = reg.execute("find_symbol", {"name": "Invoice"}, ctx)
    assert out.ok and "pkg/billing.py:6" in out.output
    assert "app.py:1" in reg.execute("find_references", {"name": "make_invoice"}, ctx).output
    assert reg.execute("code_overview", {}, ctx).data["symbols"] > 5
    assert not reg.execute("find_symbol", {"name": "SECRET_KEY"}, ctx).data["matches"]


def test_compact_prompt_for_small_context_models(proj):
    model = ScriptedModel([json.dumps({"thought": "t", "final": {"status": "completed", "summary": "s"}})], context_tokens=512)
    Agent(model, proj, PermissionPolicy(PermissionMode.AUTO), Mode.ASK).run("where is make_invoice?")
    system = model.calls[0][0].content
    assert system.startswith("INFINITY HYPER 5.5 agent.") and len(system) < 400 and "read_file" in system

    big = ScriptedModel([json.dumps({"thought": "t", "final": {"status": "completed", "summary": "s"}})], context_tokens=32768)
    Agent(big, proj, PermissionPolicy(PermissionMode.AUTO), Mode.ASK).run("fix make_invoice")
    full = big.calls[0][0].content
    assert "Possibly relevant code" in full and "pkg/billing.py:10" in full
