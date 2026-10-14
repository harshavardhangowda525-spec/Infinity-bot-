import { fireEvent, render, screen } from "@testing-library/react";
import { eventsToLines } from "../components/ExecutionStream";
import { modelPillText } from "../components/ModelStatus";
import type { ChatTurn, Status } from "../lib/types";
import ChatThread from "../ui/ChatThread";
import Composer from "../ui/Composer";
import Markdown, { parseBlocks } from "../ui/Markdown";
import ModelSelect from "../ui/ModelSelect";
import { parseRoute, routeHash } from "../ui/Sidebar";

function C(props: Partial<React.ComponentProps<typeof Composer>> = {}) {
  return (
    <Composer mode="normal" onMode={() => {}} onSubmit={() => {}} busy={false} {...props} />
  );
}

describe("Composer", () => {
  it("submits on Enter and inserts a newline on Shift+Enter", async () => {
    const onSubmit = vi.fn();
    render(<C onSubmit={onSubmit} />);
    const box = screen.getByLabelText("Message");
    fireEvent.change(box, { target: { value: "hello" } });
    fireEvent.keyDown(box, { key: "Enter", shiftKey: true });
    expect(onSubmit).not.toHaveBeenCalled();
    fireEvent.keyDown(box, { key: "Enter" });
    await Promise.resolve();
    expect(onSubmit).toHaveBeenCalledWith("hello");
  });

  it("does not submit while blocked and says why", () => {
    const onSubmit = vi.fn();
    render(<C onSubmit={onSubmit} disabledReason="No model loaded: x" />);
    const box = screen.getByLabelText("Message");
    expect(box.getAttribute("placeholder")).toBe("No model loaded: x");
    fireEvent.change(box, { target: { value: "hi" } });
    fireEvent.keyDown(box, { key: "Enter" });
    expect(onSubmit).not.toHaveBeenCalled();
    expect((screen.getByRole("button", { name: "Send" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it("shows Stop while generating and stops on Escape", () => {
    const onStop = vi.fn();
    render(<C busy onStop={onStop} />);
    fireEvent.click(screen.getByRole("button", { name: "Stop generating" }));
    fireEvent.keyDown(screen.getByLabelText("Message"), { key: "Escape" });
    expect(onStop).toHaveBeenCalledTimes(2);
  });

  it("switches between Normal and Cook; Cook shows its agent modes and project button", () => {
    const onMode = vi.fn();
    const { rerender } = render(<C onMode={onMode} />);
    expect(screen.getByRole("radio", { name: "Normal" }).getAttribute("aria-checked")).toBe("true");
    expect(screen.queryByLabelText("Cook mode")).toBeNull();
    fireEvent.click(screen.getByRole("radio", { name: "Cook" }));
    expect(onMode).toHaveBeenCalledWith("cook");
    rerender(<C mode="cook" onMode={onMode} agentMode="debug" onAgentMode={() => {}} workspaceName="proj" />);
    expect((screen.getByLabelText("Cook mode") as HTMLSelectElement).value).toBe("debug");
    expect(screen.getByRole("button", { name: "Open project" }).textContent).toContain("proj");
    expect(screen.getByLabelText("Coding task")).toBeTruthy();
  });
});

describe("ModelSelect", () => {
  const ckpt = [{ path: "/c", run: "hyper-5.5-tiny", step: 1500, parameters: 1, metrics: {} }];

  it("lists the original, Fine-Tuned adapters and open models in separate groups", () => {
    render(
      <ModelSelect
        model={{ state: "ready", variant: "fine-tuned", adapter_name: "ft", checkpoint: "/c", step: 1500 }}
        checkpoints={ckpt}
        hyperAdapters={[
          { name: "ft", label: "INFINITY HYPER 5.5 — Fine-Tuned", base_checkpoint: "/c", base_present: true, base_step: 1500, trained_steps: 1, eval_loss_before: 1, eval_loss_after: 0.5 },
          { name: "orphan", label: "x", base_checkpoint: "/gone", base_present: false, base_step: 1, trained_steps: 1, eval_loss_before: null, eval_loss_after: null },
        ]}
        openModels={[{ id: "q", label: "Qwen2.5-Coder 0.5B Instruct", base: "Qwen/x", adapter: "a", adapter_present: false, context_tokens: 4096 }]}
        onSelect={() => {}}
      />,
    );
    expect(screen.getByRole("group", { name: "INFINITY HYPER 5.5 (original)" })).toBeTruthy();
    expect(screen.getByRole("option", { name: "Infinity Hyper 5.5 · hyper-5.5-tiny · step 1500" })).toBeTruthy();
    expect(screen.getByRole("group", { name: "INFINITY HYPER 5.5 — Fine-Tuned (LoRA)" })).toBeTruthy();
    expect(screen.getByRole("group", { name: "Open models (not HYPER 5.5)" })).toBeTruthy();
    expect((screen.getByLabelText("Model") as HTMLSelectElement).value).toBe("hyper-adapter:ft");
    expect((screen.getByRole("option", { name: "Infinity Hyper 5.5 — Fine-Tuned · orphan (base checkpoint missing)" }) as HTMLOptionElement).disabled).toBe(true);
    expect((screen.getByRole("option", { name: "Qwen2.5-Coder 0.5B Instruct — Fine-Tuned (not trained yet)" }) as HTMLOptionElement).disabled).toBe(true);
  });

  it("selects an open model by id and reports switches", () => {
    const onSelect = vi.fn();
    render(<ModelSelect model={{ state: "ready", family: "open-model", id: "q", product: "Qwen" }} checkpoints={ckpt}
      openModels={[{ id: "q", label: "Qwen", base: "Qwen/x", adapter: null, adapter_present: null, context_tokens: 4096 }]} onSelect={onSelect} />);
    const sel = screen.getByLabelText("Model") as HTMLSelectElement;
    expect(sel.value).toBe("open:q");
    fireEvent.change(sel, { target: { value: "/c" } });
    expect(onSelect).toHaveBeenCalledWith("/c");
  });
});

describe("Markdown", () => {
  it("parses code fences, lists, headings and paragraphs", () => {
    const b = parseBlocks("# Title\n\nSome *text*.\n\n- a\n- b\n\n```py\nprint(1)\n```");
    expect(b.map((x) => x.kind)).toEqual(["heading", "p", "ul", "code"]);
    expect(b[3]).toEqual({ kind: "code", lang: "py", text: "print(1)" });
  });

  it("renders model text safely: no raw HTML, only http(s)/mailto links", () => {
    const { container } = render(<Markdown text={'<img src=x onerror="alert(1)"> [ok](https://example.com) [bad](javascript:alert(1)) `<b>`'} />);
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
    const links = container.querySelectorAll("a");
    expect(links).toHaveLength(1);
    expect(links[0].getAttribute("href")).toBe("https://example.com");
    expect(links[0].getAttribute("rel")).toContain("noopener");
    expect(container.textContent).toContain("[bad](javascript:alert(1))");
  });

  it("gives code blocks a copy button", () => {
    render(<Markdown text={"```js\nconst a = 1;\n```"} />);
    expect(screen.getByRole("button", { name: "Copy" })).toBeTruthy();
    expect(screen.getByText("js")).toBeTruthy();
  });
});

describe("ChatThread", () => {
  const turns: ChatTurn[] = [
    { role: "user", content: "hi" },
    { role: "assistant", content: "Hello!", model: "Infinity Hyper 5.5" },
  ];

  it("offers regenerate on the last reply and edit-and-resend on user messages", () => {
    const onRegenerate = vi.fn();
    const onEdit = vi.fn();
    render(<ChatThread turns={turns} waiting={false} onRegenerate={onRegenerate} onEdit={onEdit} />);
    fireEvent.click(screen.getByRole("button", { name: "Regenerate response" }));
    expect(onRegenerate).toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Edit message" }));
    fireEvent.change(screen.getByLabelText("Edit message"), { target: { value: "hello again" } });
    fireEvent.click(screen.getByRole("button", { name: "Save & send" }));
    expect(onEdit).toHaveBeenCalledWith(0, "hello again");
    expect(screen.getByText("Infinity Hyper 5.5")).toBeTruthy();
  });

  it("shows a typing indicator while generating", () => {
    render(<ChatThread turns={[turns[0]]} waiting onRegenerate={() => {}} onEdit={() => {}} />);
    expect(screen.getByRole("status", { name: "Infinity Bot is generating a reply" })).toBeTruthy();
  });
});

describe("routes", () => {
  it("round-trips hash routes", () => {
    expect(parseRoute("#/c/42")).toEqual({ name: "chat", id: 42 });
    expect(parseRoute("#/cook")).toEqual({ name: "cook" });
    expect(parseRoute("#/nonsense")).toEqual({ name: "chat" });
    expect(routeHash({ name: "chat", id: 7 })).toBe("#/c/7");
    expect(routeHash({ name: "settings" })).toBe("#/settings");
  });
});

describe("ExecutionStream", () => {
  it("renders only lines derived from real events", () => {
    const lines = eventsToLines([
      { seq: 1, time: 0, type: "task_start", mode: "code", task: "t" },
      { seq: 2, time: 0, type: "output", line: "1 passed\n" },
      { seq: 3, time: 0, type: "tool_result", ok: false, observation: "[run_command] error\nexit 1" },
      { seq: 4, time: 0, type: "task_end", status: "blocked", reason: "model invalid" },
      { seq: 5, time: 0, type: "unknown_kind" },
    ]);
    expect(lines.map((l) => l.text)).toEqual([
      "▶ task (code): t",
      "1 passed",
      "✗ [run_command] error  exit 1",
      "■ BLOCKED — model invalid",
    ]);
  });
});

describe("model pill", () => {
  it("names the original, the Fine-Tuned version and open models distinctly", () => {
    expect(modelPillText(null).text).toBe("Runtime offline");
    const s = { model: { state: "unavailable", reason: "x" } } as unknown as Status;
    expect(modelPillText(s).text).toContain("unavailable");
    const r = { model: { state: "ready", step: 1500 } } as unknown as Status;
    expect(modelPillText(r).text).toBe("Infinity Hyper 5.5");
    const ft = { model: { state: "ready", variant: "fine-tuned", step: 1500 } } as unknown as Status;
    expect(modelPillText(ft).text).toBe("Infinity Hyper 5.5 — Fine-Tuned");
    const open = { model: { state: "ready", family: "open-model", product: "Qwen2.5-Coder 0.5B Instruct — Fine-Tuned" } } as unknown as Status;
    expect(modelPillText(open).text).toBe("Qwen2.5-Coder 0.5B Instruct — Fine-Tuned");
    expect(modelPillText(open).text).not.toContain("Hyper");
  });
});
