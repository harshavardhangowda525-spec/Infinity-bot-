import { fireEvent, render, screen } from "@testing-library/react";
import { useRef } from "react";
import CommandDock from "../components/CommandDock";
import { eventsToLines } from "../components/ExecutionStream";
import { modelPillText } from "../components/ModelStatus";
import type { DockMode, Status } from "../lib/types";

function Dock(props: { onSubmit: (t: string) => void; busy?: boolean; disabledReason?: string | null; onMode?: (m: DockMode) => void; onCancel?: () => void }) {
  const amp = useRef(0);
  return (
    <CommandDock
      mode="code"
      onMode={props.onMode ?? (() => {})}
      onSubmit={props.onSubmit}
      onCancel={props.onCancel ?? (() => {})}
      busy={props.busy ?? false}
      disabledReason={props.disabledReason ?? null}
      model={{ state: "ready", checkpoint: "/c", step: 1500 }}
      checkpoints={[{ path: "/c", run: "hyper-5.5-tiny", step: 1500, parameters: 1, metrics: {} }]}
      onSelectModel={() => {}}
      onAttach={() => {}}
      workspaceName="proj"
      taskText={null}
      onListening={() => {}}
      amplitude={amp}
    />
  );
}

describe("CommandDock", () => {
  it("submits on Enter and inserts newline on Shift+Enter", async () => {
    const onSubmit = vi.fn();
    render(<Dock onSubmit={onSubmit} />);
    const box = screen.getByLabelText("Task for INFINITY BOT");
    fireEvent.change(box, { target: { value: "fix the bug" } });
    fireEvent.keyDown(box, { key: "Enter", shiftKey: true });
    expect(onSubmit).not.toHaveBeenCalled();
    fireEvent.keyDown(box, { key: "Enter" });
    expect(onSubmit).toHaveBeenCalledWith("fix the bug");
  });
  it("does not submit when blocked and shows why", () => {
    const onSubmit = vi.fn();
    render(<Dock onSubmit={onSubmit} disabledReason="Attach a project first" />);
    const box = screen.getByPlaceholderText("Attach a project first");
    fireEvent.change(box, { target: { value: "x" } });
    fireEvent.keyDown(box, { key: "Enter" });
    expect(onSubmit).not.toHaveBeenCalled();
  });
  it("switches modes and shows Stop while busy", () => {
    const onMode = vi.fn();
    const onCancel = vi.fn();
    render(<Dock onSubmit={() => {}} onMode={onMode} busy onCancel={onCancel} />);
    fireEvent.click(screen.getByRole("radio", { name: "Debug" }));
    expect(onMode).toHaveBeenCalledWith("debug");
    fireEvent.click(screen.getByText("Stop"));
    expect(onCancel).toHaveBeenCalled();
  });
  it("lists real checkpoints in the model selector", () => {
    render(<Dock onSubmit={() => {}} />);
    expect(screen.getByRole("option", { name: "HYPER 5.5 · hyper-5.5-tiny · step 1500" })).toBeTruthy();
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

describe("modelPillText", () => {
  it("distinguishes offline, unavailable and ready", () => {
    expect(modelPillText(null).text).toBe("Runtime offline");
    const s = { model: { state: "unavailable", reason: "x" } } as unknown as Status;
    expect(modelPillText(s).text).toContain("unavailable");
    const r = { model: { state: "ready", step: 1500 } } as unknown as Status;
    expect(modelPillText(r).text).toBe("INFINITY HYPER 5.5 · step 1500 (LOCAL)");
  });
});

describe("open models are never shown as INFINITY HYPER 5.5", () => {
  it("labels the pill with the open model's own name", () => {
    const s = { model: { state: "ready", family: "open-model", product: "Qwen2.5-Coder 0.5B Instruct + INFINITY adapter" } } as unknown as Status;
    expect(modelPillText(s).text).toBe("Qwen2.5-Coder 0.5B Instruct + INFINITY adapter (LOCAL)");
    expect(modelPillText(s).text).not.toContain("HYPER");
  });

  it("offers Chat mode and lists open models separately, disabling untrained adapters", () => {
    const amp = { current: 0 };
    render(
      <CommandDock
        mode="chat" onMode={() => {}} onSubmit={() => {}} onCancel={() => {}} busy={false} disabledReason={null}
        model={{ state: "ready", family: "open-model", id: "qwen", product: "Qwen" }}
        checkpoints={[]}
        openModels={[
          { id: "qwen", label: "Qwen2.5-Coder 0.5B Instruct", base: "Qwen/x", adapter: null, adapter_present: null, context_tokens: 4096 },
          { id: "qwen-a", label: "Qwen2.5-Coder 0.5B Instruct", base: "Qwen/x", adapter: "adapters/a", adapter_present: false, context_tokens: 4096 },
        ]}
        onSelectModel={() => {}} onAttach={() => {}} workspaceName={null} taskText={null} onListening={() => {}} amplitude={amp}
      />,
    );
    expect(screen.getByRole("radio", { name: "Chat" }).getAttribute("aria-checked")).toBe("true");
    expect(screen.getByPlaceholderText("Message INFINITY BOT…")).toBeTruthy();
    expect(screen.getByRole("group", { name: "Open models (not HYPER 5.5)" })).toBeTruthy();
    const untrained = screen.getByRole("option", { name: "Qwen2.5-Coder 0.5B Instruct + adapter (not trained yet)" }) as HTMLOptionElement;
    expect(untrained.disabled).toBe(true);
    expect((screen.getByLabelText("Model") as HTMLSelectElement).value).toBe("open:qwen");
  });
});
