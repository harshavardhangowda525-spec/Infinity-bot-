import { deriveCoreState, statusLabel, type CoreInputs } from "../lib/coreState";

const base: CoreInputs = {
  connected: true, busy: false, awaitingApproval: false, listening: false, speaking: false,
  lastAgentState: null, lastEnd: null, now: 1_000_000,
};

describe("deriveCoreState", () => {
  it("is idle with no activity (never fakes processing)", () => {
    expect(deriveCoreState(base)).toBe("idle");
    expect(statusLabel(base, "idle")).toBe("IDLE");
  });
  it("shows error when the runtime is disconnected", () => {
    const i = { ...base, connected: false, busy: true };
    expect(deriveCoreState(i)).toBe("error");
    expect(statusLabel(i, "error")).toBe("RUNTIME DISCONNECTED");
  });
  it("maps agent states", () => {
    expect(deriveCoreState({ ...base, busy: true })).toBe("thinking");
    expect(deriveCoreState({ ...base, busy: true, lastAgentState: "thinking" })).toBe("thinking");
    const ex = { ...base, busy: true, lastAgentState: "running_checks" };
    expect(deriveCoreState(ex)).toBe("executing");
    expect(statusLabel(ex, "executing")).toBe("RUNNING TESTS");
  });
  it("pauses visual activity while awaiting approval", () => {
    const i = { ...base, busy: true, awaitingApproval: true, lastAgentState: "applying_changes" };
    expect(deriveCoreState(i)).toBe("idle");
    expect(statusLabel(i, "idle")).toBe("AWAITING YOUR APPROVAL");
  });
  it("flashes success/error briefly after a task ends, then returns to idle", () => {
    const end = (status: string, agoMs: number) => ({ ...base, lastEnd: { status, time: (base.now - agoMs) / 1000 } });
    expect(deriveCoreState(end("completed_verified", 1000))).toBe("success");
    expect(deriveCoreState(end("answered", 1000))).toBe("success");
    expect(deriveCoreState(end("blocked", 1000))).toBe("error");
    expect(deriveCoreState(end("partial", 1000))).toBe("error");
    expect(deriveCoreState(end("cancelled", 1000))).toBe("idle");
    expect(deriveCoreState(end("completed_verified", 10_000))).toBe("idle");
  });
  it("listening and speaking come only from real mic / speech flags", () => {
    expect(deriveCoreState({ ...base, listening: true })).toBe("listening");
    expect(deriveCoreState({ ...base, speaking: true })).toBe("speaking");
  });
});
