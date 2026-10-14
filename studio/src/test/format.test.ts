import { allowance, displayModel, inr, modelName, num } from "../lib/format";

describe("format", () => {
  it("formats INR with Indian grouping", () => {
    expect(inr(500)).toBe("₹500");
    expect(inr(1000)).toBe("₹1,000");
    expect(inr(2000)).toBe("₹2,000");
    expect(num(1234567)).toBe("12,34,567");
  });
  it("never invents an allowance", () => {
    expect(allowance(null)).toBe("Not configured yet");
    expect(allowance(undefined)).toBe("Not configured yet");
    expect(allowance(10000)).toBe("10,000 credits / month");
  });
});

describe("modelName", () => {
  it("names the original, the Fine-Tuned version, open models and the empty state honestly", () => {
    expect(modelName({ state: "ready", step: 1500 })).toBe("Infinity Hyper 5.5");
    expect(modelName({ state: "ready", variant: "fine-tuned", step: 1500 })).toBe("Infinity Hyper 5.5 — Fine-Tuned");
    expect(modelName({ state: "ready", family: "open-model", product: "Qwen2.5-Coder 0.5B Instruct" })).toBe("Qwen2.5-Coder 0.5B Instruct");
    expect(modelName({ state: "unavailable" })).toBe("No model loaded");
    expect(modelName(undefined)).toBe("No model loaded");
    expect(displayModel("INFINITY HYPER 5.5 — Fine-Tuned")).toBe("Infinity Hyper 5.5 — Fine-Tuned");
    expect(displayModel("Qwen2.5-Coder 0.5B Instruct")).toBe("Qwen2.5-Coder 0.5B Instruct");
  });
});
