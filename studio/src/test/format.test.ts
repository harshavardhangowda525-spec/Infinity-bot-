import { allowance, inr, modelName, num } from "../lib/format";

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
  it("names HYPER 5.5, open models and the empty state honestly", () => {
    expect(modelName({ state: "ready", step: 1500 })).toBe("INFINITY HYPER 5.5 · step 1500");
    expect(modelName({ state: "ready", family: "open-model", product: "Qwen2.5-Coder 0.5B Instruct" })).toBe("Qwen2.5-Coder 0.5B Instruct");
    expect(modelName({ state: "unavailable" })).toBe("No model loaded");
    expect(modelName(undefined)).toBe("No model loaded");
  });
});
