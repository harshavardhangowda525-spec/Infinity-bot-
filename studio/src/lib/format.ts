const inrFmt = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
const numFmt = new Intl.NumberFormat("en-IN");

/** ₹1,000 — Indian digit grouping. */
export const inr = (n: number) => inrFmt.format(n);
export const num = (n: number) => numFmt.format(n);
export const date = (unix: number | null | undefined) => (unix ? new Date(unix * 1000).toLocaleString() : "—");
export const day = (unix: number) => new Date(unix * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });

/** The loaded model's real name: open models by their own label, never as INFINITY HYPER 5.5. */
export function modelName(
  m: { state: string; family?: string; product?: string; step?: number; variant?: string } | null | undefined,
): string {
  if (!m || m.state !== "ready") return "No model loaded";
  if (m.family === "open-model") return m.product ?? "Open model";
  if (m.variant === "fine-tuned") return "Infinity Hyper 5.5 — Fine-Tuned";
  return "Infinity Hyper 5.5";
}

/** Display form of a model label recorded by the server ("INFINITY HYPER 5.5 …" → "Infinity Hyper 5.5 …"). */
export function displayModel(label: string | null | undefined): string {
  return (label ?? "").replace(/INFINITY HYPER 5\.5/g, "Infinity Hyper 5.5");
}

/** Allowance text that never invents a number. */
export const allowance = (credits: number | null | undefined) =>
  credits === null || credits === undefined ? "Not configured yet" : `${num(credits)} credits / month`;
