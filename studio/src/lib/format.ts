const inrFmt = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 });
const numFmt = new Intl.NumberFormat("en-IN");

/** ₹1,000 — Indian digit grouping. */
export const inr = (n: number) => inrFmt.format(n);
export const num = (n: number) => numFmt.format(n);
export const date = (unix: number | null | undefined) => (unix ? new Date(unix * 1000).toLocaleString() : "—");
export const day = (unix: number) => new Date(unix * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });

/** Allowance text that never invents a number. */
export const allowance = (credits: number | null | undefined) =>
  credits === null || credits === undefined ? "Not configured yet" : `${num(credits)} credits / month`;
