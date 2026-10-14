import type { Checkpoint, HyperAdapter, ModelStatus, OpenModel } from "../lib/types";
import { Icon } from "./Icons";

/** Real model choices only: HYPER 5.5 checkpoints, HYPER 5.5 — Fine-Tuned adapters, and labelled open models. */
export default function ModelSelect({
  model, checkpoints, hyperAdapters = [], openModels = [], onSelect, disabled,
}: {
  model: ModelStatus | undefined;
  checkpoints: Checkpoint[];
  hyperAdapters?: HyperAdapter[];
  openModels?: OpenModel[];
  onSelect: (value: string) => void;
  disabled?: boolean;
}) {
  const value =
    model?.state !== "ready"
      ? ""
      : model.family === "open-model"
        ? `open:${model.id}`
        : model.variant === "fine-tuned"
          ? `hyper-adapter:${model.adapter_name}`
          : model.checkpoint ?? "";
  const none = checkpoints.length === 0 && openModels.length === 0 && hyperAdapters.length === 0;
  return (
    <div className="relative inline-flex items-center">
      <label className="sr-only" htmlFor="model-select">
        Model
      </label>
      <select
        id="model-select"
        value={value}
        onChange={(e) => e.target.value && onSelect(e.target.value)}
        disabled={disabled || none}
        title="Model"
        className="max-w-[200px] cursor-pointer appearance-none truncate rounded-lg border border-line bg-surface py-1.5 pl-2.5 pr-7 text-xs text-fg/90 transition hover:bg-hover disabled:cursor-not-allowed disabled:opacity-60 sm:max-w-[240px]"
      >
        {model?.state !== "ready" && <option value="">{none ? "No model trained" : "Select model"}</option>}
        {checkpoints.length > 0 && (
          <optgroup label="INFINITY HYPER 5.5 (original)">
            {checkpoints.map((c) => (
              <option key={c.path} value={c.path}>
                Infinity Hyper 5.5 · {c.run} · step {c.step}
              </option>
            ))}
          </optgroup>
        )}
        {hyperAdapters.length > 0 && (
          <optgroup label="INFINITY HYPER 5.5 — Fine-Tuned (LoRA)">
            {hyperAdapters.map((a) => (
              <option key={a.name} value={`hyper-adapter:${a.name}`} disabled={!a.base_present}>
                Infinity Hyper 5.5 — Fine-Tuned · {a.name}
                {a.base_present ? "" : " (base checkpoint missing)"}
              </option>
            ))}
          </optgroup>
        )}
        {openModels.length > 0 && (
          <optgroup label="Open models (not HYPER 5.5)">
            {openModels.map((o) => (
              <option key={o.id} value={`open:${o.id}`} disabled={o.adapter_present === false}>
                {o.label}
                {o.adapter ? (o.adapter_present ? " — Fine-Tuned" : " — Fine-Tuned (not trained yet)") : ""}
              </option>
            ))}
          </optgroup>
        )}
      </select>
      <span className="pointer-events-none absolute right-2 text-muted">{Icon.chevron}</span>
    </div>
  );
}
