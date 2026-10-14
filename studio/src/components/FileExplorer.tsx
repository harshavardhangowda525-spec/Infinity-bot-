import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import type { TreeNode } from "../lib/types";
import { Icon } from "./Panel";

function Node({ node, depth, onOpen, selected, changed }: {
  node: TreeNode;
  depth: number;
  onOpen: (path: string) => void;
  selected: string | null;
  changed: Set<string>;
}) {
  const [open, setOpen] = useState(false);
  const [children, setChildren] = useState<TreeNode[] | null>(node.children ?? null);
  const [error, setError] = useState<string | null>(null);

  const toggle = async () => {
    if (node.type === "file") {
      if (!node.sensitive) onOpen(node.path);
      return;
    }
    if (!open && children === null) {
      try {
        setChildren((await api.tree(node.path, 1)).children);
      } catch (e) {
        setError((e as Error).message);
      }
    }
    setOpen(!open);
  };

  const isChanged = changed.has(node.path);
  return (
    <li>
      <button
        onClick={toggle}
        aria-expanded={node.type === "dir" ? open : undefined}
        disabled={node.sensitive}
        title={node.sensitive ? "Secrets file — hidden from preview and from the agent" : node.path}
        className={`flex w-full items-center gap-1.5 rounded-md px-1.5 py-1 text-left text-[13px] ${
          selected === node.path ? "bg-cyan-glow/10 text-fg" : "text-fg/80 hover:bg-hover"
        } ${node.sensitive ? "cursor-not-allowed opacity-40" : ""}`}
        style={{ paddingLeft: 6 + depth * 14 }}
      >
        <span className={node.type === "dir" ? "text-cyan-glow/70" : "text-muted"}>{node.type === "dir" ? Icon.folder : Icon.file}</span>
        <span className="truncate">{node.name}</span>
        {isChanged && <span className="ml-auto h-1.5 w-1.5 shrink-0 rounded-full bg-amber-diag" title="Modified in this task" />}
      </button>
      {error && <p className="px-4 text-xs text-bad">{error}</p>}
      {open && children && (
        <ul>
          {children.map((c) => (
            <Node key={c.path} node={c} depth={depth + 1} onOpen={onOpen} selected={selected} changed={changed} />
          ))}
        </ul>
      )}
    </li>
  );
}

export default function FileExplorer({ workspace, onOpen, selected, changed, refreshKey }: {
  workspace: string | null;
  onOpen: (path: string) => void;
  selected: string | null;
  changed: Set<string>;
  refreshKey: number;
}) {
  const [root, setRoot] = useState<TreeNode[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!workspace) return;
    try {
      setRoot((await api.tree(".", 1)).children);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, [workspace]);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  if (!workspace) return <p className="p-4 text-sm text-muted">No project selected.</p>;
  if (error) return <p className="p-4 text-sm text-bad">{error}</p>;
  if (!root) return <p className="p-4 text-sm text-muted">Loading…</p>;
  if (!root.length) return <p className="p-4 text-sm text-muted">Empty directory.</p>;
  return (
    <ul className="p-1.5" aria-label="Project files">
      {root.map((n) => (
        <Node key={n.path} node={n} depth={0} onOpen={onOpen} selected={selected} changed={changed} />
      ))}
    </ul>
  );
}
