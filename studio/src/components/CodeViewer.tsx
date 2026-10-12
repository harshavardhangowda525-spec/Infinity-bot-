import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import css from "highlight.js/lib/languages/css";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import python from "highlight.js/lib/languages/python";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";
import { useEffect, useMemo, useState } from "react";
import { api } from "../lib/api";

hljs.registerLanguage("python", python);
hljs.registerLanguage("typescript", typescript);
hljs.registerLanguage("javascript", javascript);
hljs.registerLanguage("json", json);
hljs.registerLanguage("css", css);
hljs.registerLanguage("xml", xml);
hljs.registerLanguage("bash", bash);
hljs.registerLanguage("markdown", markdown);
hljs.registerLanguage("yaml", yaml);

const EXT: Record<string, string> = {
  py: "python", ts: "typescript", tsx: "typescript", js: "javascript", jsx: "javascript", mjs: "javascript",
  json: "json", css: "css", html: "xml", xml: "xml", svg: "xml", sh: "bash", bash: "bash", md: "markdown",
  yml: "yaml", yaml: "yaml", toml: "yaml",
};

export function languageFor(path: string): string | null {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  return EXT[ext] ?? null;
}

function highlightLines(code: string, lang: string | null): string[] {
  const html = lang ? hljs.highlight(code, { language: lang, ignoreIllegals: true }).value : escapeHtml(code);
  // Split highlighted HTML into lines while keeping open spans balanced.
  const lines: string[] = [];
  let open: string[] = [];
  for (const raw of html.split("\n")) {
    const line = open.join("") + raw;
    const tags = raw.match(/<span[^>]*>|<\/span>/g) ?? [];
    for (const t of tags) {
      if (t.startsWith("</")) open.pop();
      else open.push(t);
    }
    lines.push(line + "</span>".repeat(open.length));
  }
  return lines;
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

export function FileView({ path, refreshKey }: { path: string; refreshKey: number }) {
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    setContent(null);
    setError(null);
    api.file(path).then(
      (f) => alive && setContent(f.content),
      (e) => alive && setError((e as Error).message),
    );
    return () => {
      alive = false;
    };
  }, [path, refreshKey]);
  const lines = useMemo(() => (content === null ? [] : highlightLines(content, languageFor(path))), [content, path]);
  if (error) return <p className="p-4 text-sm text-red-300">{error}</p>;
  if (content === null) return <p className="p-4 text-sm text-slate-500">Loading…</p>;
  const width = String(lines.length).length;
  return (
    <pre className="hljs min-w-max p-3 font-mono text-[12.5px] leading-[1.6]">
      {lines.map((l, i) => (
        <div key={i} className="flex">
          <span className="mr-4 select-none text-right text-slate-600" style={{ width: `${width}ch` }}>
            {i + 1}
          </span>
          <span dangerouslySetInnerHTML={{ __html: l || " " }} />
        </div>
      ))}
    </pre>
  );
}

/** Unified diff renderer with old/new line numbers. */
export function DiffView({ diff }: { diff: string }) {
  const rows = useMemo(() => {
    const out: { kind: "add" | "del" | "ctx" | "hunk" | "meta"; text: string; a?: number; b?: number }[] = [];
    let a = 0;
    let b = 0;
    for (const line of diff.split("\n")) {
      if (line.startsWith("@@")) {
        const m = /@@ -(\d+)(?:,\d+)? \+(\d+)/.exec(line);
        if (m) {
          a = Number(m[1]);
          b = Number(m[2]);
        }
        out.push({ kind: "hunk", text: line });
      } else if (line.startsWith("+++") || line.startsWith("---")) {
        out.push({ kind: "meta", text: line });
      } else if (line.startsWith("+")) {
        out.push({ kind: "add", text: line.slice(1), b: b++ });
      } else if (line.startsWith("-")) {
        out.push({ kind: "del", text: line.slice(1), a: a++ });
      } else if (line !== "") {
        out.push({ kind: "ctx", text: line.slice(1), a: a++, b: b++ });
      }
    }
    return out;
  }, [diff]);
  return (
    <pre className="min-w-max p-2 font-mono text-[12px] leading-[1.6]">
      {rows.map((r, i) => (
        <div
          key={i}
          className={`flex ${
            r.kind === "add" ? "bg-emerald-400/10 text-emerald-200" : r.kind === "del" ? "bg-red-400/10 text-red-200" : r.kind === "hunk" ? "text-violet-soft" : r.kind === "meta" ? "text-slate-500" : "text-slate-300"
          }`}
        >
          <span className="w-9 select-none pr-1 text-right text-slate-600">{r.a ?? ""}</span>
          <span className="w-9 select-none pr-2 text-right text-slate-600">{r.b ?? ""}</span>
          <span className="w-3 select-none">{r.kind === "add" ? "+" : r.kind === "del" ? "−" : ""}</span>
          <span>{r.text || " "}</span>
        </div>
      ))}
    </pre>
  );
}
