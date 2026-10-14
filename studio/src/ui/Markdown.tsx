/**
 * Small, safe Markdown renderer for chat replies. It builds React elements only: no raw
 * HTML from the model is ever injected. Highlighted code goes through highlight.js, which
 * escapes its input.
 *
 * Supported: fenced code blocks, headings (#–###), bullet and numbered lists, blockquotes,
 * paragraphs, `inline code`, **bold**, *italic* / _italic_, and [links](https://…).
 */
import { highlight } from "../lib/highlight";
import { useState, type ReactNode } from "react";
import { Icon } from "./Icons";

type Block =
  | { kind: "code"; lang: string; text: string }
  | { kind: "heading"; level: 1 | 2 | 3; text: string }
  | { kind: "ul" | "ol"; items: string[] }
  | { kind: "quote"; text: string }
  | { kind: "p"; text: string };

export function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n/g, "\n").split("\n");
  const out: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^\s*```\s*([\w+#.-]*)\s*$/);
    if (fence) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) body.push(lines[i++]);
      i++; // closing fence (or end of text while streaming/truncated)
      out.push({ kind: "code", lang: fence[1].toLowerCase(), text: body.join("\n") });
      continue;
    }
    const h = line.match(/^(#{1,3})\s+(.*)$/);
    if (h) {
      out.push({ kind: "heading", level: h[1].length as 1 | 2 | 3, text: h[2] });
      i++;
      continue;
    }
    if (/^\s*[-*+]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*[-*+]\s+/, ""));
      out.push({ kind: "ul", items });
      continue;
    }
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*\d+[.)]\s+/, ""));
      out.push({ kind: "ol", items });
      continue;
    }
    if (/^\s*>/.test(line)) {
      const q: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) q.push(lines[i++].replace(/^\s*>\s?/, ""));
      out.push({ kind: "quote", text: q.join("\n") });
      continue;
    }
    if (!line.trim()) {
      i++;
      continue;
    }
    const p: string[] = [];
    while (i < lines.length && lines[i].trim() && !/^\s*(```|#{1,3}\s|[-*+]\s|\d+[.)]\s|>)/.test(lines[i])) p.push(lines[i++]);
    out.push({ kind: "p", text: p.join("\n") });
  }
  return out;
}

const SAFE_URL = /^(https?:\/\/|mailto:)/i;

export function inline(text: string, keyBase = "i"): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\[[^\]\n]+\]\([^)\s]+\))|(\*[^*\n]+\*)|(\b_[^_\n]+_\b)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let n = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0];
    const k = `${keyBase}-${n++}`;
    if (m[1]) out.push(<code key={k}>{t.slice(1, -1)}</code>);
    else if (m[2]) out.push(<strong key={k}>{t.slice(2, -2)}</strong>);
    else if (m[3]) {
      const [, label, url] = t.match(/^\[([^\]]+)\]\(([^)]+)\)$/)!;
      out.push(SAFE_URL.test(url) ? <a key={k} href={url} target="_blank" rel="noopener noreferrer">{label}</a> : t);
    } else out.push(<em key={k}>{t.slice(1, -1)}</em>);
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function CopyButton({ text, label = "Copy", className = "" }: { text: string; label?: string; className?: string }) {
  const [done, setDone] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setDone(true);
      setTimeout(() => setDone(false), 1500);
    } catch {
      /* clipboard unavailable (insecure context) */
    }
  };
  return (
    <button onClick={copy} aria-label={done ? "Copied" : label} title={done ? "Copied" : label}
      className={`inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-xs text-muted transition hover:bg-hover hover:text-fg ${className}`}>
      {done ? Icon.check : Icon.copy}
      <span className="sr-only sm:not-sr-only">{done ? "Copied" : label}</span>
    </button>
  );
}

export function CodeBlock({ lang, text }: { lang: string; text: string }) {
  let html: string | null = null;
  try {
    html = highlight(text, lang);
  } catch {
    html = null;
  }
  return (
    <div className="overflow-hidden rounded-lg border border-line bg-surface-2/60">
      <div className="flex items-center justify-between border-b border-line px-3 py-1">
        <span className="font-mono text-[11px] text-muted">{lang || "code"}</span>
        <CopyButton text={text} />
      </div>
      <pre className="scroll-thin overflow-x-auto p-3 text-[13px] leading-relaxed">
        {html !== null ? (
          // highlight.js output: escaped source wrapped in <span class="hljs-…"> only
          <code className="hljs font-mono" dangerouslySetInnerHTML={{ __html: html }} />
        ) : (
          <code className="font-mono">{text}</code>
        )}
      </pre>
    </div>
  );
}

export default function Markdown({ text }: { text: string }) {
  const blocks = parseBlocks(text);
  return (
    <div className="prose-chat break-words text-[15px] leading-7">
      {blocks.map((b, i) => {
        const k = `b${i}`;
        switch (b.kind) {
          case "code":
            return <CodeBlock key={k} lang={b.lang} text={b.text} />;
          case "heading": {
            const H = (`h${b.level}` as "h1" | "h2" | "h3");
            return <H key={k}>{inline(b.text, k)}</H>;
          }
          case "ul":
            return <ul key={k}>{b.items.map((it, j) => <li key={j}>{inline(it, `${k}-${j}`)}</li>)}</ul>;
          case "ol":
            return <ol key={k}>{b.items.map((it, j) => <li key={j}>{inline(it, `${k}-${j}`)}</li>)}</ol>;
          case "quote":
            return <blockquote key={k}>{inline(b.text, k)}</blockquote>;
          default:
            return <p key={k} className="whitespace-pre-wrap">{inline(b.text, k)}</p>;
        }
      })}
    </div>
  );
}
