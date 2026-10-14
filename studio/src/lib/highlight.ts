/** highlight.js core with the languages INFINITY BOT shows (shared by the code viewer and chat). */
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

const LANGS = { python, typescript, javascript, json, css, xml, bash, markdown, yaml };
for (const [name, lang] of Object.entries(LANGS)) if (!hljs.getLanguage(name)) hljs.registerLanguage(name, lang);
const ALIASES: Record<string, string> = {
  py: "python", ts: "typescript", tsx: "typescript", js: "javascript", jsx: "javascript", html: "xml",
  sh: "bash", shell: "bash", zsh: "bash", console: "bash", yml: "yaml", md: "markdown",
};

export function highlight(code: string, lang?: string): string {
  const name = lang ? ALIASES[lang] ?? lang : "";
  return name && hljs.getLanguage(name) ? hljs.highlight(code, { language: name }).value : hljs.highlightAuto(code).value;
}

export default hljs;
