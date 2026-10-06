import { createElement, type ReactNode } from "react";
import { createLowlight } from "lowlight";
import python from "highlight.js/lib/languages/python";
import diff from "highlight.js/lib/languages/diff";
import json from "highlight.js/lib/languages/json";
import bash from "highlight.js/lib/languages/bash";

const highlighter = createLowlight({ python, diff, json, bash });
type Token = ReturnType<typeof highlighter.highlight>["children"][number];

function renderToken(token: Token, key: number): ReactNode {
  if (token.type === "text") return token.value;
  if (token.type !== "element") return null;
  const classes = token.properties.className;
  return createElement("span", {
    key,
    className: Array.isArray(classes) ? classes.join(" ") : undefined,
  }, token.children.map(renderToken));
}

export default function CodeBlock({ code, language }: { code: string; language?: string }) {
  const name = language?.toLowerCase();
  const tokens = name && highlighter.registered(name) ? highlighter.highlight(name, code).children : undefined;
  return <pre className="code-block" tabIndex={0}><code>{tokens ? tokens.map(renderToken) : code}</code></pre>;
}
